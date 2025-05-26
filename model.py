import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.models import VGG16_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

# --- Residual Block with InstanceNorm and SE ---
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True)
        )
        self.se   = SEBlock(out_ch)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv(x)
        res = self.se(res)
        x   = self.proj(x)
        return self.relu(x + res)

# --- Non-local Global Context Module ---
class NonLocalBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        mid = in_ch // 2
        self.theta = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.phi   = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.g     = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.out   = nn.Conv2d(mid, in_ch, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        theta = self.theta(x).view(b, -1, h*w)             # b × c' × N
        phi   = self.phi(x).view(b, -1, h*w)               # b × c' × N
        attn  = torch.softmax(theta.transpose(1,2) @ phi, dim=-1)  # b × N × N
        g     = self.g(x).view(b, -1, h*w)                 # b × c' × N
        y     = g @ attn.transpose(1,2)                    # b × c' × N
        y     = y.view(b, -1, h, w)
        y     = self.out(y)
        return x + y

# --- U-Net Generator with SE, NonLocal, Multi-scale Fusion ---
class UNetGenerator(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        # Encoder
        self.e1 = ResBlock(in_ch, base_ch)
        self.e2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_ch,  base_ch*2))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_ch*2, base_ch*4))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_ch*4, base_ch*8))
        self.e5 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_ch*8, base_ch*8))
        # Bottleneck
        self.gc = NonLocalBlock(base_ch*8)
        # Decoder
        self.d4 = self._up_block(base_ch*8*2,  base_ch*4)
        self.d3 = self._up_block(base_ch*4+base_ch*8, base_ch*2)
        self.d2 = self._up_block(base_ch*2+base_ch*4, base_ch)
        self.d1 = self._up_block(base_ch+base_ch*2,   base_ch)
        # Multi-scale fusion
        fuse_ch = base_ch + base_ch*2 + base_ch*4 + base_ch*8  # 960
        self.fuse = nn.Conv2d(fuse_ch, base_ch*4, 1, bias=False)
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(base_ch*4, 2, 3, padding=1, bias=False),
            nn.Tanh()
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            ResBlock(out_ch, out_ch)
        )

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)

        b  = self.gc(x5)
        d4 = self.d4(torch.cat([b, x5], dim=1))
        d3 = self.d3(torch.cat([d4, x4], dim=1))
        d2 = self.d2(torch.cat([d3, x3], dim=1))
        d1 = self.d1(torch.cat([d2, x2], dim=1))

        x2u = F.interpolate(x2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        x3u = F.interpolate(x3, size=d1.shape[2:], mode='bilinear', align_corners=False)
        x4u = F.interpolate(x4, size=d1.shape[2:], mode='bilinear', align_corners=False)

        feat = self.fuse(torch.cat([d1, x2u, x3u, x4u], dim=1))
        return self.head(feat)

# --- PatchGAN Discriminator ---
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c=3, nf=64, n_down=3):
        super().__init__()
        layers = [self._block(input_c, nf, norm=False)]
        for i in range(n_down):
            ni, no = nf * 2**i, nf * 2**(i+1)
            stride = 1 if i == (n_down-1) else 2
            layers.append(self._block(ni, no, s=stride))
        layers.append(self._block(nf * 2**n_down, 1, s=1, norm=False, act=False))
        self.model = nn.Sequential(*layers)

    def _block(self, ni, no, k=4, s=2, p=1, norm=True, act=True):
        seq = [nn.Conv2d(ni, no, k, s, p, bias=not norm)]
        if norm: seq.append(nn.BatchNorm2d(no))
        if act:  seq.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)

# --- GANLoss ---
class GANLoss(nn.Module):
    def __init__(self, mode='vanilla'):
        super().__init__()
        self.register_buffer('real', torch.tensor(1.0))
        self.register_buffer('fake', torch.tensor(0.0))
        self.loss = nn.BCEWithLogitsLoss() if mode == 'vanilla' else nn.MSELoss()

    def forward(self, preds, is_real):
        label = self.real if is_real else self.fake
        return self.loss(preds, label.expand_as(preds))

# --- GAN Trainer ---
class GAN(nn.Module):
    def __init__(self,
                 lr_G=2e-4, lr_D=1e-4,
                 beta1=0.5, beta2=0.999,
                 lambda_L1=100., lambda_perc=0.1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net_G = UNetGenerator().to(self.device)
        self.net_D = PatchDiscriminator().to(self.device)

        self.GANcriterion   = GANLoss('vanilla').to(self.device)
        self.L1criterion    = nn.L1Loss()
        self.lambda_L1      = lambda_L1

        # Perceptual loss via VGG16 up to conv3_3
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        for p in vgg.parameters(): p.requires_grad = False
        self.perc_net       = vgg.to(self.device).eval()
        self.perc_criterion = nn.L1Loss()
        self.lambda_perc    = lambda_perc

        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        self.scheduler_G = ReduceLROnPlateau(self.opt_G, mode='min', factor=0.95, patience=5, verbose=True)

    def set_grad(self, model, req_grad):
        for p in model.parameters():
            p.requires_grad = req_grad

    def setup_input(self, data):
        self.L  = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake = self.net_G(self.L)

    def backward_D(self):
        fake_img = torch.cat([self.L, self.fake.detach()], dim=1)
        real_img = torch.cat([self.L, self.ab], dim=1)
        lossF = self.GANcriterion(self.net_D(fake_img), False)
        lossR = self.GANcriterion(self.net_D(real_img), True)
        self.loss_D = 0.5 * (lossF + lossR)
        self.loss_D.backward()

    def backward_G(self):
        fake_img       = torch.cat([self.L, self.fake], dim=1)
        self.loss_G_GAN = self.GANcriterion(self.net_D(fake_img), True)
        self.loss_G_L1  = self.L1criterion(self.fake, self.ab) * self.lambda_L1

        fv = (fake_img + 1) * 0.5
        rv = (torch.cat([self.L, self.ab], dim=1) + 1) * 0.5
        feat_f = self.perc_net(fv)
        feat_r = self.perc_net(rv)
        self.loss_G_perc = self.perc_criterion(feat_f, feat_r) * self.lambda_perc

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perc
        self.loss_G.backward()

    def optimize(self):
        # update D
        self.forward()
        self.net_D.train()
        self.set_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        # update G
        self.net_G.train()
        self.set_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

    def warmup_optimize(self):
        """Warm-up: only optimize G with L1 loss."""
        self.forward()
        self.net_G.train()
        self.set_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.loss_G_L1 = self.L1criterion(self.fake, self.ab) * self.lambda_L1
        self.loss_G_L1.backward()
        self.opt_G.step()

# --- Utilities to load pretrained weights ---
def get_encoder_weights(model_path='model.pth'):
    ckpt = torch.load(model_path, map_location='cpu')
    sd   = ckpt.get('model_state_dict', ckpt)
    return {k: v for k, v in sd.items() if k.split('.')[0] in ('e1','e2','e3','e4','e5')}

def load_trained_model(model_path='model.pth'):
    ckpt  = torch.load(model_path, map_location='cpu')
    model = GAN()
    try:
        model.load_state_dict(ckpt['model_state_dict'])
    except:
        model.load_state_dict(ckpt)
    return model.eval()
