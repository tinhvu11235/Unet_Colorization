import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchvision.models import VGG16_Weights

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
        theta = self.theta(x).view(b, -1, h*w)
        phi   = self.phi(x).view(b, -1, h*w)
        attn  = torch.softmax(theta.transpose(1,2) @ phi, dim=-1)
        g     = self.g(x).view(b, -1, h*w)
        y     = g @ attn.transpose(1,2)
        y     = y.view(b, -1, h, w)
        y     = self.out(y)
        return x + y

# --- Generator: ResUNet + SE + NonLocal + Multi-scale Fusion ---
class UNetGenerator(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        # Encoder
        self.e1 = ResBlock(in_ch, base_ch)
        self.e2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_ch,  base_ch*2))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_ch*2, base_ch*4))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_ch*4, base_ch*8))
        self.e5 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_ch*8, base_ch*8))
        # Bottleneck global context
        self.gc = NonLocalBlock(base_ch*8)
        # Decoder
        self.d4 = self._up_block(base_ch*8*2,  base_ch*4)
        self.d3 = self._up_block(base_ch*4+base_ch*8, base_ch*2)
        self.d2 = self._up_block(base_ch*2+base_ch*4, base_ch)
        self.d1 = self._up_block(base_ch+base_ch*2,   base_ch)
        # Multi-scale fusion
        fuse_ch = base_ch + base_ch*2 + base_ch*4 + base_ch*8  # 64+128+256+512 = 960
        self.fuse = nn.Conv2d(fuse_ch, base_ch*4, 1, bias=False)
        # Head to 2 channels
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
        d4 = self.d4(torch.cat([b,  x5], dim=1))
        d3 = self.d3(torch.cat([d4, x4], dim=1))
        d2 = self.d2(torch.cat([d3, x3], dim=1))
        d1 = self.d1(torch.cat([d2, x2], dim=1))
        x2_up = F.interpolate(x2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        x3_up = F.interpolate(x3, size=d1.shape[2:], mode='bilinear', align_corners=False)
        x4_up = F.interpolate(x4, size=d1.shape[2:], mode='bilinear', align_corners=False)
        concat = torch.cat([d1, x2_up, x3_up, x4_up], dim=1)
        feat   = self.fuse(concat)
        return self.head(feat)

# --- Utility to load encoder weights ---
def get_encoder_weights(model_path='model.pth'):
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    enc_keys = tuple(f"e{i}" for i in range(1,6))
    return {k: v for k, v in state_dict.items() if k.split('.')[0] in enc_keys}

def load_trained_model(model_path='model.pth'):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = UNetGenerator()
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model.load_state_dict(checkpoint)
    return model.eval()

# --- Discriminator and Loss ---
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        layers = [self._layer(input_c, num_filters, norm=False)]
        for i in range(n_down):
            ni = num_filters * 2**i
            nf = num_filters * 2**(i+1)
            stride = 1 if i==(n_down-1) else 2
            layers.append(self._layer(ni, nf, s=stride))
        layers.append(self._layer(num_filters*2**n_down, 1, s=1, norm=False, act=False))
        self.model = nn.Sequential(*layers)
    def _layer(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        seq = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: seq.append(nn.BatchNorm2d(nf))
        if act:  seq.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*seq)
    def forward(self, x):
        return self.model(x)

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss() if gan_mode=='vanilla' else nn.MSELoss()
    def get_labels(self, preds, target_is_real):
        lbl = self.real_label if target_is_real else self.fake_label
        return lbl.expand_as(preds)
    def forward(self, preds, target_is_real):
        return self.loss(preds, self.get_labels(preds, target_is_real))

# --- GAN Trainer with Perceptual Loss ---
class GAN(nn.Module):
    def __init__(self, lr_G=2e-4, lr_D=1e-4, beta1=0.5, beta2=0.999,
                 lambda_L1=100., lambda_perc=0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_G = UNetGenerator().to(self.device)
        self.net_D = PatchDiscriminator(input_c=3).to(self.device)
        self.GANcriterion = GANLoss('vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        # Perceptual loss: VGG16 up to conv3_3
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(self.device).eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.perc_net = vgg
        self.perc_criterion = nn.L1Loss()
        self.lambda_L1 = lambda_L1
        self.lambda_perc = lambda_perc
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        self.scheduler_G = ReduceLROnPlateau(self.opt_G, mode='min', factor=0.95, patience=5, verbose=True)

    def set_requires_grad(self, model, req_grad=True):
        for p in model.parameters(): p.requires_grad = req_grad

    def setup_input(self, data):
        self.L  = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake = torch.cat([self.L, self.fake_color.detach()], dim=1)
        pred_fake = self.net_D(fake)
        loss_D_fake = self.GANcriterion(pred_fake, False)
        real = torch.cat([self.L, self.ab], dim=1)
        pred_real = self.net_D(real)
        loss_D_real = self.GANcriterion(pred_real, True)
        self.loss_D = 0.5 * (loss_D_fake + loss_D_real)
        self.loss_D.backward()

    def backward_G(self):
        fake = torch.cat([self.L, self.fake_color], dim=1)
        pred_fake = self.net_D(fake)
        self.loss_G_GAN = self.GANcriterion(pred_fake, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        # perceptual
        fake_v = (fake + 1) * 0.5
        real_v = (torch.cat([self.L, self.ab], dim=1) + 1) * 0.5
        feat_f = self.perc_net(fake_v)
        feat_r = self.perc_net(real_v)
        self.loss_G_perc = self.perc_criterion(feat_f, feat_r) * self.lambda_perc
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perc
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        # D step
        self.net_D.train(); self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad(); self.backward_D(); self.opt_D.step()
        # G step
        self.net_G.train(); self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad(); self.backward_G(); self.opt_G.step()

# --- Pretrain Discriminator ---
def pretrain_discriminator(train_dl, gan_model, epochs=3):
    shuffled = DataLoader(
        train_dl.dataset, batch_size=train_dl.batch_size, shuffle=True,
        num_workers=getattr(train_dl, 'num_workers', 0),
        pin_memory=getattr(train_dl, 'pin_memory', False),
        drop_last=getattr(train_dl, 'drop_last', False),
        collate_fn=getattr(train_dl, 'collate_fn', None)
    )
    gan_model.net_D.train()
    gan_model.set_requires_grad(gan_model.net_D, True)
    for e in range(epochs):
        run, rl, fl = 0.0, 0.0, 0.0
        for data in shuffled:
            gan_model.setup_input(data)
            with torch.no_grad(): gan_model.forward()
            gan_model.opt_D.zero_grad(); gan_model.backward_D(); gan_model.opt_D.step()
            run += gan_model.loss_D.item()
        print(f"Epoch {e+1}/{epochs} D loss: {run/len(shuffled):.4f}")
