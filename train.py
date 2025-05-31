import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.checkpoint as cp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import transforms
from skimage.color import lab2rgb
from einops import rearrange
import warnings


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding cho tensor [B, C, H, W]."""
    def __init__(self, embed_dim, height, width):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be divisible by 2")
        d_half = embed_dim // 2
        # PE cho trục H
        pos_h = torch.arange(height).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_half, 2) * -(math.log(10000.0) / d_half))
        pe_h = torch.zeros(height, d_half)
        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term)
        # PE cho trục W
        pos_w = torch.arange(width).unsqueeze(1)
        pe_w = torch.zeros(width, d_half)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term)
        # Ghép PE
        pe = torch.zeros(height, width, embed_dim)
        for y in range(height):
            pe[y, :, :d_half] = pe_h[y]
        for x in range(width):
            pe[:, x, d_half:] = pe_w[x]
        pe = pe.permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe


class WindowAttention2D(nn.Module):
    """Window-based multi-head self-attention."""
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def forward(self, q, k, v):
        B, E, H, W = q.shape
        ws = self.window_size
        q_win = rearrange(q, 'b e (h ws1) (w ws2) -> (b h w) (ws1 ws2) e', ws1=ws, ws2=ws)
        k_win = rearrange(k, 'b e (h ws1) (w ws2) -> (b h w) (ws1 ws2) e', ws1=ws, ws2=ws)
        v_win = rearrange(v, 'b e (h ws1) (w ws2) -> (b h w) (ws1 ws2) e', ws1=ws, ws2=ws)
        q2, k2, v2 = q_win.permute(1, 0, 2), k_win.permute(1, 0, 2), v_win.permute(1, 0, 2)
        attn_out, _ = self.mha(q2, k2, v2)
        attn_out = attn_out.permute(1, 0, 2)
        out = rearrange(
            attn_out,
            '(b h w) (ws1 ws2) e -> b e (h ws1) (w ws2)',
            b=B, h=H // ws, w=W // ws, ws1=ws, ws2=ws
        )
        return out


class AttentionGate(nn.Module):
    """Attention gate cho skip-connection."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, 1, bias=False)
        self.psi = nn.Conv2d(F_int, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class ConvBlock(nn.Module):
    """Hai lớp Conv2d + BatchNorm + ReLU."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Encoder U-Net: MaxPool + ConvBlock."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x):
        return self.conv(self.pool(x))


class Decoder(nn.Module):
    """Decoder U-Net: ConvTranspose + AttentionGate + ConvBlock."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)
        self.gate = AttentionGate(out_c, out_c, out_c // 2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Áp dụng attention gate
        skip = self.gate(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class DecoderNoGate(nn.Module):
    """Decoder U-Net: ConvTranspose + ConvBlock (không có AttentionGate)."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Không sử dụng attention gate, concat trực tiếp
        return self.conv(torch.cat([x, skip], dim=1))


class NonLocalBlock(nn.Module):
    """Non-local global context (self-attention toàn cục)."""
    def __init__(self, in_ch):
        super().__init__()
        mid = in_ch // 2
        self.theta = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.phi   = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.g     = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.out   = nn.Conv2d(mid, in_ch, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        # Theta, Phi, G projection
        theta = self.theta(x).view(b, -1, h*w)  # (B, mid, N)
        phi   = self.phi(x).view(b, -1, h*w)    # (B, mid, N)
        attn  = torch.softmax(theta.transpose(1, 2) @ phi, dim=-1)  # (B, N, N)
        g     = self.g(x).view(b, -1, h*w)      # (B, mid, N)
        # Weighted sum
        y     = g @ attn.transpose(1, 2)        # (B, mid, N)
        y     = y.view(b, -1, h, w)             # (B, mid, H, W)
        # Project back và cộng residual
        return x + self.out(y)


class MultiScaleAttentionBlock(nn.Module):
    """Khối attention đa quy mô: projections → WindowAttention → fusion."""
    def __init__(self, in_ch, embed_dim, heads, window_size, save_mode=True):
        super().__init__()
        self.embed_dim, self.ws, self.save = embed_dim, window_size, save_mode
        self.kp = nn.Conv2d(in_ch[0], embed_dim, 1, bias=False)
        self.qp = nn.Conv2d(in_ch[1], embed_dim, 1, bias=False)
        self.vp = nn.Conv2d(in_ch[2], embed_dim, 1, bias=False)
        self.pos_enc = None
        self.attn = WindowAttention2D(embed_dim, heads, window_size)
        self.sp = nn.Conv2d(in_ch[1], embed_dim, 1, bias=False)
        self.post = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        x5, d1, d2 = feats
        B, _, H, W = d1.shape
        # Khởi tạo PositionalEncoding2D nếu cần
        if self.pos_enc is None or self.pos_enc.pe.shape[2:] != (H, W):
            self.pos_enc = PositionalEncoding2D(self.embed_dim, H, W).to(d1.device)

        def prep(x, proj):
            y = proj(x)
            if y.shape[2:] != (H, W):
                y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            return self.pos_enc(y)

        q, k, v = prep(d1, self.qp), prep(x5, self.kp), prep(d2, self.vp)
        attn_out = cp.checkpoint(self.attn, q, k, v) if self.save else self.attn(q, k, v)
        skip = self.sp(d1)
        if skip.shape[2:] != (H, W):
            skip = F.interpolate(skip, size=(H, W), mode='bilinear', align_corners=False)
        return self.post(torch.cat([attn_out, skip], dim=1))


class UNetMSAttnGenerator(nn.Module):
    def __init__(self, save_mode=True, window_size=8, fpn_channels=64):
        super().__init__()
        # ---------------------- Encoder ----------------------
        self.inp = ConvBlock(1, 64)        # x1: (B,64,256,256)
        self.enc1 = Encoder(64, 128)       # x2: (B,128,128,128)
        self.enc2 = Encoder(128, 256)      # x3: (B,256, 64, 64)
        self.enc3 = Encoder(256, 512)      # x4: (B,512, 32, 32)
        self.enc4 = Encoder(512, 1024)     # x5: (B,1024,16, 16)

        # ------------ NonLocal Global Context -------------
        # Đổi tên thành `self.nl_block` để tránh trùng với từ khóa
        self.nl_block = NonLocalBlock(1024)  # áp vào x5

        # ---------------------- Decoder ----------------------
        # 2 tầng giải mã sâu (có attention gate)
        self.dec1 = Decoder(1024, 512)     # d1: (B,512,32,32)
        self.dec2 = Decoder(512, 256)      # d2: (B,256,64,64)
        # 2 tầng giải mã nông (không attention gate)
        self.dec3 = DecoderNoGate(256, 128)  # d3: (B,128,128,128)
        self.dec4 = DecoderNoGate(128, 64)   # d4: (B, 64,256,256)

        # ------------ Multi-Scale Attention Cascade -------------
        self.msa1 = MultiScaleAttentionBlock([1024, 512, 256], 192, 3, window_size, save_mode)
        self.msa2 = MultiScaleAttentionBlock([192, 256, 128], 192, 3, window_size, save_mode)
        self.msa3 = MultiScaleAttentionBlock([192, 128, 64], 192, 3, window_size, save_mode)

        # ---------------------- FPN (Multi-Scale Fusion) ----------------------
        # Lateral 1×1 conv cho x5, x4, x3, x2 → fpn_channels kênh
        self.lat5 = nn.Sequential(
            nn.Conv2d(1024, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.lat4 = nn.Sequential(
            nn.Conv2d(512, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.lat3 = nn.Sequential(
            nn.Conv2d(256, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.lat2 = nn.Sequential(
            nn.Conv2d(128, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        # Không cần khối fuse tại từng bước, vì ta dùng add fusion (top-down)

        # ---------------------- Head Cuối cùng ----------------------
        # Input: d4 (64 kênh) + m3_up (192 kênh) + fpn_up (64 kênh) = 320 kênh
        # Qua Conv3×3 → 2 kênh ab → Tanh
        self.head = nn.Sequential(
            nn.Conv2d(64 + 192 + fpn_channels, 2, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        # ---------------------- Encode ----------------------
        x1 = self.inp(x)           # (B, 64, 256, 256)
        x2 = self.enc1(x1)         # (B,128,128,128)
        x3 = self.enc2(x2)         # (B,256, 64, 64)
        x4 = self.enc3(x3)         # (B,512, 32, 32)
        x5 = self.enc4(x4)         # (B,1024,16, 16)

        # ------------ NonLocal Global Context -------------
        nl = self.nl_block(x5)     # (B,1024,16,16)

        # ---------------------- Decode ----------------------
        d1 = self.dec1(nl, x4)     # (B, 512, 32, 32)
        d2 = self.dec2(d1, x3)     # (B, 256, 64, 64)
        d3 = self.dec3(d2, x2)     # (B, 128,128,128)
        d4 = self.dec4(d3, x1)     # (B,  64,256,256)

        # ------------ Multi-Scale Attention Cascade -------------
        m1 = self.msa1([x5, d1, d2])  # (B,192,32,32)
        m2 = self.msa2([m1, d2, d3])  # (B,192,64,64)
        m3 = self.msa3([m2, d3, d4])  # (B,192,128,128)

        # 5) Upsample m3 về 256×256
        m3_up = F.interpolate(m3, size=d4.shape[2:], mode='bilinear', align_corners=False)  # (B,192,256,256)

        # ---------------------- FPN Top-Down Add Fusion ----------------------
        p5 = self.lat5(x5)     # (B,64,16,16)
        p4 = self.lat4(x4) + F.interpolate(p5, size=x4.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.relu(p4)        # (B,64,32,32)
        p3 = self.lat3(x3) + F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.relu(p3)        # (B,64,64,64)
        p2 = self.lat2(x2) + F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        p2 = F.relu(p2)        # (B,64,128,128)

        # Upsample p2 lên 256×256 (giống độ phân giải d4/m3_up)
        fpn_up = F.interpolate(p2, size=d4.shape[2:], mode='bilinear', align_corners=False)  # (B,64,256,256)

        # ---------------------- Head Kết hợp ----------------------
        feat = torch.cat([d4, m3_up, fpn_up], dim=1)  # (B, 64 + 192 + 64 = 320, 256, 256)
        out = self.head(feat)                         # (B, 2, 256, 256)
        return out



def init_weights_Generator(m):
    """Khởi tạo weights cho generator."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [
            self.get_layers(
                num_filters * 2 ** i,
                num_filters * 2 ** (i + 1),
                s=1 if i == (n_down - 1) else 2
            )
            for i in range(n_down)
        ]
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(nf))
        if act:
            layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        return self

    def forward(self, x):
        return self.model(x)

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.       # từ [-1,1] → [0,100]
    ab = ab * 110.           # từ [-1,1] → [-110,110]
    Lab = np.concatenate([L, ab], axis=0).transpose(1, 2, 0)  # (3,H,W) → (H,W,3)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rgb = lab2rgb(Lab)   # float64 ∈ [0,1]

    return rgb

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        return (self.real_label if target_is_real else self.fake_label).expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        return self.loss(preds, labels)


class GAN(nn.Module):
    def __init__(self, lr_G=2e-4, lr_D=1e-4, beta1=0.5, beta2=0.999,
                 lambda_L1=100.0, lambda_perc=0.01):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hệ số loss
        self.lambda_L1 = lambda_L1
        self.lambda_perc = lambda_perc

        # Generator & Discriminator
        self.net_G = UNetMSAttnGenerator(save_mode=True, window_size=8).to(self.device)\
                          .apply(init_weights_Generator)
        self.net_D = PatchDiscriminator(input_c=3).init_weights().to(self.device)

        # Criterion
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()

        # Optimizers & Scheduler
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        self.scheduler_G = ReduceLROnPlateau(self.opt_G, mode='min',
                                             factor=0.95, patience=5, verbose=True)

        # ---- PHẦN PERCEPTUAL LOSS ----
        # Load VGG16 pretrained, chỉ lấy các layer đầu tiên (đến idx 15 cho relu3_3)
        vgg_full = torchvision.models.vgg16(pretrained=True).features
        self.vgg = nn.Sequential(*list(vgg_full.children())[:16]).to(self.device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Normalize ImageNet: input RGB ∈ [0,1]
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # -------------------------------

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        # L và ab ∈ [-1,1]
        self.L = data['L'].to(self.device)      # (B,1,H,W)
        self.ab = data['ab'].to(self.device)    # (B,2,H,W)

    def forward(self):
        self.fake_color = self.net_G(self.L)    # (B,2,H,W)

    def backward_D(self, noGAN=False):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)  # (B,3,H,W)
        fake_preds = self.net_D(fake_image.detach())
        if noGAN:
            self.loss_D_fake = torch.tensor(0.0, device=self.L.device)
            self.loss_D_real = torch.tensor(0.0, device=self.L.device)
            self.loss_D = torch.tensor(0.0, device=self.L.device)
            return

        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        self.loss_D.backward()

    def backward_G(self):
        # 1) GAN loss
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)

        # 2) L1 loss pixel-wise
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1

        # 3) Perceptual Loss trên VGG16
        self.loss_G_perc = self._perceptual_loss(self.fake_color, self.ab) * self.lambda_perc

        # Tổng loss G
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perc
        self.loss_G.backward()

    def warmup_optimize(self):
        self.forward()
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G_L1.backward()
        self.opt_G.step()

    def optimize(self):
        # --- Update Discriminator ---
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D(noGAN=False)
        self.opt_D.step()

        # --- Update Generator ---
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

    # ---------------------------------------------------------
    # HÀM TÍNH PERCEPTUAL LOSS
    # ---------------------------------------------------------
    def _perceptual_loss(self, fake_ab, real_ab):
        """
        fake_ab, real_ab: (B,2,H,W) ∈ [-1,1]
        self.L: (B,1,H,W) ∈ [-1,1]
        Trả về: tổng L1 giữa feature maps ở hai layer [8,15] của VGG16
        """
        B, _, H, W = fake_ab.shape
        fake_rgb_list = []
        real_rgb_list = []

        # Chuyển từng ảnh batch từ Lab → RGB, rồi sang tensor [0,1]
        L_cpu = self.L.detach().cpu().numpy()           # (B,1,H,W)
        fake_ab_cpu = fake_ab.detach().cpu().numpy()     # (B,2,H,W)
        real_ab_cpu = real_ab.detach().cpu().numpy()

        for i in range(B):
            L_i       = L_cpu[i]       # (1,H,W)
            fake_ab_i = fake_ab_cpu[i] # (2,H,W)
            real_ab_i = real_ab_cpu[i]

            # Lab → RGB (numpy float64 [0,1])
            fake_rgb_np = lab_to_rgb(L_i, fake_ab_i)   # (H,W,3)
            real_rgb_np = lab_to_rgb(L_i, real_ab_i)   # (H,W,3)

            # Chuyển sang tensor (3,H,W) ∈ [0,1], đưa lên device
            fake_rgb_tensor = torch.from_numpy(fake_rgb_np.transpose(2,0,1)).unsqueeze(0)\
                                    .float().to(self.device)
            real_rgb_tensor = torch.from_numpy(real_rgb_np.transpose(2,0,1)).unsqueeze(0)\
                                    .float().to(self.device)

            fake_rgb_list.append(fake_rgb_tensor)
            real_rgb_list.append(real_rgb_tensor)

        fake_rgb_batch = torch.cat(fake_rgb_list, dim=0)  # (B,3,H,W)
        real_rgb_batch = torch.cat(real_rgb_list, dim=0)

        # Normalize theo ImageNet
        fake_norm = torch.zeros_like(fake_rgb_batch)
        real_norm = torch.zeros_like(real_rgb_batch)
        for j in range(B):
            fake_norm[j] = self.normalize(fake_rgb_batch[j])
            real_norm[j] = self.normalize(real_rgb_batch[j])

        # Feed qua VGG, tính L1 giữa feature maps tại layer idx 8 và 15
        loss_perc = torch.tensor(0.0, device=self.device)
        x_fake = fake_norm
        x_real = real_norm
        for idx, layer in enumerate(self.vgg):
            x_fake = layer(x_fake)
            x_real = layer(x_real)
            if idx in [8, 15]:  # 8: relu2_2, 15: relu3_3
                loss_perc = loss_perc + F.l1_loss(x_fake, x_real)

        return loss_perc

def pretrain_discriminator(train_dl, gan_model, lr=2e-5, epochs=3):
    print("Pretraining Discriminator...")

    shuffled_train_dl = DataLoader(
        dataset=train_dl.dataset,
        batch_size=train_dl.batch_size,
        shuffle=True,
        num_workers=train_dl.num_workers,
        pin_memory=train_dl.pin_memory,
        drop_last=train_dl.drop_last if hasattr(train_dl, 'drop_last') else False,
        collate_fn=train_dl.collate_fn if hasattr(train_dl, 'collate_fn') else None
    )

    gan_model.net_D.train()
    gan_model.set_requires_grad(gan_model.net_D, True)

    for epoch in range(epochs):
        running_loss = 0.0
        real_loss = 0.0
        fake_loss = 0.0

        for data in shuffled_train_dl:
            gan_model.setup_input(data)
            with torch.no_grad():
                gan_model.forward()
                gan_model.opt_D.zero_grad()
                gan_model.backward_D()
                gan_model.opt_D.step()

                running_loss += gan_model.loss_D.item()
                real_loss += gan_model.loss_D_real.item()
                fake_loss += gan_model.loss_D_fake.item()

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Running Loss: {running_loss / len(shuffled_train_dl):.4f}, "
              f"Real Loss: {real_loss / len(shuffled_train_dl):.4f}, "
              f"Fake Loss: {fake_loss / len(shuffled_train_dl):.4f}")

    print("Pretraining for Discriminator is complete.")
