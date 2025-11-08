import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from einops import rearrange
from skimage.color import lab2rgb
import warnings


# ======== POSITIONAL ENCODING ========
class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, height, width):
        super().__init__()
        d_half = embed_dim // 2
        pos_h = torch.arange(height).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_half, 2) * -(math.log(10000.0) / d_half))
        pe_h = torch.zeros(height, d_half)
        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term)

        pos_w = torch.arange(width).unsqueeze(1)
        pe_w = torch.zeros(width, d_half)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term)

        pe = torch.zeros(height, width, embed_dim)
        for y in range(height):
            pe[y, :, :d_half] = pe_h[y]
        for x in range(width):
            pe[:, x, d_half:] = pe_w[x]
        pe = pe.permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe


# ======== WINDOW ATTENTION ========
class WindowAttention2D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, shift=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def forward(self, q, k, v):
        B, E, H, W = q.shape
        ws, ss = self.window_size, self.shift_size

        if ss > 0:
            q = torch.roll(q, shifts=(-ss, -ss), dims=(2, 3))
            k = torch.roll(k, shifts=(-ss, -ss), dims=(2, 3))
            v = torch.roll(v, shifts=(-ss, -ss), dims=(2, 3))

        q_win = rearrange(q, 'b e (h ws1) (w ws2) -> (b h w) (ws1 ws2) e', ws1=ws, ws2=ws)
        k_win = rearrange(k, 'b e (h ws1) (w ws2) -> (b h w) (ws1 ws2) e', ws1=ws, ws2=ws)
        v_win = rearrange(v, 'b e (h ws1) (w ws2) -> (b h w) (ws1 ws2) e', ws1=ws, ws2=ws)

        q2, k2, v2 = q_win.permute(1, 0, 2), k_win.permute(1, 0, 2), v_win.permute(1, 0, 2)
        attn_out, _ = self.mha(q2, k2, v2)
        attn_out = attn_out.permute(1, 0, 2)
        out = rearrange(attn_out, '(b h w) (ws1 ws2) e -> b e (h ws1) (w ws2)',
                        b=B, h=H // ws, w=W // ws, ws1=ws, ws2=ws)

        if ss > 0:
            out = torch.roll(out, shifts=(ss, ss), dims=(2, 3))
        return out


# ======== CONV BLOCKS ========
class ConvBlock(nn.Module):
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
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x):
        return self.conv(self.pool(x))


class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ======== MULTI-SCALE ATTENTION ========
class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, in_ch, embed_dim, heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
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
        if self.pos_enc is None or self.pos_enc.pe.shape[2:] != (H, W):
            self.pos_enc = PositionalEncoding2D(self.embed_dim, H, W).to(d1.device)

        def prep(x, proj):
            y = proj(x)
            if y.shape[2:] != (H, W):
                y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            return self.pos_enc(y)

        q, k, v = prep(d1, self.qp), prep(x5, self.kp), prep(d2, self.vp)
        attn_out = self.attn(q, k, v)
        skip = self.sp(d1)
        return self.post(torch.cat([attn_out, skip], dim=1))


# ======== GENERATOR ========
class UNetMSAttnGenerator(nn.Module):
    def __init__(self, window_size=8):
        super().__init__()
        self.inp = ConvBlock(1, 64)
        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024)
        self.dec1 = Decoder(1024, 512)
        self.dec2 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec4 = Decoder(128, 64)
        self.msa1 = MultiScaleAttentionBlock([1024, 512, 256], 192, 3, window_size)
        self.msa2 = MultiScaleAttentionBlock([192, 256, 128], 192, 3, window_size)
        self.msa3 = MultiScaleAttentionBlock([192, 128, 64], 192, 3, window_size)
        self.m3_up = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2)
        self.head = nn.Sequential(
            nn.Conv2d(64 + 192, 128, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.inp(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        d1 = self.dec1(x5, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)
        m1 = self.msa1([x5, d1, d2])
        m2 = self.msa2([m1, d2, d3])
        m3 = self.msa3([m2, d3, d4])
        m3_up = self.m3_up(m3)
        feat = torch.cat([d4, m3_up], dim=1)
        return self.head(feat)


# ======== INITIALIZATION ========
def init_weights_Generator(m):
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


# ======== DISCRIMINATOR ========
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1),
                                  s=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]
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


# ======== LAB to RGB UTILITY ========
def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = np.concatenate([L, ab], axis=0).transpose(1, 2, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rgb = lab2rgb(Lab)
    return rgb


# ======== GAN LOSS ========
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


# ======== GAN MODEL ========
class GAN(nn.Module):
    def __init__(self, lr_G=2e-4, lr_D=1e-4, beta1=0.5, beta2=0.999, lambda_L1=100.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.net_G = UNetMSAttnGenerator().to(self.device).apply(init_weights_Generator)
        self.net_D = PatchDiscriminator(input_c=3).init_weights().to(self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        self.scheduler_G = ReduceLROnPlateau(self.opt_G, mode='min', factor=0.95, patience=5, verbose=True)

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self, noGAN=False):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        if noGAN:
            self.loss_D = torch.tensor(0.0, device=self.L.device)
            return
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
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
        self.forward()
        # Train Discriminator
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D(noGAN=False)
        self.opt_D.step()
        # Train Generator
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
