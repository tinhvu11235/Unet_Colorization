import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# CONV BLOCKS
# =========================
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
        return self.conv(self.pool(x))  # (B,out_c,H/2,W/2)


class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)
        self.conv = ConvBlock(in_c, out_c)
    def forward(self, x, skip):
        x = self.up(x)  # (B,out_c,H*2,W*2)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))  # (B,out_c,H,W)


# =========================
# GLOBAL MHA (with resize fix)
# =========================
class GlobalMHA2D(nn.Module):
    def __init__(self, embed_dim, num_heads, qdim, kdim, vdim, kv_downsample=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.kv_downsample = kv_downsample
        self.q_proj = nn.Conv2d(qdim, embed_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(kdim, embed_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(vdim, embed_dim, 1, bias=False)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def _match_v_to_k(self, v, k):
        Hk, Wk = k.shape[2:]
        Hv, Wv = v.shape[2:]
        if (Hv, Wv) == (Hk, Wk):
            return v
        if Hv >= Hk and Wv >= Wk:
            return F.adaptive_avg_pool2d(v, (Hk, Wk))
        else:
            return F.interpolate(v, size=(Hk, Wk), mode='bilinear', align_corners=False)

    def forward(self, q_src, k_src, v_src):
        q = self.q_proj(q_src)  # (B,E,Hq,Wq)
        k = self.k_proj(k_src)  # (B,E,Hk,Wk)
        v = self.v_proj(v_src)  # (B,E,Hv,Wv)
        if self.kv_downsample > 1:
            k = F.avg_pool2d(k, self.kv_downsample, self.kv_downsample)
            v = F.avg_pool2d(v, self.kv_downsample, self.kv_downsample)
        v = self._match_v_to_k(v, k)
        B, E, Hq, Wq = q.shape
        q_seq = q.flatten(2).permute(2, 0, 1)
        k_seq = k.flatten(2).permute(2, 0, 1)
        v_seq = v.flatten(2).permute(2, 0, 1)
        out, _ = self.mha(q_seq, k_seq, v_seq)
        return out.permute(1, 2, 0).reshape(B, E, Hq, Wq)  # (B,E,Hq,Wq)


# =========================
# MULTI-SCALE ATTENTION
# =========================
class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, in_ch, embed_dim, heads, kv_downsample=1):
        super().__init__()
        Ck, Cq, Cv = in_ch
        self.attn = GlobalMHA2D(embed_dim, heads, Cq, Ck, Cv, kv_downsample)
        self.skip_proj = nn.Conv2d(Cq, embed_dim, 1, bias=False)
        self.post = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, feats):
        k_src, q_src, v_src = feats
        attn_out = self.attn(q_src, k_src, v_src)   # (B,E,Hq,Wq)
        skip = self.skip_proj(q_src)                # (B,E,Hq,Wq)
        return self.post(torch.cat([attn_out, skip], dim=1))  # (B,E,Hq,Wq)


# =========================
# GENERATOR
# =========================
class UNetMSAttnGenerator(nn.Module):
    def __init__(self, embed_dim=192, heads=3, kv_downsample=2):
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
        self.msa1 = MultiScaleAttentionBlock([1024, 512, 256], embed_dim, heads, kv_downsample)
        self.msa2 = MultiScaleAttentionBlock([embed_dim, 256, 128], embed_dim, heads, kv_downsample)
        self.msa3 = MultiScaleAttentionBlock([embed_dim, 128, 64], embed_dim, heads, kv_downsample)
        self.m3_up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.head = nn.Sequential(
            nn.Conv2d(64 + embed_dim, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.inp(x)      # (B,64,256,256)
        x2 = self.enc1(x1)    # (B,128,128,128)
        x3 = self.enc2(x2)    # (B,256,64,64)
        x4 = self.enc3(x3)    # (B,512,32,32)
        x5 = self.enc4(x4)    # (B,1024,16,16)
        d1 = self.dec1(x5, x4)  # (B,512,32,32)
        d2 = self.dec2(d1, x3)  # (B,256,64,64)
        d3 = self.dec3(d2, x2)  # (B,128,128,128)
        d4 = self.dec4(d3, x1)  # (B,64,256,256)
        m1 = self.msa1([x5, d1, d2])  # (B,192,32,32)
        m2 = self.msa2([m1, d2, d3])  # (B,192,64,64)
        m3 = self.msa3([m2, d3, d4])  # (B,192,128,128)
        m3_up = self.m3_up(m3)        # (B,192,256,256)
        feat = torch.cat([d4, m3_up], dim=1)  # (B,64+192,256,256)
        return self.head(feat)        # (B,2,256,256)


# =========================
# DISCRIMINATOR
# =========================
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c=3, num_filters=64, n_down=3):
        super().__init__()
        model = [self._block(input_c, num_filters, norm=False)]
        for i in range(1, n_down + 1):
            in_f = num_filters * 2 ** (i - 1)
            out_f = num_filters * 2 ** i
            stride = 1 if i == n_down else 2
            model += [self._block(in_f, out_f, s=stride)]
        model += [nn.Conv2d(out_f, 1, 4, 1, 1)]
        self.model = nn.Sequential(*model)
    def _block(self, ni, nf, k=4, s=2, p=1, norm=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers.append(nn.BatchNorm2d(nf))
        layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)  # (B,1,H/2^(n_down),W/2^(n_down))


# =========================
# GAN MODEL
# =========================
class GAN(nn.Module):
    def __init__(self, lr_G=2e-4, lr_D=1e-4, beta1=0.5, beta2=0.999, lambda_L1=100.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.net_G = UNetMSAttnGenerator().to(self.device)
        self.net_D = PatchDiscriminator(3).to(self.device)
        self.GANcriterion = nn.BCEWithLogitsLoss()
        self.L1criterion = nn.L1Loss()
        self.opt_G = torch.optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = torch.optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def setup_input(self, data):
        self.L = data['L'].to(self.device)      # (B,1,256,256)
        self.ab = data['ab'].to(self.device)    # (B,2,256,256)

    def forward(self):
        self.fake_color = self.net_G(self.L)    # (B,2,256,256)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        real_image = torch.cat([self.L, self.ab], dim=1)
        pred_real = self.net_D(real_image)
        pred_fake = self.net_D(fake_image.detach())
        loss_real = self.GANcriterion(pred_real, torch.ones_like(pred_real))
        loss_fake = self.GANcriterion(pred_fake, torch.zeros_like(pred_fake))
        self.loss_D = 0.5 * (loss_real + loss_fake)
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        pred_fake = self.net_D(fake_image)
        loss_G_GAN = self.GANcriterion(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = loss_G_GAN + loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
