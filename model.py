import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EncoderConvDown(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.conv = ConvBlock(out_c, out_c)

    def forward(self, x):
        x = self.down(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class GlobalMHA2D(nn.Module):
    def __init__(self, embed_dim, num_heads, Cq, Ck, Cv):
        super().__init__()
        self.q_proj = nn.Conv2d(Cq, embed_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(Ck, embed_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(Cv, embed_dim, 1, bias=False)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def _resize(self, src, target_hw):
        Ht, Wt = target_hw
        Hs, Ws = src.shape[2:]
        if (Hs, Ws) == (Ht, Wt):
            return src
        if Hs >= Ht and Ws >= Wt:
            return F.adaptive_avg_pool2d(src, (Ht, Wt))
        return F.interpolate(src, size=(Ht, Wt), mode='bilinear', align_corners=False)

    def forward(self, q_src, k_src, v_src):
        q = self.q_proj(q_src)
        k = self.k_proj(k_src)
        v = self.v_proj(v_src)
        B, E, H, W = q.shape
        k = self._resize(k, (H, W))
        v = self._resize(v, (H, W))
        q = q.flatten(2).permute(2, 0, 1)
        k = k.flatten(2).permute(2, 0, 1)
        v = v.flatten(2).permute(2, 0, 1)
        out, _ = self.mha(q, k, v)
        out = out.permute(1, 2, 0).reshape(B, E, H, W)
        return out


class EncoderMSABlock(nn.Module):
    def __init__(self, Cq, Ck, Cv, embed_dim=192, num_heads=3):
        super().__init__()
        self.attn = GlobalMHA2D(embed_dim, num_heads, Cq, Ck, Cv)
        self.out_proj = nn.Conv2d(embed_dim, Cq, 1, bias=False)
        self.norm = nn.BatchNorm2d(Cq)
        self.act = nn.ReLU(inplace=True)

    def forward(self, q_src, k_src, v_src):
        attn = self.attn(q_src, k_src, v_src)
        attn = self.out_proj(attn)
        out = q_src + attn
        out = self.norm(out)
        out = self.act(out)
        return out


class UNetMSAttnGenerator(nn.Module):
    def __init__(self, embed_dim=192, heads=3):
        super().__init__()
        self.inp = ConvBlock(1, 64)
        self.enc1 = EncoderConvDown(64, 128)
        self.enc2 = EncoderConvDown(128, 256)
        self.enc3 = EncoderConvDown(256, 512)
        self.enc4 = EncoderConvDown(512, 1024)

        self.msa4 = EncoderMSABlock(512, 1024, 256, embed_dim, heads)
        self.msa3 = EncoderMSABlock(256, 512, 128, embed_dim, heads)
        self.msa2 = EncoderMSABlock(128, 256, 64, embed_dim, heads)

        self.dec1 = Decoder(1024, 512)
        self.dec2 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec4 = Decoder(128, 64)

        self.head = nn.Sequential(
            nn.Conv2d(64, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.inp(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        A4 = self.msa4(x4, x5, x3)
        A3 = self.msa3(x3, A4, x2)
        A2 = self.msa2(x2, A3, x1)

        d1 = self.dec1(x5, A4)
        d2 = self.dec2(d1, A3)
        d3 = self.dec3(d2, A2)
        d4 = self.dec4(d3, x1)

        return self.head(d4)

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

        # Scheduler cho Generator
        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_G, mode='min', factor=0.5, patience=5, verbose=True
        )

    def set_requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        real_image = torch.cat([self.L, self.ab], dim=1)

        pred_real = self.net_D(real_image)
        pred_fake = self.net_D(fake_image.detach())

        self.loss_D_real = self.GANcriterion(pred_real, torch.ones_like(pred_real))
        self.loss_D_fake = self.GANcriterion(pred_fake, torch.zeros_like(pred_fake))
        self.loss_D = 0.5 * (self.loss_D_real + self.loss_D_fake)

        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        pred_fake = self.net_D(fake_image)

        self.loss_G_GAN = self.GANcriterion(pred_fake, torch.ones_like(pred_fake))
        self.loss_G_L1  = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def backward_G_warm_up(self):
        self.loss_G_L1  = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_L1
        self.loss_G.backward()
