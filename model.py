import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as cp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from einops import rearrange


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
        q2, k2, v2 = q_win.permute(1,0,2), k_win.permute(1,0,2), v_win.permute(1,0,2)
        attn_out, _ = self.mha(q2, k2, v2)
        attn_out = attn_out.permute(1,0,2)
        out = rearrange(
            attn_out,
            '(b h w) (ws1 ws2) e -> b e (h ws1) (w ws2)',
            b=B, h=H//ws, w=W//ws, ws1=ws, ws2=ws
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
        self.gate = AttentionGate(out_c, out_c, out_c//2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip = self.gate(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


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
            nn.Conv2d(embed_dim*2, embed_dim, 3, padding=1, bias=False),
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
        attn_out = cp.checkpoint(self.attn, q, k, v) if self.save else self.attn(q, k, v)
        skip = self.sp(d1)
        if skip.shape[2:] != (H, W):
            skip = F.interpolate(skip, size=(H, W), mode='bilinear', align_corners=False)
        return self.post(torch.cat([attn_out, skip], dim=1))


class UNetMSAttnGenerator(nn.Module):
 
    def __init__(self, save_mode=True, window_size=8): 
        super().__init__()
        self.inp  = ConvBlock(1, 64)
        self.enc1 = Encoder(64,128);  self.enc2 = Encoder(128,256)
        self.enc3 = Encoder(256,512); self.enc4 = Encoder(512,1024)
        self.dec1 = Decoder(1024,512); self.dec2 = Decoder(512,256)
        self.dec3 = Decoder(256,128);  self.dec4 = Decoder(128,64)
        self.msa1 = MultiScaleAttentionBlock([1024,512,256], 192, 3, window_size, save_mode)
        self.msa2 = MultiScaleAttentionBlock([192,256,128], 192, 3, window_size, save_mode)
        self.msa3 = MultiScaleAttentionBlock([192,128,64],  192, 3, window_size, save_mode)
        # Head mới nhận 192+64 = 256 channels
        self.head = nn.Conv2d(256, 2, 1)

    def forward(self, x):
        # 1) Encode
        x1 = self.inp(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        # 2) Decode
        d1 = self.dec1(x5, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)  # → B×64×256×256

        # 3) Multi-scale attention cascade
        m1 = self.msa1([x5, d1, d2])
        m2 = self.msa2([m1, d2, d3])
        m3 = self.msa3([m2, d3, d4])  # → B×192×128×128

        # 4) Upsample m3 về 256×256
        m3_up = F.interpolate(m3, size=d4.shape[2:], mode='bilinear', align_corners=False)  

        # 5) Concatenate với d4
        feat = torch.cat([m3_up, d4], dim=1)  # B×256×256×256

        # 6) Head → 2 channels + tanh
        out = torch.tanh(self.head(feat))      # B×2×256×256
        return out

def init_weights_Generator(m):
        # Khởi tạo Conv, ConvTranspose, Linear
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Khởi tạo BatchNorm
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        # Khởi tạo LayerNorm
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2) 
                  for i in range(n_down)]
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers.append(nn.BatchNorm2d(nf))
        if act: layers.append(nn.LeakyReLU(0.2, True))
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
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


class GAN(nn.Module):
    def __init__(self, lr_G=2e-4, lr_D=1e-4, beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.net_G =  UNetMSAttnGenerator(save_mode=True, window_size=8).to(self.device).apply(init_weights_Generator)
        self.net_D = PatchDiscriminator(input_c=3).init_weights().to(self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        self.scheduler_G = ReduceLROnPlateau(self.opt_G,mode='min',factor=0.95,patience=5,verbose=True)
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self, noGAN = False):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
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
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
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
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D(noGAN = False)
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

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

