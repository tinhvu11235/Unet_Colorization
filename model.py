import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# 1. Conv block & Encoder
# ======================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)
    def forward(self, x):
        return self.conv(self.pool(x))

# ======================
# 2. Global context VAE
# ======================
class GlobalContextVAE(nn.Module):
    """
    Lấy feature sâu nhất (B, C, H, W) -> global vector
    -> MLP ra [mu, logvar] (cùng dim z_dim)
    -> reparameterization trick: z = mu + eps * std
    """
    def __init__(self, in_ch, hidden=512, z_dim=256, use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        if use_attn:
            # simple learned query cho attention global
            self.query = nn.Parameter(torch.randn(1, in_ch, 1, 1))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * z_dim, 1)   # -> mu + logvar
        )

    def forward(self, f):
        B, C, H, W = f.shape

        if self.use_attn:
            # attention scalar trên spatial
            attn = torch.sum(f * self.query, dim=1, keepdim=True)   # (B,1,H,W)
            attn = torch.softmax(attn.flatten(1), dim=1).view(B, 1, H, W)
            f = f * attn

        h = self.pool(f)                  # (B, C, 1, 1)
        h = self.mlp(h).flatten(1)        # (B, 2*z_dim)
        mu, logvar = torch.chunk(h, 2, dim=1)   # mỗi cái (B, z_dim)

        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                # (B, z_dim)

        return z, mu, logvar

# ======================
# 3. FiLM blocks
# ======================
class FiLM(nn.Module):
    def __init__(self, feat_ch, z_dim):
        super().__init__()
        self.gamma = nn.Linear(z_dim, feat_ch)
        self.beta  = nn.Linear(z_dim, feat_ch)
        # Bạn ban đầu set zero init ở đây
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x, z):
        # x: (B, C, H, W), z: (B, z_dim)
        g = self.gamma(z).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        b = self.beta(z).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return x * (1.0 + g) + b

class DecoderFiLM(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim, film_on_skip=False, skip_channels=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels)
        self.film = FiLM(out_channels, z_dim)
        self.film_on_skip = film_on_skip
        if self.film_on_skip:
            assert skip_channels is not None, "Need skip_channels when film_on_skip=True"
            self.film_skip = FiLM(skip_channels, z_dim)

    def forward(self, prev_output, skip_output, z):
        x = self.up(prev_output)
        # align spatial size nếu lệch 1–2 pixel
        if x.size(-1) != skip_output.size(-1) or x.size(-2) != skip_output.size(-2):
            x = F.interpolate(x, size=skip_output.shape[-2:], mode='bilinear', align_corners=False)
        if self.film_on_skip:
            skip_output = self.film_skip(skip_output, z)
        x = torch.cat([x, skip_output], dim=1)
        x = self.conv(x)
        x = self.film(x, z)
        return x

# ======================
# 4. UNet + FiLM + VAE
# ======================
class UNetGeneratorFiLMVAE(nn.Module):
    """
    - Encoder: giống UNet thường
    - GlobalContextVAE: lấy feature x5 -> z, mu, logvar
    - DecoderFiLM: FiLM được điều kiện bởi z
    - Forward trả về: out, mu, logvar
    """
    def __init__(self, z_dim=256, use_ctx_attn=False, film_on_skip=False):
        super().__init__()
        # Encoder
        self.input_layer = ConvBlock(1, 64)
        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024)

        self.global_ctx = GlobalContextVAE(
            in_ch=1024,
            hidden=512,
            z_dim=z_dim,
            use_attn=use_ctx_attn
        )

        # Decoder + FiLM
        self.dec1 = DecoderFiLM(in_channels=1024, out_channels=512, z_dim=z_dim,
                                film_on_skip=film_on_skip, skip_channels=512)
        self.dec2 = DecoderFiLM(in_channels=512,  out_channels=256, z_dim=z_dim,
                                film_on_skip=film_on_skip, skip_channels=256)
        self.dec3 = DecoderFiLM(in_channels=256,  out_channels=128, z_dim=z_dim,
                                film_on_skip=film_on_skip, skip_channels=128)
        self.dec4 = DecoderFiLM(in_channels=128,  out_channels=64,  z_dim=z_dim,
                                film_on_skip=film_on_skip, skip_channels=64)

        # Output: 2 channel (ab)
        self.output_layer = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.input_layer(x)  # (B, 64, H, W)
        x2 = self.enc1(x1)        # (B,128,H/2,W/2)
        x3 = self.enc2(x2)        # (B,256,H/4,W/4)
        x4 = self.enc3(x3)        # (B,512,H/8,W/8)
        x5 = self.enc4(x4)        # (B,1024,H/16,W/16)

        # Global VAE latent
        z, mu, logvar = self.global_ctx(x5)  # (B,z_dim), (B,z_dim), (B,z_dim)

        # Decoder + FiLM
        d1 = self.dec1(x5, x4, z)
        d2 = self.dec2(d1, x3, z)
        d3 = self.dec3(d2, x2, z)
        d4 = self.dec4(d3, x1, z)

        out = self.output_layer(d4)
        out = torch.tanh(out)  # ab ∈ [-1,1]

        
        return out, mu, logvar

# ======================
# 5. Init & build model
# ======================
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

def build_model(z_dim=256, use_ctx_attn=False, film_on_skip=False, init=True):
    model = UNetGeneratorFiLMVAE(
        z_dim=z_dim,
        use_ctx_attn=use_ctx_attn,
        film_on_skip=film_on_skip
    )
    if init:
        model.apply(init_weights)
    return model

@torch.no_grad()
def load_trained_model_vae(model_path='model_vae.pth',
                           z_dim=256, use_ctx_attn=False, film_on_skip=False,
                           map_location='cpu'):
    model = build_model_vae(z_dim=z_dim, use_ctx_attn=use_ctx_attn,
                            film_on_skip=film_on_skip, init=True)
    ckpt = torch.load(model_path, map_location=torch.device(map_location))
    missing, unexpected = model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    print("[load] missing keys:", missing)
    print("[load] unexpected keys:", unexpected)
    model.eval()
    return model
