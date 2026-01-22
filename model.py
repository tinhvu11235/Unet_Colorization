# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from diffusers.models.embeddings import get_timestep_embedding


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, groups=32):
#         super().__init__()
#         g = min(groups, out_channels)
#         while out_channels % g != 0 and g > 1:
#             g -= 1
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
#             nn.GroupNorm(g, out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
#             nn.GroupNorm(g, out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.block(x)


# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels, groups=32):
#         super().__init__()
#         self.pool = nn.MaxPool2d(2)
#         self.conv = ConvBlock(in_channels, out_channels, groups=groups)

#     def forward(self, x):
#         return self.conv(self.pool(x))


# class GlobalContext(nn.Module):
#     def __init__(self, in_ch, hidden=256, z_dim=256):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_ch, hidden, 1),
#             nn.SiLU(),
#             nn.Conv2d(hidden, z_dim, 1),
#         )

#     def forward(self, f):
#         return self.mlp(self.pool(f)).flatten(1)


# class FiLM(nn.Module):
#     def __init__(self, feat_ch, z_dim):
#         super().__init__()
#         self.gamma = nn.Linear(z_dim, feat_ch)
#         self.beta = nn.Linear(z_dim, feat_ch)
#         nn.init.zeros_(self.gamma.weight)
#         nn.init.zeros_(self.gamma.bias)
#         nn.init.zeros_(self.beta.weight)
#         nn.init.zeros_(self.beta.bias)

#     def forward(self, x, z):
#         g = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
#         b = self.beta(z).unsqueeze(-1).unsqueeze(-1)
#         return x * (1.0 + g) + b


# class DecoderFiLM(nn.Module):
#     def __init__(self, in_channels, out_channels, z_dim, film_on_skip=False, skip_channels=None, groups=32):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
#         self.conv = ConvBlock(in_channels, out_channels, groups=groups)
#         self.film = FiLM(out_channels, z_dim)
#         self.film_on_skip = film_on_skip
#         if film_on_skip:
#             self.film_skip = FiLM(skip_channels, z_dim)

#     def forward(self, prev_output, skip_output, z):
#         x = self.up(prev_output)
#         if x.shape[-2:] != skip_output.shape[-2:]:
#             x = F.interpolate(x, size=skip_output.shape[-2:], mode="bilinear", align_corners=False)
#         if self.film_on_skip:
#             skip_output = self.film_skip(skip_output, z)
#         x = torch.cat([x, skip_output], dim=1)
#         x = self.conv(x)
#         x = self.film(x, z)
#         return x


# class UNetDenoiserFiLM(nn.Module):
#     def __init__(self, z_dim=256, time_emb_dim=256, film_on_skip=False, groups=32):
#         super().__init__()

#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_emb_dim, z_dim),
#             nn.SiLU(),
#             nn.Linear(z_dim, z_dim),
#         )

#         self.L_input = ConvBlock(1, 64, groups=groups)
#         self.L_enc1 = Encoder(64, 128, groups=groups)
#         self.L_enc2 = Encoder(128, 256, groups=groups)
#         self.L_enc3 = Encoder(256, 512, groups=groups)
#         self.L_enc4 = Encoder(512, 1024, groups=groups)
#         self.global_ctx = GlobalContext(1024, hidden=256, z_dim=z_dim)

#         self.fuse = nn.Sequential(
#             nn.Linear(z_dim * 2, z_dim),
#             nn.SiLU(),
#             nn.Linear(z_dim, z_dim),
#         )

#         self.input_layer = ConvBlock(3, 64, groups=groups)
#         self.enc1 = Encoder(64, 128, groups=groups)
#         self.enc2 = Encoder(128, 256, groups=groups)
#         self.enc3 = Encoder(256, 512, groups=groups)
#         self.enc4 = Encoder(512, 1024, groups=groups)

#         self.dec1 = DecoderFiLM(1024 + 1024, 512, z_dim, film_on_skip, 1024, groups=groups)
#         self.dec2 = DecoderFiLM(512 + 512, 256, z_dim, film_on_skip, 512, groups=groups)
#         self.dec3 = DecoderFiLM(256 + 256, 128, z_dim, film_on_skip, 256, groups=groups)
#         self.dec4 = DecoderFiLM(128 + 128, 64, z_dim, film_on_skip, 128, groups=groups)

#         self.output_layer = nn.Conv2d(64, 2, kernel_size=1)

#     def compute_L_feats(self, L):
#         l1 = self.L_input(L)
#         l2 = self.L_enc1(l1)
#         l3 = self.L_enc2(l2)
#         l4 = self.L_enc3(l3)
#         l5 = self.L_enc4(l4)
#         z_ctx = self.global_ctx(l5)
#         return z_ctx, (l1, l2, l3, l4)

#     def forward(self, ab_t, L, t, z_ctx=None):
#         t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
#         z_t = self.time_mlp(t_emb)

#         z_ctx, (l1, l2, l3, l4) = self.compute_L_feats(L)
#         z = self.fuse(torch.cat([z_t, z_ctx], dim=1))

#         x = torch.cat([ab_t, L], dim=1)
#         x1 = self.input_layer(x)
#         x2 = self.enc1(x1)
#         x3 = self.enc2(x2)
#         x4 = self.enc3(x3)
#         x5 = self.enc4(x4)

#         s4 = torch.cat([x4, l4], dim=1)
#         s3 = torch.cat([x3, l3], dim=1)
#         s2 = torch.cat([x2, l2], dim=1)
#         s1 = torch.cat([x1, l1], dim=1)

#         d1 = self.dec1(x5, s4, z)
#         d2 = self.dec2(d1, s3, z)
#         d3 = self.dec3(d2, s2, z)
#         d4 = self.dec4(d3, s1, z)

#         return self.output_layer(d4)


# def _init_weights(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.GroupNorm):
#         nn.init.ones_(m.weight)
#         nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)


# def build_model(
#     z_dim=256,
#     time_emb_dim=256,
#     film_on_skip=False,
#     init_weights=True,
#     groups=32,
# ):
#     model = UNetDenoiserFiLM(
#         z_dim=z_dim,
#         time_emb_dim=time_emb_dim,
#         film_on_skip=film_on_skip,
#         groups=groups,
#     )

#     if init_weights:
#         model.apply(_init_weights)
#         nn.init.zeros_(model.output_layer.weight)
#         if model.output_layer.bias is not None:
#             nn.init.zeros_(model.output_layer.bias)

#     return model
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import get_timestep_embedding


def _make_gn(groups: int, channels: int):
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 32):
        super().__init__()
        self.norm1 = _make_gn(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = _make_gn(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 32):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim, groups)
        self.down = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        x = self.res(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, groups: int = 32):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.res = ResBlock(in_ch + skip_ch, out_ch, time_dim, groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return x


class UNetBaseline(nn.Module):
    def __init__(self, base_ch: int = 64, time_emb_dim: int = 256, groups: int = 32):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.in_conv = nn.Conv2d(3, base_ch, 3, padding=1, bias=False)

        self.d1 = DownBlock(base_ch, base_ch * 2, time_emb_dim, groups)
        self.d2 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, groups)
        self.d3 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, groups)
        self.d4 = DownBlock(base_ch * 8, base_ch * 16, time_emb_dim, groups)

        self.mid1 = ResBlock(base_ch * 16, base_ch * 16, time_emb_dim, groups)
        self.mid2 = ResBlock(base_ch * 16, base_ch * 16, time_emb_dim, groups)

        self.u1 = UpBlock(base_ch * 16, base_ch * 16, base_ch * 8, time_emb_dim, groups)
        self.u2 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4, time_emb_dim, groups)
        self.u3 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, time_emb_dim, groups)
        self.u4 = UpBlock(base_ch * 2, base_ch * 2, base_ch, time_emb_dim, groups)

        self.out_norm = _make_gn(groups, base_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_ch, 2, 1)

    def forward(self, ab_t: torch.Tensor, L: torch.Tensor, t: torch.Tensor):
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x = torch.cat([ab_t, L], dim=1)
        x = self.in_conv(x)

        x, s1 = self.d1(x, t_emb)
        x, s2 = self.d2(x, t_emb)
        x, s3 = self.d3(x, t_emb)
        x, s4 = self.d4(x, t_emb)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.u1(x, s4, t_emb)
        x = self.u2(x, s3, t_emb)
        x = self.u3(x, s2, t_emb)
        x = self.u4(x, s1, t_emb)

        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GroupNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_model(
    time_emb_dim=256,
    init_weights=True,
    groups=32,
):
    model = UNetBaseline(base_ch=64, time_emb_dim=time_emb_dim, groups=groups)
    if init_weights:
        model.apply(_init_weights)
        nn.init.zeros_(model.out_conv.weight)
        if model.out_conv.bias is not None:
            nn.init.zeros_(model.out_conv.bias)
    return model
