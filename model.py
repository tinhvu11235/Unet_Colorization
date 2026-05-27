import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


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


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, prev_output, skip_output):
        x = self.upconv(prev_output)
        x = torch.cat([x, skip_output], dim=1)
        return self.conv(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = attn_dropout
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn_mask = relative_position_bias.unsqueeze(0)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)
    return windows


def window_reverse(windows, window_size, h, w, c):
    b = int(windows.shape[0] / ((h // window_size) * (w // window_size)))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, c)
    return x


class SwinBlock2D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim, 
            window_size=window_size, 
            num_heads=num_heads, 
            attn_dropout=attn_dropout, 
            proj_dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = x.shape[1], x.shape[2]

        shift_size = 0
        if min(hp, wp) > self.window_size:
            shift_size = min(self.shift_size, self.window_size // 2)

        x = self.norm1(x)
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, hp, wp, c)

        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        if pad_h > 0 or pad_w > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + x

        mlp_input = x.permute(0, 2, 3, 1).contiguous()
        mlp_out = self.mlp(self.norm2(mlp_input))
        mlp_out = mlp_out.permute(0, 3, 1, 2).contiguous()
        x = x + mlp_out
        return x


class SwinStage2D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        depth=2,
        window_size=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            shift = 0 if i % 2 == 0 else window_size // 2
            blocks.append(
                SwinBlock2D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class SwinAdapter2D(nn.Module):
    def __init__(
        self,
        in_channels,
        attn_dim,
        num_heads,
        depth=1,
        window_size=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=False)
        self.reduce_norm = nn.BatchNorm2d(attn_dim)
        self.reduce_act = nn.ReLU(inplace=True)
        self.swin = SwinStage2D(
            dim=attn_dim,
            num_heads=num_heads,
            depth=depth,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.expand = nn.Conv2d(attn_dim, in_channels, kernel_size=1, bias=False)
        self.expand_norm = nn.BatchNorm2d(in_channels)

        nn.init.constant_(self.expand_norm.weight, 0)
        nn.init.constant_(self.expand_norm.bias, 0)

    def forward(self, x):
        shortcut = x
        x = self.reduce(x)
        x = self.reduce_norm(x)
        x = self.reduce_act(x)
        x = self.swin(x)
        x = self.expand(x)
        x = self.expand_norm(x)
        return shortcut + x


class UNetGenerator(nn.Module):
    def __init__(
        self,
        use_segmentation=False,
        num_seg_classes=182,
        use_swin_deep=False,
        swin_window_size=8,
        swin_enc3_depth=2,
        swin_enc4_depth=2,
        swin_num_heads_enc3=4,
        swin_num_heads_enc4=8,
        swin_attn_dim_enc3=256,
        swin_attn_dim_enc4=256,
        swin_mlp_ratio=4.0,
        swin_dropout=0.0,
        swin_attn_dropout=0.0,
    ):
        super().__init__()
        self.use_segmentation = use_segmentation
        self.num_seg_classes = num_seg_classes
        self.use_swin_deep = use_swin_deep

        self.arch_kwargs = {
            "use_segmentation": use_segmentation,
            "num_seg_classes": num_seg_classes,
            "use_swin_deep": use_swin_deep,
            "swin_window_size": swin_window_size,
            "swin_enc3_depth": swin_enc3_depth,
            "swin_enc4_depth": swin_enc4_depth,
            "swin_num_heads_enc3": swin_num_heads_enc3,
            "swin_num_heads_enc4": swin_num_heads_enc4,
            "swin_attn_dim_enc3": swin_attn_dim_enc3,
            "swin_attn_dim_enc4": swin_attn_dim_enc4,
            "swin_mlp_ratio": swin_mlp_ratio,
            "swin_dropout": swin_dropout,
            "swin_attn_dropout": swin_attn_dropout,
        }

        self.input_layer = ConvBlock(1, 64)
        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024)

        if use_swin_deep:
            self.swin_enc3 = SwinAdapter2D(
                in_channels=512,
                attn_dim=swin_attn_dim_enc3,
                num_heads=swin_num_heads_enc3,
                depth=swin_enc3_depth,
                window_size=swin_window_size,
                mlp_ratio=swin_mlp_ratio,
                dropout=swin_dropout,
                attn_dropout=swin_attn_dropout,
            )
            self.swin_enc4 = SwinAdapter2D(
                in_channels=1024,
                attn_dim=swin_attn_dim_enc4,
                num_heads=swin_num_heads_enc4,
                depth=swin_enc4_depth,
                window_size=swin_window_size,
                mlp_ratio=swin_mlp_ratio,
                dropout=swin_dropout,
                attn_dropout=swin_attn_dropout,
            )
        else:
            self.swin_enc3 = nn.Identity()
            self.swin_enc4 = nn.Identity()

        self.dec1 = Decoder(1024, 512)
        self.dec2 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec4 = Decoder(128, 64)
        self.output_layer = nn.Conv2d(64, 2, kernel_size=1)
        self.seg_output_layer = nn.Conv2d(64, num_seg_classes, kernel_size=1) if use_segmentation else None

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x4 = self.swin_enc3(x4)
        x5 = self.enc4(x4)
        x5 = self.swin_enc4(x5)

        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)

        color = self.output_layer(x)
        color = torch.tanh(color)

        if self.use_segmentation:
            seg_logits = self.seg_output_layer(x)
            return {"ab": color, "seg": seg_logits}

        return color

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return self


def get_encoder_weights(model_path='model.pth'):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    encoder_prefixes = ('input_layer', 'enc', 'swin_enc3', 'swin_enc4')
    encoder_state_dict = {k: v for k, v in state_dict.items() if k.startswith(encoder_prefixes)}
    return encoder_state_dict


def load_trained_model(model_path='model.pth'):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    generator_kwargs = checkpoint.get('generator_kwargs', {})
    model = UNetGenerator(**generator_kwargs)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception:
        if 'Unet_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['Unet_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model.eval()
    return model


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [
            self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2)
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


class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported gan_mode: {gan_mode}")

    def get_labels(self, preds, target_is_real):
        labels = self.real_label if target_is_real else self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


class GAN(nn.Module):
    def __init__(
        self,
        lr_G=2e-4,
        lr_D=1e-4,
        beta1=0.5,
        beta2=0.999,
        lambda_L1=100.,
        use_segmentation=False,
        num_seg_classes=182,
        seg_ignore_index=255,
        lambda_seg=0.01,
        lambda_obj=0.01,
        object_min_pixels=16,
        object_use_connected_components=True,
        generator_kwargs=None,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.use_segmentation = use_segmentation
        self.seg_ignore_index = seg_ignore_index
        self.lambda_seg = lambda_seg
        self.lambda_obj = lambda_obj
        self.object_min_pixels = object_min_pixels
        self.object_use_connected_components = object_use_connected_components
        self.generator_kwargs = generator_kwargs or {}

        generator_init_kwargs = {
            "use_segmentation": use_segmentation,
            "num_seg_classes": num_seg_classes,
            **self.generator_kwargs,
        }

        self.net_G = UNetGenerator(**generator_init_kwargs).init_weights().to(self.device)
        self.net_D = PatchDiscriminator(input_c=3).init_weights().to(self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=seg_ignore_index) if use_segmentation else None
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        self.scheduler_G = ReduceLROnPlateau(self.opt_G, mode='min', factor=0.95, patience=5)

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        if self.use_segmentation and 'seg' in data:
            self.seg = data['seg'].long().to(self.device)

    def forward(self):
        output = self.net_G(self.L)
        if isinstance(output, dict):
            self.fake_color = output['ab']
            self.pred_seg = output['seg']
        else:
            self.fake_color = output
            self.pred_seg = None

    def backward_D(self, noGAN=False):
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

    def _compute_seg_loss(self):
        if not self.use_segmentation or self.pred_seg is None or not hasattr(self, 'seg'):
            return torch.tensor(0.0, device=self.L.device)
        return self.seg_criterion(self.pred_seg, self.seg) * self.lambda_seg

    def _compute_object_recon_loss(self):
        if (
            self.lambda_obj <= 0
            or not self.use_segmentation
            or not hasattr(self, 'seg')
            or self.seg is None
        ):
            return torch.tensor(0.0, device=self.L.device)

        error_map = torch.mean(torch.abs(self.fake_color - self.ab), dim=1)
        object_losses = []

        seg_cpu = self.seg.detach().cpu().numpy().astype(np.int32)

        for b in range(seg_cpu.shape[0]):
            seg_np = seg_cpu[b]
            err_b = error_map[b]
            valid_classes = np.unique(seg_np)
            valid_classes = valid_classes[valid_classes != self.seg_ignore_index]

            for cls_id in valid_classes.tolist():
                class_mask = (seg_np == cls_id).astype(np.uint8)
                if class_mask.sum() < self.object_min_pixels:
                    continue

                if self.object_use_connected_components:
                    num_labels, labels = cv2.connectedComponents(class_mask, connectivity=8)
                    for comp_id in range(1, num_labels):
                        comp_mask_np = (labels == comp_id)
                        if int(comp_mask_np.sum()) < self.object_min_pixels:
                            continue
                        comp_mask = torch.from_numpy(comp_mask_np).to(err_b.device, dtype=torch.bool)
                        object_losses.append(err_b[comp_mask].mean())
                else:
                    cls_mask = torch.from_numpy(class_mask.astype(bool)).to(err_b.device)
                    object_losses.append(err_b[cls_mask].mean())

        if not object_losses:
            return torch.tensor(0.0, device=self.L.device)

        return torch.stack(object_losses).mean() * self.lambda_obj

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G_SEG = self._compute_seg_loss()
        self.loss_G_OBJ = self._compute_object_recon_loss()
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_SEG + self.loss_G_OBJ
        self.loss_G.backward()

    def warmup_optimize(self):
        self.forward()
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G_SEG = self._compute_seg_loss()
        self.loss_G_OBJ = self._compute_object_recon_loss()
        self.loss_G_warmup = self.loss_G_L1 + self.loss_G_SEG + self.loss_G_OBJ
        self.loss_G_warmup.backward()
        self.opt_G.step()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D(noGAN=False)
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()


def pretrain_discriminator(train_dl, gan_model, lr=2e-5, epochs=3):
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