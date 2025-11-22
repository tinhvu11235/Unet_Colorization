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
        x = self.pool(x)
        x = self.conv(x)
        return x


class GlobalContextVAE(nn.Module):
    def __init__(self, in_ch, hidden=512, z_dim=256, use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        if use_attn:
            self.query = nn.Parameter(torch.randn(1, in_ch, 1, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * z_dim, 1)
        )

    def forward(self, f):
        B, C, H, W = f.shape
        if self.use_attn:
            attn = torch.sum(f * self.query, dim=1, keepdim=True)
            attn = torch.softmax(attn.flatten(1), dim=1).view(B, 1, H, W)
            f = f * attn
        h = self.pool(f)
        h = self.mlp(h).flatten(1)
        mu, logvar = torch.chunk(h, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class FiLM(nn.Module):
    def __init__(self, feat_ch, z_dim):
        super().__init__()
        self.gamma = nn.Linear(z_dim, feat_ch)
        self.beta = nn.Linear(z_dim, feat_ch)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, z):
        g = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(z).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + g) + b


class DecoderFiLM(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim, film_on_skip=False, skip_channels=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels)
        self.film = FiLM(out_channels, z_dim)
        self.film_on_skip = film_on_skip
        if self.film_on_skip:
            assert skip_channels is not None
            self.film_skip = FiLM(skip_channels, z_dim)

    def forward(self, prev_output, skip_output, z):
        x = self.up(prev_output)
        if x.size(-1) != skip_output.size(-1) or x.size(-2) != skip_output.size(-2):
            x = F.interpolate(x, size=skip_output.shape[-2:], mode='bilinear', align_corners=False)
        if self.film_on_skip:
            skip_output = self.film_skip(skip_output, z)
        x = torch.cat([x, skip_output], dim=1)
        x = self.conv(x)
        x = self.film(x, z)
        return x


class UNetGeneratorFiLMVAE(nn.Module):
    def __init__(self, z_dim=256, use_ctx_attn=False, film_on_skip=False):
        super().__init__()
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
        self.dec1 = DecoderFiLM(
            in_channels=1024 + 512,
            out_channels=512,
            z_dim=z_dim,
            film_on_skip=film_on_skip,
            skip_channels=512
        )
        self.dec2 = DecoderFiLM(
            in_channels=512 + 256,
            out_channels=256,
            z_dim=z_dim,
            film_on_skip=film_on_skip,
            skip_channels=256
        )
        self.dec3 = DecoderFiLM(
            in_channels=256 + 128,
            out_channels=128,
            z_dim=z_dim,
            film_on_skip=film_on_skip,
            skip_channels=128
        )
        self.dec4 = DecoderFiLM(
            in_channels=128 + 64,
            out_channels=64,
            z_dim=z_dim,
            film_on_skip=film_on_skip,
            skip_channels=64
        )
        self.output_layer = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        z, mu, logvar = self.global_ctx(x5)
        d1 = self.dec1(x5, x4, z)
        d2 = self.dec2(d1, x3, z)
        d3 = self.dec3(d2, x2, z)
        d4 = self.dec4(d3, x1, z)
        out = self.output_layer(d4)
        out = torch.tanh(out)
        return out, mu, logvar


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


@torch.no_grad()
def load_trained_model(model_path='model_vae.pth',
                       z_dim=256, use_ctx_attn=False, film_on_skip=False,
                       map_location='cpu'):
    model = UNetGeneratorFiLMVAE(
        z_dim=z_dim,
        use_ctx_attn=use_ctx_attn,
        film_on_skip=film_on_skip
    )
    model.apply(init_weights)
    ckpt = torch.load(model_path, map_location=torch.device(map_location))
    missing, unexpected = model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    print("[load] missing keys:", missing)
    print("[load] unexpected keys:", unexpected)
    model.eval()
    return model


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

    def forward(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class GAN(nn.Module):
    def __init__(self,
                 lr_G=2e-4, lr_D=1e-4,
                 beta1=0.5, beta2=0.999,
                 lambda_L1=100.,
                 lambda_KL=0.0001,
                 z_dim=256,
                 use_ctx_attn=False,
                 film_on_skip=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.lambda_KL = lambda_KL
        self.net_G = UNetGeneratorFiLMVAE(
            z_dim=z_dim,
            use_ctx_attn=use_ctx_attn,
            film_on_skip=film_on_skip
        ).to(self.device)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            net_G = nn.DataParallel(net_G)
            net_D = nn.DataParallel(net_D)
        self.net_G.apply(init_weights)
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
        self.fake_color, self.mu, self.logvar = self.net_G(self.L)

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

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G_KL = kl_loss(self.mu, self.logvar) * self.lambda_KL
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_KL
        self.loss_G.backward()

    def warmup_optimize(self):
        self.forward()
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G_KL = kl_loss(self.mu, self.logvar) * self.lambda_KL
        loss = self.loss_G_L1 + self.loss_G_KL
        loss.backward()
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
            gan_model.forward()
            gan_model.opt_D.zero_grad()
            gan_model.backward_D(noGAN=False)
            gan_model.opt_D.step()
            running_loss += gan_model.loss_D.item()
            real_loss += gan_model.loss_D_real.item()
            fake_loss += gan_model.loss_D_fake.item()
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Running Loss: {running_loss / len(shuffled_train_dl):.4f}, "
            f"Real Loss: {real_loss / len(shuffled_train_dl):.4f}, "
            f"Fake Loss: {fake_loss / len(shuffled_train_dl):.4f}"
        )
    print("Pretraining for Discriminator is complete.")
