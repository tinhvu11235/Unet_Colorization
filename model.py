import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = ConvBlock(1, 64)
        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024)
        self.dec1 = Decoder(1024, 512)
        self.dec2 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec4 = Decoder(128, 64)
        self.output_layer = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        x = self.output_layer(x)
        x = torch.tanh(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return self
    
def get_encoder_weights(model_path='model.pth'):
 
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    encoder_state_dict = {k: v for k, v in state_dict.items() 
                          if k.startswith('input_layer') or k.startswith('enc')}
    return encoder_state_dict 

def load_trained_model(model_path='model.pth'):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = UNetGenerator()
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

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
    def __init__(self, lr_G=2e-4, lr_D=1e-4, beta1=0.5, beta2=0.999, lambda_L1=30.):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.net_G = UNetGenerator().init_weights().to(self.device)
        self.net_D = PatchDiscriminator(input_c=3).init_weights().to(self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

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

