import os
import torch
from torchvision import transforms
from PIL import Image
import gdown
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

def download_model(url, output_path):
    """Download model checkpoint từ Google Drive nếu file chưa tồn tại."""
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} đã tồn tại, bỏ qua download.")

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
        return self.output_layer(x)

def load_trained_model(model_path='model.pth'):
    """Load model đã được huấn luyện từ checkpoint."""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = UNetGenerator()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(img_path, target_size=(256, 256)):
    """
    Tiền xử lý ảnh:
      - Load ảnh gốc (RGB)
      - Chuyển sang không gian màu LAB để tách kênh L
      - Chuẩn hóa kênh L và resize theo kích thước đầu vào của model
    """
    img_rgb = Image.open(img_path).convert('RGB')
    img_lab = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2LAB)
    L_channel = img_lab[:, :, 0]           # Kênh L (độ sáng)
    L_channel_original = L_channel.copy()  # Lưu lại kích thước gốc của kênh L

    # Chuẩn hóa kênh L về [0, 1]
    L_channel_normalized = L_channel / 255.0
    # Chuyển về PIL Image ở dạng grayscale (range [0, 255])
    L_channel_pil = Image.fromarray((L_channel_normalized * 255).astype(np.uint8)).convert('L')
    
    preprocess_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    L_tensor = preprocess_transform(L_channel_pil).unsqueeze(0)  # Thêm batch dimension
    return img_rgb, L_channel_original, L_tensor

def predict_ab(model, L_tensor, original_shape):
    """
    Dự đoán kênh AB từ model và resize về kích thước ảnh gốc.
    - Đầu ra model có dạng (1, 2, 256, 256)
    - Chuyển đổi về dạng (H, W, 2) và rescale từ [-1, 1] sang [0, 255]
    """
    with torch.no_grad():
        output = model(L_tensor)
    predicted_AB = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    predicted_AB_rescaled = (predicted_AB + 1) * 127.5
    predicted_AB_resized = cv2.resize(predicted_AB_rescaled, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    return predicted_AB_resized

def reconstruct_rgb(L_channel_original, predicted_AB):
    """
    Hồi tạo ảnh LAB bằng cách kết hợp kênh L gốc và kênh AB dự đoán, sau đó chuyển đổi sang RGB.
    """
    lab_image = np.zeros((L_channel_original.shape[0], L_channel_original.shape[1], 3), dtype=np.uint8)
    lab_image[:, :, 0] = L_channel_original
    lab_image[:, :, 1:] = predicted_AB.astype(np.uint8)
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return rgb_image

def main(input_path, output_path):
    # Download checkpoint model từ Google Drive nếu chưa có
    model_url = 'https://drive.google.com/uc?id=1dD7PQt1RB-IqNVJFHlnsG9MdkmdDuRxH'
    model_path = 'model.pth'
    download_model(model_url, model_path)
    
    # Load model đã được huấn luyện
    model = load_trained_model(model_path)
    
    # Tiền xử lý ảnh đầu vào
    img_rgb, L_channel_original, L_tensor = preprocess_image(input_path)
    
    # Dự đoán kênh AB và resize về kích thước ảnh gốc
    predicted_AB = predict_ab(model, L_tensor, L_channel_original.shape)
    
    # Hồi tạo ảnh RGB từ ảnh LAB (với L gốc và AB dự đoán)
    predicted_rgb = reconstruct_rgb(L_channel_original, predicted_AB)
    
    # Lưu ảnh kết quả
    predicted_image = Image.fromarray(predicted_rgb)
    predicted_image.save(output_path)
    
    # Hiển thị ảnh gốc và ảnh dự đoán
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Ground Truth Image (RGB)')
    axes[0].axis('off')
    
    axes[1].imshow(predicted_rgb)
    axes[1].set_title('Predicted Image (Reconstructed RGB)')
    axes[1].axis('off')
    
    plt.show()
    print(f"Predicted image saved at: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Colorize images using a pretrained model")
    parser.add_argument('input_path', type=str, help="Path to the input image")
    parser.add_argument('output_path', type=str, help="Path to save the output image")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
