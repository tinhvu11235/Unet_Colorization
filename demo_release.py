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
import model

def download_model(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists, skipping download.")

def preprocess_image(img_path, target_size=(256, 256)):
    img_rgb = Image.open(img_path).convert('RGB')
    img_lab = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2LAB)
    L_channel = img_lab[:, :, 0]
    L_channel_original = L_channel.copy()
    L_channel_normalized = L_channel / 255.0
    L_channel_pil = Image.fromarray((L_channel_normalized * 255).astype(np.uint8)).convert('L')
    preprocess_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    L_tensor = preprocess_transform(L_channel_pil).unsqueeze(0) 
    return img_rgb, L_channel_original, L_tensor


def predict_ab(model, L_tensor, original_shape):
    with torch.no_grad():
        output = model(L_tensor)

    predicted_AB = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    predicted_AB_rescaled = (predicted_AB + 1) * 127.5
    predicted_AB_resized = cv2.resize(predicted_AB_rescaled, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    return predicted_AB_resized

def reconstruct_rgb(L_channel_original, predicted_AB):
    lab_image = np.zeros((L_channel_original.shape[0], L_channel_original.shape[1], 3), dtype=np.uint8)
    lab_image[:, :, 0] = L_channel_original
    lab_image[:, :, 1:] = predicted_AB.astype(np.uint8)
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return rgb_image


def main(input_path, output_dir):
    model_url = 'https://drive.google.com/uc?id=1dD7PQt1RB-IqNVJFHlnsG9MdkmdDuRxH'
    model_path = 'model.pth'
    download_model(model_url, model_path)
    
    colorizor = model.load_trained_model(model_path)
    img_rgb, L_channel_original, L_tensor = preprocess_image(input_path)
    predicted_AB = predict_ab(colorizor, L_tensor, L_channel_original.shape)
    predicted_rgb = reconstruct_rgb(L_channel_original, predicted_AB)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "unet-predict.jpg")
    predicted_image = Image.fromarray(predicted_rgb)
    predicted_image.save(output_path)

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
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input image")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output directory to save the predicted image")
    args = parser.parse_args()
    main(args.input, args.output)
