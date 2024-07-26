import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from super_resolution_train256_test128 import *


def load_model():
    model = RUNet()  # Make sure RUNet is correctly defined
    model.compile(optimizer=Adam(learning_rate=0.001), loss='perceptual_loss', metrics=['psnr', 'ssim', 'mse'])
    model.load_weights('/content/drive/MyDrive/ProjectPhaseB/weight/super_resolution_weights.h5')
    return model


def super_resolve_and_display(image_path, save_path):
    model = load_model()
    low_res_img = image_preprocess(image_path)
    high_res_img = model.predict(low_res_img)

    # Post-process to ensure it's displayable: clip to [0, 1], remove batch dimension
    high_res_img = np.clip(high_res_img.squeeze(), 0, 1)

    # Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(low_res_img.squeeze())
    plt.title('Low Resolution')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(high_res_img)
    plt.title('High Resolution')
    plt.axis('off')
    plt.show()

    # Save the high-resolution image
    high_res_img = (high_res_img * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(high_res_img, cv2.COLOR_RGB2BGR))
    print(f"High-resolution image saved to {save_path}")


# Example usage
image_path = '/content/drive/MyDrive/ProjectPhaseB/12_128.jpg'
save_path = '/content/drive/MyDrive/ProjectPhaseB/12_high_res.jpg'
super_resolve_and_display(image_path, save_path)
