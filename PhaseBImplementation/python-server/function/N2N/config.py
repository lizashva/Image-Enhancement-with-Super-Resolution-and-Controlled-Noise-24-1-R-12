from function.N2N.test import test_N2N
import os
from PIL import Image
import json
import glob

class TestConfig:
    def __init__(self, noise_type, path, noisy_source=True):
        self.data = path
        self.noise_type = noise_type
        self.load_ckpt = f'./models/{noise_type}/n2n-{noise_type}.pt'
        self.noise_param = self.get_noise_param(noise_type)
        self.seed = 1
        self.crop_size = self.get_image_size()
        self.show_output = 1
        self.noisy_source = noisy_source

    def get_image_size(self):
        img = os.listdir(self.data)[0]
        path_img = os.path.join(self.data, img)
        with Image.open(path_img) as img:
            width, height = img.size
        img_size = width if (width < 256) else 256
        return img_size
    
    def get_noise_param(self, noise):
        if noise == 'gaussian':
            return 50
        elif noise == "poisson":
            return 25
        elif noise == "bernouli":
            return 25
        else:
            return 25
        
def noise2noise(noise_type, path):
    params = TestConfig(noise_type, path)
    denoised_path = test_N2N(params)
    return denoised_path