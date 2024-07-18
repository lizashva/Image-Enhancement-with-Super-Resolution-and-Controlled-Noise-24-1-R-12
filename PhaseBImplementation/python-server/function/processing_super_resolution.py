from flask import Flask, request, jsonify, send_from_directory
import tempfile
import os
import cv2
from flask_cors import CORS
from PIL import Image
import numpy as np              # pkg to read filepaths from the dataset folder       
import tensorflow.keras as K 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer,Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Add, Concatenate
from tensorflow.keras.applications.vgg19 import VGG19
class pixel_shuffle(Layer):
    def __init__(self, scale, **kwargs):
        super(pixel_shuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        if self.scale > 1:
            return tf.nn.depth_to_space(inputs, self.scale)
        return inputs
# Define the custom loss and metrics
# load pre-trained (imagenet) vgg network, excluding fully-connected layer on the top
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None,None,3))
vgg_layer = K.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)


def perceptual_loss(y_true,y_pred):
    '''This function computes the perceptual loss using an already trained VGG layer'''
    y_t=vgg_layer(y_true)
    y_p=vgg_layer(y_pred)
    loss=K.losses.mean_squared_error(y_t,y_p)
    return loss

def psnr(y_true,y_pred):
    return tf.image.psnr(y_true,y_pred,1.0)
def ssim(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,1.0)

#blocks definition for upscaling/downscaling

def add_down_block(x_inp, filters, kernel_size=(3, 3), padding="same", strides=1,r=False):
    x = K.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x_inp)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = K.layers.BatchNormalization()(x)
    if r:
        # if r=True then we import an (1X1) Conv2D after input layer
        # in order the dimensions of 2 tensors coincide.
        x_inp = K.layers.Conv2D(filters,(1,1), padding=padding, strides=strides)(x_inp)
    x = K.layers.Add()([x,x_inp])
    return x

def add_up_block(x_inp, skip, filters, kernel_size=(3, 3), padding="same", strides=1, upscale_factor=2):
    # Calculate the upsampling factor based on the difference in input sizes
    upsampling_factor = skip.shape[1] // x_inp.shape[1]

    x = K.layers.UpSampling2D(size=(upsampling_factor, upsampling_factor))(x_inp)

    # Adjust the shape of the skip connection tensor
    skip = K.layers.Conv2D(filters, kernel_size=(1, 1), padding="same")(skip)

    x = K.layers.Concatenate()([x, skip])
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = K.layers.Activation('relu')(x)
    return x

def add_bottleneck(x_inp,filters, kernel_size=(3, 3), padding="same", strides=1):
    x = K.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x_inp)
    x = K.layers.Activation('relu')(x)
    return x

def RUNet(input_size_train=256, input_size_test=128):
    """
    Implementing with Keras the Robust UNet Architecture as proposed by
    Xiaodan Hu, Mohamed A. Naiel, Alexander Wong, Mark Lamm, Paul Fieguth
    in "RUNet: A Robust UNet Architecture for Image Super-Resolution"
    """
    inputs = K.layers.Input((input_size_train, input_size_train, 3))

    down_1 = K.layers.Conv2D(64, (7, 7), padding="same", strides=1)(inputs)
    down_1 = K.layers.BatchNormalization()(down_1)
    down_1 = K.layers.Activation('relu')(down_1)

    down_2 = K.layers.MaxPool2D(pool_size=(2, 2))(down_1)
    down_2 = add_down_block(down_2, 64)
    down_2 = add_down_block(down_2, 64)
    down_2 = add_down_block(down_2, 64)
    down_2 = add_down_block(down_2, 128, r=True)

    down_3 = K.layers.MaxPool2D(pool_size=(2, 2), strides=2)(down_2)
    down_3 = add_down_block(down_3, 128)
    down_3 = add_down_block(down_3, 128)
    down_3 = add_down_block(down_3, 128)
    down_3 = add_down_block(down_3, 256, r=True)

    down_4 = K.layers.MaxPool2D(pool_size=(2, 2))(down_3)
    down_4 = add_down_block(down_4, 256)
    down_4 = add_down_block(down_4, 256)
    down_4 = add_down_block(down_4, 256)
    down_4 = add_down_block(down_4, 256)
    down_4 = add_down_block(down_4, 256)
    down_4 = add_down_block(down_4, 512, r=True)

    down_5 = K.layers.MaxPool2D(pool_size=(2, 2))(down_4)
    down_5 = add_down_block(down_5, 512)
    down_5 = add_down_block(down_5, 512)
    down_5 = K.layers.BatchNormalization()(down_5)
    down_5 = K.layers.Activation('relu')(down_5)

    bn_1 = add_bottleneck(down_5, 1024)
    bn_2 = add_bottleneck(bn_1, 512)

    up_1 = add_up_block(bn_2, down_5, 512, upscale_factor=input_size_train // input_size_test)
    up_2 = add_up_block(up_1, down_4, 384, upscale_factor=2)
    up_3 = add_up_block(up_2, down_3, 256, upscale_factor=2)
    up_4 = add_up_block(up_3, down_2, 96, upscale_factor=2)

    up_5 = pixel_shuffle(scale=2)(up_4)
    up_5 = K.layers.Concatenate()([up_5, down_1])
    up_5 = K.layers.Conv2D(99, (3, 3), padding="same", strides=1)(up_5)
    up_5 = K.layers.Activation('relu')(up_5)
    up_5 = K.layers.Conv2D(99, (3, 3), padding="same", strides=1)(up_5)
    up_5 = K.layers.Activation('relu')(up_5)

    outputs = K.layers.Conv2D(3, (1, 1), padding="same")(up_5)
    model = K.models.Model(inputs, outputs)
    return model

def load_model_super_resolution():
    model_path = 'models/super_resolution_weights.h5'
    print(f"Loading model from: {model_path}")
    model = RUNet(input_size_train=256)  # Correctly setting the expected input size
    model.load_weights(model_path)
    model.compile(optimizer='adam', loss=perceptual_loss)  # Ensuring loss function is correctly set
    print("Model loaded successfully.")
    return model


def preprocess_image(img_path, input_size=256):  # Defaulting to 256 as it's required by the model
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("The provided image path does not exist or the file is not an image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))  # Ensuring image is resized to 256x256
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def super_resolve_image(image_path, save_path):
    print(f"Performing super-resolution on image: {image_path}")
    model = load_model_super_resolution()

    # Preprocess the image, ensure input size is set to 256 which is expected by your model
    low_res_img = preprocess_image(image_path, 256)
    
    # Use the model to predict the high-resolution version of the image
    high_res_img = model.predict(low_res_img)
    high_res_img = np.clip(high_res_img.squeeze(), 0, 1)  # Ensure values are between 0 and 1

    # Convert the high-resolution image from numpy array to PIL Image
    high_res_img = (high_res_img * 255).astype(np.uint8)
    high_res_img_pil = Image.fromarray(cv2.cvtColor(high_res_img, cv2.COLOR_RGB2BGR))

    # Save the high-resolution image using PIL, which supports .save()
    high_res_img_pil.save(save_path)
    print(f"High-resolution image saved at: {save_path}")

    return save_path