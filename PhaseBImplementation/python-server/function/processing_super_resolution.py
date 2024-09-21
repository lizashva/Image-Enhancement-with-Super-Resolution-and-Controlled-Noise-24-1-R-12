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
import matplotlib.pyplot as plt
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
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
vgg_layer = K.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
for l in vgg_layer.layers:
    l.trainable = False

def perceptual_loss(y_true, y_pred):
    '''This function computes the perceptual loss using an already trained VGG layer'''
    y_t = vgg_layer(y_true)
    y_p = vgg_layer(y_pred)
    loss = K.losses.mean_squared_error(y_t, y_p)
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
def image_proc_net_test():
    """
    Function which creates the model to preprocess images for testing mode.
    """
    inputs = tf.keras.layers.Input((None, None, 3))
    # Testing mode: just an upsampling
    x = tf.keras.layers.UpSampling2D((2, 2))(inputs)

    model = tf.keras.models.Model(inputs, x)

    for l in model.layers:
        l.trainable = False

    return model

# Instantiating the test model
image_proc_test = image_proc_net_test()

def image_preprocess_test(image):
    """
    Function which preprocesses an image automatically for testing mode only.

    input:
      - image: np array (image tensor)
    """
    image = np.expand_dims(image, axis=0)
    preprocessed_image = tf.squeeze(image_proc_test(image))
    return preprocessed_image

def read_image(img_path):
    """
    Function which loads an image from a path and converts it into np array
    """
    img = cv2.imread(img_path)
    b, g, r = cv2.split(img)  # cv2 reads BGR instead of canonical RGB
    img = cv2.merge([r, g, b])  # Switching it to RGB
    return img

def preprocess_image(image_path):
    """
    Function which reads and preprocesses an image from a given path for testing mode.
    """
    img = read_image(image_path)
    preprocessed_img = image_preprocess_test(img)
    return preprocessed_img

def show_pictures(img_idx, x_batch, y_batch, model):
    """
    Function which shows 3 images:
    - Ground truth: High Resolution image
    - Low Resolution image
    - Super Resolution image using our trained model
    """
    fig = plt.figure(figsize=(15, 18))

    ax1 = fig.add_subplot(1, 3, 1)
    im = model.predict(np.expand_dims(x_batch[img_idx], axis=0))
    im = np.squeeze(im)
    ax1.imshow(abs(im))
    ax1.set_title('Super Resolution (from LR)')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(x_batch[img_idx])
    ax2.set_title('Low Resolution')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(y_batch[img_idx])
    ax3.set_title('Ground truth')

    plt.show()

def super_resolve_single_image(model, image_path, save_path, plot=False):
    """
    Function to super-resolve a single image using a given model.

    input:
      - model: the trained model for super-resolution
      - image_path: path to the low-resolution image
      - save_path: path to save the super-resolved image
      - plot: boolean flag to indicate whether to plot the images
    """
    img_y = read_image(image_path)
    img_x = image_preprocess_test(img_y)
    low_res_img = np.array(img_x, np.float32) / 255.0
    high_res_pred = model.predict(np.expand_dims(low_res_img, axis=0))
    high_res_img = np.clip(high_res_pred.squeeze(), 0, 1)  # Ensure valid image range
    # show_pictures(0, img_x, img_y, model)
    # Ensure the directory for the save path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the super-resolved image
    high_res_img = (high_res_img * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(high_res_img, cv2.COLOR_RGB2BGR))

    # Verify that the image was saved
    if os.path.isfile(save_path):
        print(f"High-resolution image saved to {save_path}")
    else:
        print(f"Error: Failed to save high-resolution image to {save_path}")

    # # Plot the images if plot is True
    # if plot:
    #     fig = plt.figure(figsize=(15, 18))

    #     ax1 = fig.add_subplot(1, 3, 1)
    #     ax1.imshow(np.squeeze(high_res_img))
    #     ax1.set_title('Super Resolution (from LR)')
    #     ax1.axis('off')

    #     ax2 = fig.add_subplot(1, 3, 2)
    #     ax2.imshow(np.squeeze(low_res_img))
    #     ax2.set_title('Low Resolution')
    #     ax2.axis('off')

    #     ax3 = fig.add_subplot(1, 3, 3)
    #     ax3.imshow(np.squeeze(y_batch[0]))
    #     ax3.set_title('Ground truth')
    #     ax3.axis('off')

    #     plt.show()

    return save_path

