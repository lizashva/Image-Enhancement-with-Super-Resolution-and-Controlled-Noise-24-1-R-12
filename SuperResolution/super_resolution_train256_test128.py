import cv2
import tensorflow as tf         # tensorflow
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input, MaxPool2D, GaussianNoise, UpSampling2D
import tensorflow.keras as K
from tensorflow.keras.layers import Lambda
import math
# defining the image input and our batch size
input_size = 128  # 128x128x3 images (training_set)
batch_size = 16

# Perceptual loss function

# load pre-trained (imagenet) vgg network, excluding fully-connected layer on the top
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None,None,3))
vgg_layer = K.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
# make the net not trainable
for l in vgg_layer.layers: l.trainable=False

print(vgg_layer.summary())

def perceptual_loss(y_true,y_pred):
    '''This function computes the perceptual loss using an already trained VGG layer'''
    y_t=vgg_layer(y_true)
    y_p=vgg_layer(y_pred)
    loss = K.losses.mse(y_t, y_p)
    return loss

# defining other metrics:
def psnr(y_true,y_pred):
    return tf.image.psnr(y_true,y_pred,1.0)
def ssim(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,1.0)

def image_proc_net(training_mode=True):
    """
    Function which creates the model to preprocess images.
    2 different types of model with different layers depending
    on whether it is in training mode or in testing mode
    """
    inputs = Input((None, None, 3))
    if training_mode:
        # training mode: downsampling, gaussian noise and upsampling
        x = MaxPool2D(pool_size=(2,2))(inputs)
        x = GaussianNoise(stddev=0.1)(x)
        x = UpSampling2D((2,2))(x)
    else:
        # testing mode: just an upsampling
        x = K.layers.UpSampling2D((2, 2))(inputs)

    model = K.models.Model(inputs, x)

    for l in model.layers: l.trainable=False

    return model

# istantiating the two models (for train and test)
image_proc_train = image_proc_net(training_mode=True)
image_proc_test = image_proc_net(training_mode=False)

#training_mode
print(image_proc_train.summary())

#testing_mode
print(image_proc_test.summary())

def image_preprocess(image, training_mode=True):
    """
    Function which preprocess automatically an image
    wether it's in training or testing mode

    input:
      - image: np array (image tensor)
      - training_mode: binary flag
    """
    image = np.expand_dims(image,axis=0)
    if training_mode:
        return tf.squeeze(image_proc_train(image))
    else:
        return tf.squeeze(image_proc_test(image))

# reading image
def read_image(img_path):
    """
    function which loads an image from a path and convert it into np array
    """
    img = cv2.imread(img_path)
    b,g,r = cv2.split(img)   # cv2 reads BGR instead of canonical RGB
    img = cv2.merge([r,g,b]) # Switching it to RGB

    return img

import os
import numpy as np

all_images = []  # All train image paths

# 128x128 in training phase
path = 'Linnaeus_5_bigger/train_256'

# Check if the path exists
if not os.path.exists(path):
    print(f"Path does not exist: {path}")
else:
    files = os.listdir(path)
    for file in files:
        path_file = os.path.join(path, file)
        for dirpath, dirnames, filenames in os.walk(path_file):
            # Append full path to filenames
            filenames = [os.path.join(dirpath, x) for x in filenames]
            all_images.extend(filenames)

    # Picking only 2000 images
    all_images = all_images[:2000]
    np.random.shuffle(all_images)  # Random shuffling

    # Keep 85% for train and 15% for validation
    train_len = int(len(all_images) * 0.85)

    train_list = all_images[:train_len]
    val_list = all_images[train_len:]

    print(f"Total images: {len(all_images)}")
    print(f"Training images: {len(train_list)}")
    print(f"Validation images: {len(val_list)}")


# test
# for the test set we opted to take the same dataset
# but with images of size 64x64 instead of 128x128

test_list = []

path = 'Linnaeus_5_bigger/test_128'
files = os.listdir(path)
for file in files:
    path_file = path+'/'+file
    for (dirpath, dirnames, filenames) in os.walk(path_file):
        filenames = list(map(lambda x:path_file+'/'+x,filenames))
        test_list.extend(filenames)

np.random.shuffle(test_list) # random shuffling

test_list = test_list[:500]

print('images: ')
print('train:', len(train_list), 'val:', len(val_list), 'test:', len(test_list))

# defining train, validation and test generator

def train_generator():
    global batch_size
    while True:
        for start in range(0, len(train_list), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(train_list))
                    ids_train_batch = train_list[start:end]
                    for i,ids in enumerate(ids_train_batch):
                        img_y = read_image(ids)
                        img_x = image_preprocess(img_y, training_mode=True)
                        x_batch.append(np.array(img_x,np.float32)/255.)
                        y_batch.append(np.array(img_y,np.float32)/255.)
                    x_batch = np.array(x_batch)
                    y_batch = np.array(y_batch)
                    yield x_batch,y_batch

def valid_generator():
    global batch_size
    while True:
        for start in range(0, len(val_list), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(val_list))
                    ids_val_batch = val_list[start:end]
                    for i,ids in enumerate(ids_val_batch):
                        img_y = read_image(ids)
                        img_x = image_preprocess(img_y, training_mode=True)
                        x_batch.append(np.array(img_x,np.float32)/255.)
                        y_batch.append(np.array(img_y,np.float32)/255.)
                    x_batch = np.array(x_batch)
                    y_batch = np.array(y_batch)
                    yield x_batch,y_batch


def test_generator():
    global batch_size
    while True:
        for start in range(0, len(test_list), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(test_list))
                    ids_test_batch = test_list[start:end]
                    for i,ids in enumerate(ids_test_batch):
                        img_y = read_image(ids)
                        img_x = image_preprocess(img_y, training_mode=False)
                        x_batch.append(np.array(img_x,np.float32)/255.)
                        y_batch.append(np.array(img_y,np.float32)/255.)
                    x_batch = np.array(x_batch)
                    y_batch = np.array(y_batch)
                    yield x_batch,y_batch

# pixel shuffle
def pixel_shuffle(scale):
    '''
    This function implements pixel shuffling.
    ATTENTION: the scale should be bigger than 2, otherwise just returns the input.
    '''
    # if scale > 1:
    #     return lambda x: tf.nn.depth_to_space(x, scale)
    # else:
    #     return lambda x:x
    if scale > 1:
        return Lambda(lambda x: tf.nn.depth_to_space(x, scale))
    else:
        return Lambda(lambda x: x)  # Return identity function if scale <= 1

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

model = RUNet()
model.summary()


steps_per_epoch = math.ceil(len(train_list) / batch_size)
validation_steps = math.ceil(len(val_list) / batch_size)


opt=K.optimizers.Adam(learning_rate=0.001) # Adam optimizer
model.compile(optimizer=opt,loss=perceptual_loss,metrics=[psnr,ssim,K.losses.mse])
history = model.fit(train_generator(),
                              steps_per_epoch=steps_per_epoch,
                              epochs=20,
                              verbose=1,
                              validation_data=valid_generator(),
                              shuffle=True,
                              validation_steps=validation_steps)

path = '/super_resolution_weights.h5'

# Ensure the directory exists, create it if necessary
directory = os.path.dirname(path)
if not os.path.exists(directory):
    os.makedirs(directory)


# Assuming 'model' is your trained model
model.save_weights(path)
print(f"Model weights saved to '{path}'.")


