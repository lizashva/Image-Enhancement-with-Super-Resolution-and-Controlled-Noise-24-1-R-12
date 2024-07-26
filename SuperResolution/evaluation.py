"""## Part 4 - Experimental evaluation

#### Metrics: history

We show in the following part the results in term of metrics during the training epochs.
"""
import matplotlib.pyplot as plt # function to show images
import os
import cv2
import numpy as np

def history_results(history, par1='loss', par2='val_loss', title='loss'):
    """
    Plot the history of the the 2 metrics (par1, par2) during
    the training (epochs)
    """
    plt.plot(history.history[par1])
    plt.plot(history.history[par2])
    plt.title(title)
    plt.ylabel(par1)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()

    # Create the 'evaluation' directory if it does not exist
    os.makedirs('evaluation', exist_ok=True)
    # Save the plot to the 'evaluation' directory
    plt.savefig(os.path.join('evaluation', filename))
    plt.close()
    return



history_results(history, 'psnr', 'val_psnr', 'psnr')

history_results(history, 'ssim', 'val_ssim', 'ssim')

history_results(history, 'mean_squared_error', 'val_mean_squared_error', 'means_squared_error')

"""#### Showing some visual results of Super-Resolution

In testing mode, 64x64 pixels images are used and then the result it's compared with 128x128 pixels images.
"""


def show_pictures(img_idx, x_batch, y_batch):
    """
    Function which shows 3 images:
    - Ground truth: High Resolution image
    - Low Resolution image
    - Super Resolution image using our trained model
    """
    fig = plt.figure(figsize=(15,18))

    ax1 = fig.add_subplot(1,3,1)
    im = model(np.expand_dims(x_batch[img_idx],axis=0))
    im = np.squeeze(im)
    #ax1.imshow((abs(im) * 255).astype(np.uint8))
    ax1.imshow(abs(im))
    ax1.set_title('Super Resolution (from LR)')

    ax2 = fig.add_subplot(1,3,2)
    #ax2.imshow(x_batch[img_idx] * 255).astype(np.uint8))
    ax2.imshow(x_batch[img_idx])
    ax2.set_title('Low Resolution')

    ax3 = fig.add_subplot(1,3,3)
    #ax3.imshow((y_batch[img_idx] * 255).astype(np.uint8))
    ax3.imshow(y_batch[img_idx])
    ax3.set_title('Ground truth')

    os.makedirs('results', exist_ok=True)
    # Save the montage to the 'results' directory
    plt.savefig(os.path.join('results'))
    plt.close()

    return

# trying to visualize the results with some training images

chunk_size=20

x_chunk = []
y_chunk = []

ids_train_chunk = train_list[0:chunk_size]
for i, ids in enumerate(ids_train_chunk):
    img_y = read_image(ids)
    img_x = image_preprocess(img_y)
    x_chunk.append(np.array(img_x,np.float32)/255.)
    y_chunk.append(np.array(img_y,np.float32)/255.)
x_chunk = np.array(x_chunk)
y_chunk = np.array(y_chunk)

x_chunk.shape

show_pictures(7, x_chunk, y_chunk)
show_pictures(16, x_chunk, y_chunk)
show_pictures(19, x_chunk, y_chunk)


def read_image(img_path):
    """
    Reads an image from a file path.
    Converts from BGR to RGB format.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image at {img_path}")
        return None
    b, g, r = cv2.split(img)   # cv2 reads BGR instead of canonical RGB
    img = cv2.merge([r, g, b]) # Switching it to RGB
    return img

def get_image_paths(root_dir, size):
    """
    Collects image paths from the directory and filters by size.
    """
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if size in filename:  # Assuming size (e.g., '64' or '256') is in the filename
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

# Assuming image_preprocess is defined elsewhere

# Get all 256x256 image paths for training
train_root_dir = 'Linnaeus_5_bigger/train_256'
train_size = '256'
train_list = get_image_paths(train_root_dir, train_size)

# Get all 128x128 image paths for testing
test_root_dir = 'Linnaeus_5_bigger/test_128'
test_size = '128'
test_list = get_image_paths(test_root_dir, test_size)

# Prepare training data
x_train_chunk = []
y_train_chunk = []
train_chunk_size = 10  # Adjust the chunk size as needed
ids_train_chunk = train_list[:train_chunk_size]

for i, ids in enumerate(ids_train_chunk):
    # Construct the path for the corresponding 256x256 image
    ids_256 = ids.replace(train_size, train_size)
    print(f"Processing: 256x256 image at {ids_256}")

    # Check if the 256x256 image file exists
    if not os.path.isfile(ids_256):
        print(f"256x256 image file does not exist: {ids_256}")
        continue

    img_x = read_image(ids_256)     # x: 256x256 img
    if img_x is None:
        print(f"Skipping image at {ids_256}")
        continue

    img_x = image_preprocess(img_x, training_mode=False)

    x_train_chunk.append(np.array(img_x, np.float32) / 255.)
    y_train_chunk.append(np.array(img_x, np.float32) / 255.)  # Same image for both x and y for training

x_train_chunk = np.array(x_train_chunk)
y_train_chunk = np.array(y_train_chunk)

print(f"Processed {len(x_train_chunk)} training image pairs.")

# Prepare testing data
x_test_chunk = []
y_test_chunk = []
test_chunk_size = 10  # Adjust the chunk size as needed
ids_test_chunk = test_list[:test_chunk_size]

for i, ids in enumerate(ids_test_chunk):
    # Construct the path for the corresponding 128x128 image
    ids_128 = ids.replace(test_size, test_size)
    print(f"Processing: 128x128 image at {ids_128}")

    # Check if the 128x128 image file exists
    if not os.path.isfile(ids_128):
        print(f"128x128 image file does not exist: {ids_128}")
        continue

    img_x = read_image(ids_128)     # x: 128x128 img
    if img_x is None:
        print(f"Skipping image at {ids_128}")
        continue

    img_x = image_preprocess(img_x, training_mode=False)

    x_test_chunk.append(np.array(img_x, np.float32) / 255.)
    y_test_chunk.append(np.array(img_x, np.float32) / 255.)  # Same image for both x and y for testing

x_test_chunk = np.array(x_test_chunk)
y_test_chunk = np.array(y_test_chunk)

print(f"Processed {len(x_test_chunk)} testing image pairs.")

x_chunk.shape

show_pictures(4, x_chunk, y_chunk)
show_pictures(5, x_chunk, y_chunk)
show_pictures(9, x_chunk, y_chunk)