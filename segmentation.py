import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np

# Constants
classes = 42
BATCH_SIZE = 11
BUFFER_SIZE = 142
IMG_SIZE = (256, 256)
IMAGES_PATH = 'png_images'
MASKS_PATH = 'png_masks_8bit'
MASKS_DATA_TYPE = tf.uint8 # Use tf.unit8 for 8-bit and tf.unit16 for 16-bit.
FROZEN_EPOCHS = 50
UNFROZEN_EPOCHS = 150
FROZEN_LR = 0.4e-7
UNFROZEN_LR = 0.07e-7


# Load image and mask paths
def load_paths(img_dir, mask_dir):
    img_paths = [os.path.join(root, file) for root, _, files in os.walk(img_dir) for file in files]
    mask_paths = [os.path.join(root, file) for root, _, files in os.walk(mask_dir) for file in files]
    return sorted(img_paths), sorted(mask_paths)

image_paths, mask_paths = load_paths(IMAGES_PATH, MASKS_PATH)

# Load and preprocess data
def parse_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3, dtype=tf.uint8) # load with 3 channels, for RGB. Each one is Integer. 8-bit precision.
    img = tf.image.resize(img, IMG_SIZE) # Resize to IMG_SIZE.
    return tf.clip_by_value(tf.cast(img, tf.float32) / 255.0, 0.0, 1.0) # Normalize to 0 until 1, and add clipping in case it gets out of range.

# def parse_mask(mask_path):
#     mask = tf.io.read_file(mask_path)
#     mask = tf.image.decode_png(mask, channels=1, dtype=MASKS_DATA_TYPE)  # Load with 1 channel, for Grayscale. The value is Integer.

#     mask = tf.image.resize(mask, IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Resize mask, use Nearest Neighbor.
#     mask = tf.clip_by_value(mask, 0, classes - 1)  # Clip any values out of range. Range is 0 until classes-1. Classes include 0 which is the background.

#     mask = tf.cast(mask, tf.int32)  # Cast to Integer 32-bit.

#     # Update here: Find unique values in the mask using TensorFlow operations
#     mask_flat = tf.reshape(mask, [-1])
#     unique_parsed_values, _ = tf.unique(mask_flat)

#     print("Unique values in parsed_mask:")
#     print(unique_parsed_values)
#     return mask

def parse_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1, dtype=MASKS_DATA_TYPE) # load with 1 channel, for Grayscale. The value is Integer.

    mask = tf.image.resize(mask, IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # Resize mask, use Nearest Neighbor.
    mask = tf.clip_by_value(mask, 0, classes - 1) # Clip any values out of range. Range is 0 until classes-1. Classes include 0 which is the background.
    return tf.cast(mask, tf.int32) # Cast to Integer 32-bit.

def load_data(img_paths, mask_paths):
    images = tf.data.Dataset.from_tensor_slices(img_paths).map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    masks = tf.data.Dataset.from_tensor_slices(mask_paths).map(parse_mask, num_parallel_calls=tf.data.AUTOTUNE)
    return tf.data.Dataset.zip((images, masks))

dataset = load_data(image_paths, mask_paths)

# Augmentation functions
def augment_flip(image, mask):
    seed = tf.random.experimental.stateless_split([1, 2], num=1)[0]  # Same seed for image and mask transformations

    # Flip left-right
    image = tf.image.stateless_random_flip_left_right(image, seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed)

    # Flip up-down
    image = tf.image.stateless_random_flip_up_down(image, seed)
    mask = tf.image.stateless_random_flip_up_down(mask, seed)

    return image, mask

def augment_rotate(image, mask):
    # Rotate 90 degrees
    angle = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, angle)
    mask = tf.image.rot90(mask, angle)

    return image, mask

def augment_brightness_contrast(image, mask):
    seed = tf.random.experimental.stateless_split([1, 2], num=1)[0]  # Same seed for image and mask transformations

    # Random brightness and contrast on images only
    image = tf.image.stateless_random_brightness(image, 0.2, seed)
    image = tf.image.stateless_random_contrast(image, 0.8, 1.2, seed)

    return image, mask

def augment_hue_saturation(image, mask):
    seed = tf.random.experimental.stateless_split([1, 2], num=1)[0]  # Same seed for image and mask transformations

    # Random hue
    image = tf.image.stateless_random_hue(image, 0.2, seed)
    # Random saturation
    image = tf.image.stateless_random_saturation(image, 0.6, 1.4, seed)

    return image, mask

def augment_zoom(image, mask):
    # Random zoom using resize with crop or pad
    scale = tf.random.uniform([], minval=0.8, maxval=1.2, dtype=tf.float32)
    new_size = tf.cast(scale * IMG_SIZE[0], tf.int32)
    image = tf.image.resize_with_crop_or_pad(image, new_size, new_size)
    mask = tf.image.resize_with_crop_or_pad(mask, new_size, new_size)
    image = tf.image.resize(image, IMG_SIZE)
    mask = tf.image.resize(mask, IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, mask

def augment_combo1(image, mask):
    image, mask = augment_flip(image, mask)
    image, mask = augment_rotate(image, mask)
    image, mask = augment_brightness_contrast(image, mask)
    image, mask = augment_hue_saturation(image, mask)
    image, mask = augment_zoom(image, mask)

    return image, mask

def augment_combo2(image, mask):
    image, mask = augment_rotate(image, mask)
    image, mask = augment_flip(image, mask)
    image, mask = augment_brightness_contrast(image, mask)
    image, mask = augment_hue_saturation(image, mask)
    image, mask = augment_zoom(image, mask)
    
    return image, mask

def augment_combo3(image, mask):
    image, mask = augment_brightness_contrast(image, mask)
    image, mask = augment_hue_saturation(image, mask)
    image, mask = augment_zoom(image, mask)
    image, mask = augment_rotate(image, mask)
    image, mask = augment_flip(image, mask)
    
    return image, mask

def augment_combo4(image, mask):
    image, mask = augment_brightness_contrast(image, mask)
    image, mask = augment_hue_saturation(image, mask)
    image, mask = augment_zoom(image, mask)
    image, mask = augment_flip(image, mask)
    image, mask = augment_rotate(image, mask)
    
    return image, mask

def augment_combo5(image, mask):
    image, mask = augment_zoom(image, mask)
    image, mask = augment_brightness_contrast(image, mask)
    image, mask = augment_hue_saturation(image, mask)
    image, mask = augment_flip(image, mask)
    image, mask = augment_rotate(image, mask)

    return image, mask

# Apply augmentations and combine with original dataset
augmented_flip = dataset.map(augment_flip, num_parallel_calls=tf.data.AUTOTUNE)
augmented_rotate = dataset.map(augment_rotate, num_parallel_calls=tf.data.AUTOTUNE)
augmented_brightness_contrast = dataset.map(augment_brightness_contrast, num_parallel_calls=tf.data.AUTOTUNE)
augmented_hue_saturation = dataset.map(augment_hue_saturation, num_parallel_calls=tf.data.AUTOTUNE)
augmented_zoom = dataset.map(augment_zoom, num_parallel_calls=tf.data.AUTOTUNE)

augmented_flip2 = dataset.map(augment_flip, num_parallel_calls=tf.data.AUTOTUNE)
augmented_rotate2 = dataset.map(augment_rotate, num_parallel_calls=tf.data.AUTOTUNE)
augmented_brightness_contrast2 = dataset.map(augment_brightness_contrast, num_parallel_calls=tf.data.AUTOTUNE)
augmented_hue_saturation2 = dataset.map(augment_hue_saturation, num_parallel_calls=tf.data.AUTOTUNE)
augmented_zoom2 = dataset.map(augment_zoom, num_parallel_calls=tf.data.AUTOTUNE)

augmented_flip3 = dataset.map(augment_flip, num_parallel_calls=tf.data.AUTOTUNE)
augmented_rotate3 = dataset.map(augment_rotate, num_parallel_calls=tf.data.AUTOTUNE)
augmented_brightness_contrast3 = dataset.map(augment_brightness_contrast, num_parallel_calls=tf.data.AUTOTUNE)
augmented_hue_saturation3 = dataset.map(augment_hue_saturation, num_parallel_calls=tf.data.AUTOTUNE)
augmented_zoom3 = dataset.map(augment_zoom, num_parallel_calls=tf.data.AUTOTUNE)

augmented_flip4 = dataset.map(augment_flip, num_parallel_calls=tf.data.AUTOTUNE)
augmented_rotate4 = dataset.map(augment_rotate, num_parallel_calls=tf.data.AUTOTUNE)
augmented_brightness_contrast4 = dataset.map(augment_brightness_contrast, num_parallel_calls=tf.data.AUTOTUNE)
augmented_hue_saturation4 = dataset.map(augment_hue_saturation, num_parallel_calls=tf.data.AUTOTUNE)
augmented_zoom4 = dataset.map(augment_zoom, num_parallel_calls=tf.data.AUTOTUNE)

augmented_combo1 = dataset.map(augment_combo1, num_parallel_calls=tf.data.AUTOTUNE)
augmented_combo2 = dataset.map(augment_combo2, num_parallel_calls=tf.data.AUTOTUNE)
augmented_combo3 = dataset.map(augment_combo3, num_parallel_calls=tf.data.AUTOTUNE)
augmented_combo4 = dataset.map(augment_combo4, num_parallel_calls=tf.data.AUTOTUNE)
augmented_combo5 = dataset.map(augment_combo5, num_parallel_calls=tf.data.AUTOTUNE)

combined_dataset1 = dataset.concatenate(augmented_flip).concatenate(augmented_rotate).concatenate(augmented_brightness_contrast).concatenate(augmented_hue_saturation).concatenate(augmented_zoom)
combined_dataset2 = combined_dataset1.concatenate(augmented_flip2).concatenate(augmented_rotate2).concatenate(augmented_brightness_contrast2).concatenate(augmented_hue_saturation2).concatenate(augmented_zoom2)
combined_dataset3 = combined_dataset2.concatenate(augmented_flip3).concatenate(augmented_rotate3).concatenate(augmented_brightness_contrast3).concatenate(augmented_hue_saturation3).concatenate(augmented_zoom3)
combined_dataset4 = combined_dataset3.concatenate(augmented_flip4).concatenate(augmented_rotate4).concatenate(augmented_brightness_contrast4).concatenate(augmented_hue_saturation4).concatenate(augmented_zoom4)
combined_dataset = combined_dataset4.concatenate(augmented_combo1).concatenate(augmented_combo2).concatenate(augmented_combo3).concatenate(augmented_combo4).concatenate(augmented_combo5)

train_size = int(0.8 * len(image_paths) * 26)  # 80% train set. Original dataset plus 25 augmented datasets = 26 times the original dataset.
train_dataset = combined_dataset.take(train_size).cache().repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = combined_dataset.skip(train_size).take(len(image_paths) * 26 - train_size).batch(BATCH_SIZE) # Original dataset plus 25 augmented datasets = 26 times the original dataset. Minus the train_size, this will be used as test/validation dataset.


# Define the RGB color values as a list of tuples
colors = [
    (0, 0, 0),
    (138, 255, 237),
    (133, 26, 149),
    (108, 3, 54),
    (93, 93, 93),
    (40, 58, 241),
    (77, 59, 77),
    (118, 78, 88),
    (147, 48, 48),
    (255, 255, 255),
    (240, 120, 240),
    (20, 103, 68),
    (217, 224, 255),
    (164, 193, 255),
    (161, 19, 255),
    (151, 151, 127),
    (249, 201, 31),
    (92, 113, 70),
    (15, 14, 181),
    (125, 15, 12),
    (108, 157, 9),
    (72, 89, 157),
    (232, 180, 8),
    (108, 51, 181),
    (255, 16, 19),
    (220, 168, 112),
    (201, 198, 198),
    (17, 174, 140),
    (177, 179, 107),
    (201, 132, 132),
    (234, 187, 103),
    (203, 164, 180),
    (139, 73, 171),
    (198, 195, 234),
    (108, 0, 51),
    (47, 163, 184),
    (251, 17, 125),
    (156, 123, 137),
    (224, 148, 177),
    (243, 217, 243),
    (227, 227, 227),
    (183, 13, 95)
]

# Normalize the RGB values to the [0, 1] range
colors = [(r/255, g/255, b/255) for r, g, b in colors]

# Create the ListedColormap
cmap = ListedColormap(colors)



# Visualize samples and save figures
def visualize_samples(dataset, num_images=5, folder='figures', prefix='sample'):
    os.makedirs(folder, exist_ok=True)
    
    for idx, (images, masks) in enumerate(dataset.take(1)):
        plt.figure(figsize=(10, 10))
        for i in range(min(num_images, images.shape[0])):
            print(f'Image {i} - min: {tf.reduce_min(images[i]).numpy()}, max: {tf.reduce_max(images[i]).numpy()}')
            plt.subplot(num_images, 2, i * 2 + 1)
            plt.imshow(images[i])
            plt.title('Input Image')
            plt.colorbar()
            plt.axis('off')

            # unique_vals = np.unique(tf.squeeze(masks[i], axis=-1).numpy())
            # print(f'Unique values in mask {i}: {unique_vals}')
            plt.subplot(num_images, 2, i * 2 + 2)
            plt.imshow(tf.squeeze(masks[i], axis=-1), cmap=cmap)
            plt.title('Ground Truth Mask')
            plt.colorbar()
            plt.axis('off')
        
        filename = os.path.join(folder, f'{prefix}_{idx}.png')
        plt.savefig(filename)
        plt.close()

visualize_samples(train_dataset)

# Model setup
base_model = tf.keras.applications.DenseNet121(input_shape=[256, 256, 3], include_top=False, weights='imagenet')
skip_names = ['conv1/relu', 'pool2_relu', 'pool3_relu', 'pool4_relu', 'relu']
skip_outputs = [base_model.get_layer(name).output for name in skip_names]
downstack = tf.keras.Model(inputs=base_model.input, outputs=skip_outputs)
downstack.trainable = False

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    skips = downstack(inputs)
    x = skips[-1]
    for up, skip in zip([pix2pix.upsample(f, 3) for f in [1024, 512, 256, 128, 64]], reversed(skips[:-1])):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(output_channels, 1, padding='same')(x)  # Add final Conv2D layer to match dimensions
    return tf.keras.Model(inputs, x)

model = unet_model(classes)

# Custom optimizer with gradient clipping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Preprocess function
def preprocess(image, mask):
    return image, mask

train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Training
optimizer_frozen = tf.keras.optimizers.Adam(learning_rate=FROZEN_LR, clipnorm=1.0)
model.compile(optimizer=optimizer_frozen, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, steps_per_epoch=train_size // BATCH_SIZE, validation_steps=(len(image_paths) * 26 - train_size) // BATCH_SIZE, epochs=FROZEN_EPOCHS, callbacks=[early_stopping])

downstack.trainable = True

optimizer_unfrozen = tf.keras.optimizers.Adam(learning_rate=UNFROZEN_LR, clipnorm=1.0)
model.compile(optimizer=optimizer_unfrozen, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, steps_per_epoch=train_size // BATCH_SIZE, validation_steps=(len(image_paths) * 26 - train_size) // BATCH_SIZE, epochs=UNFROZEN_EPOCHS, callbacks=[early_stopping])

# Visualize predictions and save figures
def visualize_predictions(model, dataset, num_images=5, folder='figures', prefix='prediction'):
    os.makedirs(folder, exist_ok=True)
    for idx, (images, masks) in enumerate(dataset.take(1)):
        predictions = model(images, training=False)  # Model inference
        plt.rcParams["figure.figsize"] = (10, 10)
        plt.figure(figsize=(10, 10))

        for i in range(min(num_images, images.shape[0])):
            # Display predicted mask
            plt.subplot(num_images, 3, i * 3 + 1)
            plt.imshow(tf.argmax(predictions[i], axis=-1), cmap=cmap)  # Display the predicted mask
            plt.title('Prediction')
            plt.colorbar()
            plt.axis('off')

            # Display ground truth mask (RGB)
            plt.subplot(num_images, 3, i * 3 + 2)
            plt.imshow(tf.squeeze(masks[i], axis=-1), cmap=cmap)  # Display the RGB ground truth mask
            plt.title('Ground Truth')
            plt.colorbar()
            plt.axis('off')

            # Display input image
            plt.subplot(num_images, 3, i * 3 + 3)
            plt.imshow(images[i])  # Display the image (no batch dimension)
            plt.title('Input Image')
            plt.colorbar()
            plt.axis('off')

        filename = os.path.join(folder, f'{prefix}_{idx}.png')
        plt.savefig(filename)
        plt.close()


visualize_predictions(model, val_dataset)
