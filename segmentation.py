import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing import image
import numpy as np
import glob
import io

# Constants
BATCH_SIZE = 11
BUFFER_SIZE = 142
IMG_SIZE = (256, 256)
IMAGES_PATH = 'dataset/png_images'
MASKS_PATH = 'dataset/png_masks_8bit'
MASKS_DATA_TYPE = tf.uint8 # Use tf.unit8 for 8-bit and tf.unit16 for 16-bit.
FROZEN_EPOCHS = 50
UNFROZEN_EPOCHS = 56
FROZEN_LR = 0.4e-7
UNFROZEN_LR = 0.07e-7
EARLY_STOPPING_PATIENCE = 10

class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, validation_data, cmap):
        super(ImageLogger, self).__init__()
        self.log_dir = log_dir
        self.validation_data = validation_data
        self.cmap = cmap
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def prepare_log_image(self, image, use_cmap=False):
        buf = io.BytesIO()
        if use_cmap:
            plt.imsave(buf, image, cmap=self.cmap, format='png')
        else:
            # Normalize the image values to the range [0, 1]
            image_min = tf.reduce_min(image)
            image_max = tf.reduce_max(image)
            image_normalized = (image - image_min) / (image_max - image_min)
            image_normalized_uint8 = (image_normalized * 255).numpy().astype(np.uint8)
            plt.imsave(buf, image_normalized_uint8, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=3)
        image = tf.expand_dims(image, 0)  # Add batch dimension

        return image

    def log_multiple_images(self, tag, images, step):
        # Stack images along the batch dimension
        stacked_images = tf.stack(images, axis=0)
        stacked_images = tf.squeeze(stacked_images, axis=1)

        with self.file_writer.as_default():
            tf.summary.image(tag, stacked_images, step=step)

    def log_unique_classes(self, tag, classes, step):
        with self.file_writer.as_default():
            tf.summary.scalar(f"{tag}_unique_classes", len(classes), step=step)
            for class_id in classes:
                tf.summary.scalar(f"{tag}_class_{class_id}", class_id, step=step)

    def on_epoch_end(self, epoch, logs=None):
        val_data = next(iter(self.validation_data))
        images, labels = val_data[0], val_data[1]

        # Define how many images to log
        num_images_to_log = 4
        folder = 'dataset'
        prefix = 'epoch_validation_image'


        # Make predictions on the validation images
        predictions = self.model.predict(images)

        # Expand dimensions to add channel dimension (for grayscale images)
        pred_masks = tf.expand_dims(tf.argmax(predictions, axis=-1, output_type=tf.int32), axis=-1)
        true_masks = tf.expand_dims(tf.cast(labels, tf.uint8), axis=-1)

        # Ensure the true_masks are properly shaped and do not add extra dimensions
        if len(true_masks.shape) == 5:
            true_masks = tf.squeeze(true_masks, axis=-1)

        for i in range(min(num_images_to_log, images.shape[0])):
            # Display predicted mask
            plt.subplot(num_images_to_log, 3, i * 3 + 1)
            plt.imshow(tf.argmax(predictions[i], axis=-1), cmap=cmap)  # Display the predicted mask
            plt.title('Prediction')
            plt.colorbar()
            plt.axis('off')

            # Display ground truth mask (RGB)
            plt.subplot(num_images_to_log, 3, i * 3 + 2)
            plt.imshow(tf.squeeze(labels[i], axis=-1), cmap=cmap)  # Display the RGB ground truth mask
            plt.title('Ground Truth')
            plt.colorbar()
            plt.axis('off')

            # Display input image
            plt.subplot(num_images_to_log, 3, i * 3 + 3)
            plt.imshow(images[i])  # Display the image (no batch dimension)
            plt.title('Input Image')
            plt.colorbar()
            plt.axis('off')

        filename = os.path.join(folder, f'{prefix}_{epoch+1}.png')
        plt.savefig(filename)
        plt.close()

        # Collect images to log
        validation_images_to_log = [self.prepare_log_image(images[i]) for i in range(num_images_to_log)]
        pred_masks_to_log = [self.prepare_log_image(pred_masks[i], True) for i in range(num_images_to_log)]
        true_masks_to_log = [self.prepare_log_image(true_masks[i], True) for i in range(num_images_to_log)]

        # Log the images
        self.log_multiple_images("Validation Images", validation_images_to_log, epoch)
        self.log_multiple_images("Predicted Masks", pred_masks_to_log, epoch)
        self.log_multiple_images("True Masks", true_masks_to_log, epoch)

         # Log unique classes in predicted masks
        unique_classes = np.unique(pred_masks.numpy())
        self.log_unique_classes("Predicted Masks", unique_classes, epoch)

        # Log shapes for debugging
        print(f"Epoch {epoch+1} - Validation Images shape: {images.shape}")
        print(f"Epoch {epoch+1} - Predicted Masks shape: {pred_masks.shape}")
        print(f"Epoch {epoch+1} - True Masks shape: {true_masks.shape}")
        print(f"Epoch {epoch+1} - Unique Classes in Predicted Masks: {unique_classes}")


# Function to find the latest checkpoint
def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'unet_checkpoint_epoch_*.keras'))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[3].split('.')[0]))
        return latest_checkpoint
    return None

# Load image and mask paths
def load_paths(img_dir, mask_dir):
    img_paths = [os.path.join(root, file) for root, _, files in os.walk(img_dir) for file in files]
    mask_paths = [os.path.join(root, file) for root, _, files in os.walk(mask_dir) for file in files]
    return sorted(img_paths), sorted(mask_paths)

# Load and preprocess data
def parse_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3, dtype=tf.uint8) # load with 3 channels, for RGB. Each one is Integer. 8-bit precision.
    img = tf.image.resize(img, IMG_SIZE) # Resize to IMG_SIZE.
    return tf.clip_by_value(tf.cast(img, tf.float32) / 255.0, 0.0, 1.0) # Normalize to 0 until 1, and add clipping in case it gets out of range.

# Function to read classes and colors from label_map_colors file
def read_classes_and_colors(label_map_colors_file):
    with open(label_map_colors_file, 'r') as file:
        lines = file.readlines()
    
    class_color_map = {}

    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.strip().split(':')
        class_number = int(parts[0])
        color_rgb = tuple(map(int, parts[2].strip('()').split(',')))
        class_color_map[class_number] = color_rgb

    sorted_class_numbers = sorted(class_color_map.keys())
    sorted_colors = [class_color_map[class_number] for class_number in sorted_class_numbers]

    return len(sorted_class_numbers), sorted_colors

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

            plt.subplot(num_images, 2, i * 2 + 2)
            plt.imshow(tf.squeeze(masks[i], axis=-1), cmap=cmap)
            plt.title('Ground Truth Mask')
            plt.colorbar()
            plt.axis('off')
        
        filename = os.path.join(folder, f'{prefix}_{idx}.png')
        plt.savefig(filename)
        plt.close()

def unet_model(downstack, output_channels):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    skips = downstack(inputs)
    x = skips[-1]
    for up, skip in zip([pix2pix.upsample(f, 3) for f in [1024, 512, 256, 128, 64]], reversed(skips[:-1])):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(output_channels, 1, padding='same')(x)  # Add final Conv2D layer to match dimensions
    return tf.keras.Model(inputs, x)

# Preprocess function
def preprocess(image, mask):
    return image, mask

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

def visualize_and_save_predictions(predictions, images, cmap, folder='predictions', prefix='prediction'):
    num_images = len(predictions)
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for i in range(num_images):
        plt.figure(figsize=(10, 5))
        
        # Display predicted mask
        plt.subplot(1, 2, 1)
        plt.imshow(tf.argmax(predictions[i], axis=-1), cmap=cmap)  # Display the predicted mask
        plt.title('Prediction')
        plt.colorbar()
        plt.axis('off')

        # Display input image
        plt.subplot(1, 2, 2)
        plt.imshow(tf.squeeze(images[i]))  # Display the image (no batch dimension)
        plt.title('Input Image')
        plt.colorbar()
        plt.axis('off')
        
        filename = os.path.join(folder, f'{prefix}_{i}.png')
        plt.savefig(filename)
        plt.close()

label_map_colors_file = 'dataset/label_map_colors_8.txt'
# Read classes and colors for cmap
classes, colors = read_classes_and_colors(label_map_colors_file)
print("classes=", classes)
print("colors=", colors)
# Normalize the RGB values to the [0, 1] range
colors = [(r/255, g/255, b/255) for r, g, b in colors]
# Create the ListedColormap
cmap = ListedColormap(colors)

checkpoint_dir = 'model/custom-model/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_filepath = os.path.join(checkpoint_dir, 'unet_checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras')
# Load the latest checkpoint if available
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
initial_epoch = 0
if latest_checkpoint:
    initial_epoch = int(latest_checkpoint.split('_')[3].split('.')[0])
    if initial_epoch < FROZEN_EPOCHS + UNFROZEN_EPOCHS:
        print(f"Resuming training from {latest_checkpoint}")
else:
    print("Starting training from scratch")

if initial_epoch < FROZEN_EPOCHS + UNFROZEN_EPOCHS:
    image_paths, mask_paths = load_paths(IMAGES_PATH, MASKS_PATH)

    dataset = load_data(image_paths, mask_paths)

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


    visualize_samples(train_dataset)

    # Model setup
    base_model = tf.keras.applications.DenseNet121(input_shape=[256, 256, 3], include_top=False, weights='imagenet')
    skip_names = ['conv1/relu', 'pool2_relu', 'pool3_relu', 'pool4_relu', 'relu']
    skip_outputs = [base_model.get_layer(name).output for name in skip_names]
    downstack = tf.keras.Model(inputs=base_model.input, outputs=skip_outputs)
    downstack.trainable = False

    model = unet_model(downstack, classes)

    model.load_weights(latest_checkpoint)
   
    # Early stopping.
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True)
    # Define checkpoint callbacks to save all checkpoints
    checkpoint_all = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

    # Define a log directory for TensorBoard
    log_dir = 'model/custom-model/logs'
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Instantiate the image logger callback
    image_logger = ImageLogger(log_dir=log_dir, validation_data=val_dataset, cmap=cmap)

    callbacks_list = [checkpoint_all, early_stopping, tensorboard_callback, image_logger]



    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Training
    # Check if training should continue from the frozen phase
    if initial_epoch < FROZEN_EPOCHS:
        # Custom optimizer with gradient clipping
        optimizer_frozen = tf.keras.optimizers.Adam(learning_rate=FROZEN_LR, clipnorm=1.0)
        model.compile(optimizer=optimizer_frozen, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        history_frozen = model.fit(
            train_dataset, 
            validation_data=val_dataset, 
            steps_per_epoch=train_size // BATCH_SIZE, 
            validation_steps=(len(image_paths) * 26 - train_size) // BATCH_SIZE, 
            initial_epoch=initial_epoch,
            epochs=FROZEN_EPOCHS, 
            callbacks=callbacks_list)
        
        initial_epoch = FROZEN_EPOCHS  # Update initial_epoch to start unfrozen phase

    # Unfreeze layers and continue training
    downstack.trainable = True

    # Custom optimizer with gradient clipping
    optimizer_unfrozen = tf.keras.optimizers.Adam(learning_rate=UNFROZEN_LR, clipnorm=1.0)
    model.compile(optimizer=optimizer_unfrozen, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


    if initial_epoch < FROZEN_EPOCHS + UNFROZEN_EPOCHS:
        # Continue training with unfrozen layers
        history_unfrozen = model.fit(
            train_dataset, 
            validation_data=val_dataset, 
            initial_epoch=max(initial_epoch, FROZEN_EPOCHS), # Continue from the next epoch
            steps_per_epoch=train_size // BATCH_SIZE, 
            validation_steps=(len(image_paths) * 26 - train_size) // BATCH_SIZE, 
            epochs=FROZEN_EPOCHS + UNFROZEN_EPOCHS,
            callbacks=callbacks_list)

    # Define directory for the full model
    full_model_dir = 'model/custom-model'
    os.makedirs(full_model_dir, exist_ok=True)
    full_model_filepath = os.path.join(full_model_dir, 'unet_model_complete.keras')

    # Save the entire model
    model.save(full_model_filepath)

    visualize_predictions(model, val_dataset)
else:
    print(f"No training needed, epoch of checkpoint={initial_epoch} and FROZEN_EPOCHS=${FROZEN_EPOCHS} + UNFROZEN_EPOCHS={UNFROZEN_EPOCHS}")


# Inference.

# Load the entire model for inference
loaded_model = tf.keras.models.load_model('model/custom-model/unet_model_complete.keras')

# Example usage
img_path = 'predictions/input_0.png'
preprocessed_image = parse_image(img_path)

# Add a batch dimension before making predictions
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

# Make predictions
predictions = loaded_model.predict(preprocessed_image)

# Post-process the predictions if needed
#predicted_mask = predictions[0]  # Example to extract the prediction mask


# Visualize and save predictions
visualize_and_save_predictions(predictions, [preprocessed_image], cmap)
