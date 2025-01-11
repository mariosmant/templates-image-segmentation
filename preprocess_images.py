import os
from PIL import Image
import tensorflow as tf
import numpy as np

BIT_DEPTH_NUMBER = 8
BIT_DEPTH_8_MAX_VALUE = 255
BIT_DEPTH_16_MAX_VALUE = 65535

unique_colors = {}

# Function to convert jpg_images to png_images without ICC profile
def convert_images(input_folder, output_folder, convert_to_png=True):
    for filename in os.listdir(input_folder):
        file_root, file_ext = os.path.splitext(filename)
        if file_ext.lower() in ('.jpg', '.jpeg') or (convert_to_png and file_ext.lower() == '.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{file_root}.png")

            if os.path.exists(input_path):
                try:
                    with Image.open(input_path) as img:
                        if convert_to_png or not img.info.get('icc_profile'):  # Only process if no ICC profile for PNG
                            img.save(output_path, 'PNG', icc_profile=None)
                            print(f"Converted {filename} to PNG without ICC profile.")
                        else:
                            print(f"Skipped {filename}, PNG has ICC profile.")
                except Exception as e:
                    print(f"Error converting {filename}: {e}")
            else:
                print(f"Warning: File {input_path} does not exist!")

# Function to parse the label map
def parse_labelmap(labelmap_path):
    label_map = {}
    class_map = {}
    current_class_value = 1

    with open(labelmap_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.strip().split(':')
            if len(parts) < 2:
                continue
            label, color_rgb = parts[:2]
            color_rgb = tuple(map(int, color_rgb.split(',')))
            label_map[color_rgb] = label
            if color_rgb == (0, 0, 0):
                class_map[color_rgb] = 0  # 0 reserved for (0, 0, 0) background
            else:
                class_map[color_rgb] = current_class_value
                current_class_value += 1
    
    return label_map, class_map

# Function to encode classes to single-channel image
def encode_classes_to_single_channel(image_path, output_path, label_map, class_map, bit_depth=8):
    image_file = tf.io.read_file(image_path)
    rgb_image = tf.image.decode_png(image_file, channels=3)
    rgb_image_np = rgb_image.numpy()
    height, width, _ = rgb_image_np.shape
    single_channel_image = np.zeros((height, width), dtype=np.uint32)
    
    for y in range(height):
        for x in range(width):
            rgb = tuple(rgb_image_np[y, x])
            single_channel_image[y, x] = class_map[rgb]

    single_channel_image_expanded = np.expand_dims(single_channel_image, axis=-1)
    dtype = tf.uint8 if bit_depth == 8 else tf.uint16
    single_channel_image_tensor = tf.convert_to_tensor(single_channel_image_expanded, dtype=dtype)
    encoded_image = tf.image.encode_png(single_channel_image_tensor)
    tf.io.write_file(output_path, encoded_image)

# Function to process images and save class map
def process_images(folder_path, output_folder, labelmap_path, bit_depth=8):
    label_map, class_map = parse_labelmap(labelmap_path)
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            encode_classes_to_single_channel(image_path, output_path, label_map, class_map, bit_depth)
    
    class_map_size = len(class_map)
    filename_label_map_colors = f'dataset/label_map_colors_{bit_depth}.txt'
    with open(filename_label_map_colors, 'w') as file:
        file.write("#class_number:label:color_rgb:grayscale_intensity_normalized\n")
        for color_rgb, class_number in class_map.items():
            if class_map_size == 0:
                grayscale_intensity_normalized = 0
            else:
                if bit_depth == 8:
                    grayscale_intensity_normalized = np.uint8((class_number * BIT_DEPTH_8_MAX_VALUE) / (class_map_size - 1))
                elif bit_depth == 16:
                    grayscale_intensity_normalized = np.uint16((class_number * BIT_DEPTH_16_MAX_VALUE) / (class_map_size - 1))
            file.write(f"{class_number}:{label_map[color_rgb]}:{color_rgb}:{grayscale_intensity_normalized}\n")

# Define folder paths and labelmap file
jpg_input_folder = 'dataset/jpg_images'
png_masks_folder = 'dataset/png_masks'
output_folder = 'dataset/png_images'
png_masks_output_folder = f'dataset/png_masks_{BIT_DEPTH_NUMBER}bit'
labelmap_file = 'dataset/labelmap.txt'

# Convert JPG images to PNG
#convert_images(jpg_input_folder, output_folder)

# Convert PNG masks to 8-bit without ICC profile
process_images(png_masks_folder, png_masks_output_folder, labelmap_file, BIT_DEPTH_NUMBER)

print("Processing completed successfully.")
