import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

BIT_DEPTH_NUMBER = 8
BIT_DEPTH_8_MAX_VALUE = 255
BIT_DEPTH_16_MAX_VALUE = 65535

unique_colors = {}

def parse_labelmap(labelmap_path):
    label_map = {}
    class_map = {}

    # Generate a unique integer value for each unique RGB color
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
                class_map[color_rgb] = 0 # 0 reserved for (0, 0, 0) background
            else:
                class_map[color_rgb] = current_class_value
                current_class_value += 1
    
    return label_map, class_map

def encode_classes_to_single_channel(image_path, output_path, label_map, class_map, bit_depth=8):
    # Read the image file
    image_file = tf.io.read_file(image_path)
    
    # Decode the RGB image
    rgb_image = tf.image.decode_png(image_file, channels=3)
    
    # Convert the image to a numpy array
    rgb_image_np = rgb_image.numpy()
    
    # Get the image dimensions
    height, width, _ = rgb_image_np.shape
    
    # Create an empty single-channel image to store class values
    single_channel_image = np.zeros((height, width), dtype=np.uint32)
    
    for y in range(height):
        for x in range(width):
            rgb = tuple(rgb_image_np[y, x])
            single_channel_image[y, x] = class_map[rgb]

    # Convert the single-channel image to the desired bit depth

    # Expand dimensions to match the required shape (height, width, 1)
    single_channel_image_expanded = np.expand_dims(single_channel_image, axis=-1)
    
    # Convert the single-channel image back to a TensorFlow tensor
    dtype = tf.uint8 if bit_depth == 8 else tf.uint16
    single_channel_image_tensor = tf.convert_to_tensor(single_channel_image_expanded, dtype=dtype)
    
    # Encode the single-channel image as PNG
    encoded_image = tf.image.encode_png(single_channel_image_tensor)
    
    # Save the encoded image to the specified path
    tf.io.write_file(output_path, encoded_image)


def process_images(folder_path, output_folder, labelmap_path, bit_depth=8):
    # Read labelmap
    label_map, class_map = parse_labelmap(labelmap_path)
    print("label_map=", label_map)
    print("len(label_map)=", len(label_map))
    print("class_map=", class_map)
    print("len(class_map)=", len(class_map))

    # Create output directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process images
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Process image
            encode_classes_to_single_channel(image_path, output_path, label_map, class_map, bit_depth)
    
    # Save class_map to label_map_colors.txt
    class_map_size = len(class_map)
    filename_label_map_colors = f'label_map_colors_{bit_depth}.txt'
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

# Example usage
process_images('png_masks', f'png_masks_{BIT_DEPTH_NUMBER}bit', 'labelmap.txt', BIT_DEPTH_NUMBER)
