import os
from PIL import Image

# Paths to the folders
jpg_input_folder = 'jpg_images'
png_masks_folder = 'png_masks'
output_folder = 'png_images'  # Folder where PNG images will be saved

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to convert images
def convert_images(input_folder, output_folder, convert_to_png=True):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg')) or (convert_to_png and filename.lower().endswith('.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.jpg', '.png').replace('.jpeg', '.png'))

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

# Convert JPG images to PNG
convert_images(jpg_input_folder, output_folder)

# Process PNG images in png_masks_folder only if they don't have an ICC profile
convert_images(png_masks_folder, output_folder, convert_to_png=False)
