import os
import xml.etree.ElementTree as ET

def parse_and_rename(xml_file, images_folder):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Padding and counter settings
    padding = 10
    counter = 1

    for image in root.findall('image'):
        image_id = image.get('id')
        image_name = image.get('name')
        
        # Remove "images/" portion for filename generation
        original_filename = image_name.replace("images/", "")

        # Extract the file extension
        extension = os.path.splitext(original_filename)[1]

        # Generate the new filename with the specified pattern
        new_filename = f"{str(counter).zfill(padding)}{extension}"
        counter += 1

        # Update the 'name' attribute in XML
        image.set('name', f"images/{new_filename}")

        # Get the current and new file paths
        current_path = os.path.join(images_folder, original_filename)
        new_path = os.path.join(images_folder, new_filename)

        # Rename the physical file
        if os.path.exists(current_path):
            os.rename(current_path, new_path)
            print(f"Renamed {original_filename} to {new_filename}")
        else:
            print(f"Warning: File {current_path} does not exist!")

    # Save the updated XML back to the file
    tree.write(xml_file, encoding='UTF-8', xml_declaration=True)
    print("XML file has been updated with new filenames.")

def main():
    xml_file = 'import/annotations.xml'
    images_folder = 'import/images'
    parse_and_rename(xml_file, images_folder)

if __name__ == "__main__":
    main()
