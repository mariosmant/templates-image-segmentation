import os
import xml.etree.ElementTree as ET
from PIL import Image

def add_image_tags(xml_file, images_folder):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Counter settings
    start_number = 209
    padding = 10

    # Get and sort filenames in the images folder
    filenames = sorted(os.listdir(images_folder))

    for filename in filenames:
        file_number = int(os.path.splitext(filename)[0].split("_")[-1])
        if file_number >= start_number:
            # Generate the image ID and filename
            image_id = file_number - 1
            new_filename = f"{str(file_number).zfill(padding)}{os.path.splitext(filename)[1]}"

            # Get the image resolution
            image_path = os.path.join(images_folder, filename)
            with Image.open(image_path) as img:
                width, height = img.size

            # Create the new image tag
            image_element = ET.Element("image", {
                "id": str(image_id),
                "name": f"images/{new_filename}",
                "subset": "default",
                "task_id": "8",
                "width": str(width),
                "height": str(height)
            })
            
            # Append the new image tag to the root element
            root.append(image_element)

    # Pretty-print the XML with new lines and 2 spaces
    xml_str = ET.tostring(root, encoding='utf-8').decode('utf-8')
    lines = xml_str.split('><')
    pretty_xml_str = '>\n  <'.join(lines)
    
    # Add new line before closing tag of annotations
    pretty_xml_str = pretty_xml_str.replace('</annotations>', '\n</annotations>')

    # Save the updated XML back to the file
    with open(xml_file, 'w', encoding='UTF-8') as f:
        f.write(pretty_xml_str)
    print("XML file has been updated with new image tags.")

def main():
    xml_file = 'import/annotations.xml'
    images_folder = 'import/images/'
    add_image_tags(xml_file, images_folder)

if __name__ == "__main__":
    main()
