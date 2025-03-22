from PIL import Image, ImageOps
import os

# Paths
input_path = "D:/work/animals/raw-img"
output_path = "D:/work/animalsProcessed"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Selected classes
selected_classes = ["chicken", "horse", "dog", "squirrel"]

# Target resolution
target_size = (300, 300)

def resize_and_pad(img, target_size):
    """Resize with minimal padding, ensuring small dimensions are not too tiny."""
    img = ImageOps.exif_transpose(img)  # Fix orientation

    # Resize the smaller side to at least 250px before padding
    width, height = img.size
    if min(width, height) < 250:
        if width < height:  
            new_width = 250  
            new_height = 300
        else:  
            new_height = 250  
            new_width = 300
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Now do normal resize + padding
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, (255, 255, 255))
    paste_x = (target_size[0] - img.size[0]) // 2
    paste_y = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))

    return new_img

# Process images
for class_name in selected_classes:
    class_input_path = os.path.join(input_path, class_name)
    class_output_path = os.path.join(output_path, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    if not os.path.exists(class_input_path):
        print(f"Skipping {class_name}: Folder not found")
        continue

    image_files = [f for f in os.listdir(class_input_path) if f.lower().endswith(('jpg', 'jpeg'))]

    for image_file in image_files:
        img_path = os.path.join(class_input_path, image_file)
        output_img_path = os.path.join(class_output_path, image_file)
        
        try:
            with Image.open(img_path) as img:
                processed_img = resize_and_pad(img, target_size)
                processed_img.save(output_img_path, "JPEG", quality=95)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("Processing completed. Images saved in D:/work/animalsProcessed")
