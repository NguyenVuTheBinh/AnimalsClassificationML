import os

# Dataset path
dataset_path = "D:/work/animals/raw-img"

# Class name translations (Italian → English)
class_translation = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
    "scoiattolo": "squirrel", "ragno": "spider"
}

# Rename folders
for italian_name, english_name in class_translation.items():
    old_path = os.path.join(dataset_path, italian_name)
    new_path = os.path.join(dataset_path, english_name)

    if os.path.exists(old_path) and not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print(f"Renamed '{italian_name}' → '{english_name}'")
    elif os.path.exists(new_path):
        print(f"Skipping '{italian_name}', '{english_name}' already exists")
    else:
        print(f"Folder '{italian_name}' not found, skipping.")

print("\nFolder renaming complete!")
