#!/usr/bin/env python3
import os
import shutil

# 1) source root where your item‑folders live:
SRC_ROOT = os.path.expanduser("~/Desktop/cv2/waste_classification/images/images")

# 2) destination root:
DST_ROOT = os.path.expanduser("~/Desktop/cv2/waste_classification/images/merged_Categories")

# 3) define mapping from item‑folder → high‑level category
CATEGORY_MAP = {
    "Plastic": [
        "plastic_water_bottles", "plastic_soda_bottles", "plastic_detergent_bottles",
        "plastic_shopping_bags", "plastic_trash_bags", "plastic_food_containers",
        "disposable_plastic_cutlery", "plastic_straws", "plastic_cup_lids"
    ],
    "Paper_and_Cardboard": [
        "newspaper", "office_paper", "magazines", "cardboard_boxes", "cardboard_packaging"
    ],
    "Glass": [
        "glass_beverage_bottles", "glass_food_jars", "glass_cosmetic_containers"
    ],
    "Metal": [
        "aluminum_soda_cans", "aluminum_food_cans", "steel_food_cans", "aerosol_cans"
    ],
    "Organic_Waste": [
        "food_waste", "eggshells", "coffee_grounds", "tea_bags"
    ],
    "Textiles": [
        "clothing", "shoes"
    ],
}

def merge_into_categories(src_root, dst_root, mapping):
    # ensure destination root exists
    os.makedirs(dst_root, exist_ok=True)

    # create each high‑level category folder
    for cat in mapping:
        os.makedirs(os.path.join(dst_root, cat), exist_ok=True)

    # for each high‑level category, copy from each of its item‑folders
    for cat, item_folders in mapping.items():
        dst_cat_dir = os.path.join(dst_root, cat)
        for item in item_folders:
            src_item_dir = os.path.join(src_root, item)
            if not os.path.isdir(src_item_dir):
                print(f"⚠️  Warning: expected folder not found: {src_item_dir}")
                continue
            for fname in os.listdir(src_item_dir):
                src_file = os.path.join(src_item_dir, fname)
                if not os.path.isfile(src_file):
                    continue
                # resolve name collisions by appending a counter
                base, ext = os.path.splitext(fname)
                dst_file = os.path.join(dst_cat_dir, fname)
                counter = 1
                while os.path.exists(dst_file):
                    dst_file = os.path.join(dst_cat_dir, f"{base}_{counter}{ext}")
                    counter += 1
                shutil.copy2(src_file, dst_file)
        print(f"✅  Merged items for category '{cat}'")

    print(f"\n🎉 All done. Check your merged folders in:\n    {dst_root}")

if __name__ == "__main__":
    merge_into_categories(SRC_ROOT, DST_ROOT, CATEGORY_MAP)
