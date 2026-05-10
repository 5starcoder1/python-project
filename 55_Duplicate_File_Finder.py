# =================================
# DUPLICATE FILE FINDER IN PYTHON
# =================================

import os
import hashlib

# Function to generate file hash
def generate_hash(file_path):
    hash_algo = hashlib.md5()

    try:
        with open(file_path, "rb") as file:
            while chunk := file.read(4096):
                hash_algo.update(chunk)

        return hash_algo.hexdigest()

    except:
        return None

# Function to find duplicate files
def find_duplicates(folder_path):
    hashes = {}
    duplicates = []

    for root, dirs, files in os.walk(folder_path):

        for file in files:
            file_path = os.path.join(root, file)

            file_hash = generate_hash(file_path)

            if file_hash:

                if file_hash in hashes:
                    duplicates.append(file_path)

                else:
                    hashes[file_hash] = file_path

    return duplicates

# Main Program
print("===== DUPLICATE FILE FINDER =====")

folder = input("Enter folder path: ")

if os.path.exists(folder):

    duplicate_files = find_duplicates(folder)

    print("\n===== DUPLICATE FILES =====")

    if len(duplicate_files) == 0:
        print("✅ No duplicate files found!")

    else:
        for file in duplicate_files:
            print(file)

else:
    print("❌ Folder does not exist!")
