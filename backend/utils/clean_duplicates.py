import os
import hashlib

# Paths to your dataset
# Paths to your dataset
# Get project root (3 levels up from backend/utils/clean_duplicates.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")

def file_hash(path):
    """Return MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def clean_folder(folder_path):
    """Remove duplicate images inside a folder."""
    hashes = {}
    removed = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(root, file)
            h = file_hash(path)

            if h in hashes:
                print(f"[DELETE] Duplicate: {path}")
                os.remove(path)
                removed += 1
            else:
                hashes[h] = path

    return removed

if __name__ == "__main__":
    print("🔍 Cleaning Training...")
    removed_train = clean_folder(TRAIN_DIR)

    print("\n🔍 Cleaning Testing...")
    removed_test = clean_folder(TEST_DIR)

    total = removed_train + removed_test
    print(f"\n✅ Done! Removed {total} duplicate images.")
