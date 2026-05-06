#!/usr/bin/env python3
"""Simple script to download the alphamatting dataset."""

import urllib.request
import zipfile
import os
import shutil

# Dataset URLs
URLS = {
    'input_training_lowres.zip': 'https://owncloud.tuwien.ac.at/index.php/s/kbv8ZxuqehNk9vy/download?path=/Datasets&files=input_training_lowres.zip',
    'trimap_training_lowres.zip': 'https://owncloud.tuwien.ac.at/index.php/s/kbv8ZxuqehNk9vy/download?path=/Datasets&files=trimap_training_lowres.zip',
    'gt_training_lowres.zip': 'https://owncloud.tuwien.ac.at/index.php/s/kbv8ZxuqehNk9vy/download?path=/Datasets&files=gt_training_lowres.zip'
}

def download_file(url, filename):
    """Download a file from URL."""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"  Done! ({os.path.getsize(filename)} bytes)")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def extract_zip(zip_file, extract_dir):
    """Extract ZIP file to directory."""
    print(f"Extracting {zip_file}...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(extract_dir)
        print(f"  Extracted to {extract_dir}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    # Create data directories
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/trimaps', exist_ok=True)
    os.makedirs('data/gt_alpha', exist_ok=True)

    # Download and extract each dataset
    for filename, url in URLS.items():
        # Download
        if not download_file(url, filename):
            continue

        # Extract to temp directory
        temp_dir = 'temp_extract'
        if not extract_zip(filename, temp_dir):
            continue

        # Move files to proper locations
        if 'input_training' in filename:
            src_dir = os.path.join(temp_dir, 'input_training_lowres')
            dst_dir = 'data/images'
        elif 'trimap_training' in filename:
            src_dir = os.path.join(temp_dir, 'trimap_training_lowres')
            dst_dir = 'data/trimaps'
        elif 'gt_training' in filename:
            src_dir = os.path.join(temp_dir, 'gt_training_lowres')
            dst_dir = 'data/gt_alpha'

        # Copy files
        if os.path.exists(src_dir):
            for file in os.listdir(src_dir):
                if not file.startswith('.'):
                    shutil.copy2(os.path.join(src_dir, file), os.path.join(dst_dir, file))
            print(f"  Copied files to {dst_dir}")

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.remove(filename)

    print("\nDone! Dataset downloaded and organized.")

if __name__ == '__main__':
    main()
