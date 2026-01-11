"""
Download Kaggle dataset for speech enhancement training
"""
import kagglehub
import os

# Download latest version
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("anupamupadhaya/voicebank-cleantest-esc-crybaby-dog")

print("Path to dataset files:", path)

# List the contents to see what we have
print("\nDataset structure:")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:10]:  # Show first 10 files per directory
        print(f'{subindent}{file}')
    if len(files) > 10:
        print(f'{subindent}... and {len(files) - 10} more files')
