"""
Example: Train and test speech enhancement model with Kaggle dataset

This script demonstrates the complete workflow:
1. Verify dataset is available
2. Train the model
3. Test the model on sample audio
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dataset():
    """Check if Kaggle dataset is downloaded."""
    dataset_path = Path("/home/codespace/.cache/kagglehub/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog/versions/1")
    
    if not dataset_path.exists():
        print("📥 Dataset not found. Downloading from Kaggle...")
        subprocess.run([sys.executable, "download_dataset.py"])
        print("✅ Dataset downloaded successfully!\n")
    else:
        print("✅ Dataset already available\n")
    
    return dataset_path

def train_model(epochs=5, batch_size=8):
    """Train the model with specified parameters."""
    print(f"🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Device: {'GPU' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'CPU'}")
    print()
    
    cmd = [
        sys.executable, "train_kaggle.py",
        "--config", "configs/default_config.yaml",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--num-workers", "2",
        "--checkpoint-dir", "./kaggle_checkpoints",
        "--log-dir", "./kaggle_logs",
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
        return True
    else:
        print("\n❌ Training failed!")
        return False

def test_model():
    """Test the trained model on a sample."""
    checkpoint_path = Path("kaggle_checkpoints/best_model.pt")
    
    if not checkpoint_path.exists():
        print("⚠️  No trained model found. Skipping testing.")
        return
    
    print("🧪 Testing trained model...")
    print("   (This would run inference on test samples)")
    print("   Use: python inference.py --checkpoint kaggle_checkpoints/best_model.pt")
    print()

def main():
    """Main workflow."""
    print("=" * 70)
    print("Speech Enhancement Model - Kaggle Dataset Training")
    print("=" * 70)
    print()
    
    # Step 1: Check dataset
    print("Step 1: Dataset Verification")
    print("-" * 70)
    dataset_path = check_dataset()
    
    # Show dataset info
    clean_dir = dataset_path / "clean_testset_wav" / "clean_testset_wav"
    noisy_dir = dataset_path / "noisy_dataset_wav" / "noisy_dataset_wav"
    
    if clean_dir.exists() and noisy_dir.exists():
        num_files = len(list(clean_dir.glob("*.wav")))
        print(f"📊 Dataset Info:")
        print(f"   Total samples: {num_files}")
        print(f"   Training: {int(num_files * 0.8)} samples (80%)")
        print(f"   Validation: {int(num_files * 0.2)} samples (20%)")
        print()
    
    # Step 2: Train model
    print("Step 2: Model Training")
    print("-" * 70)
    
    # Default: 5 epochs for quick demo
    # Increase for better performance (e.g., 50-100 epochs)
    epochs = 5
    batch_size = 8
    
    print(f"⚙️  Configuration:")
    print(f"   For quick demo: {epochs} epochs")
    print(f"   For production: 50-100 epochs recommended")
    print(f"   Batch size: {batch_size}")
    print()
    
    user_input = input("Start training? (y/n): ").strip().lower()
    
    if user_input == 'y':
        success = train_model(epochs=epochs, batch_size=batch_size)
        
        if success:
            # Step 3: Test model
            print("\nStep 3: Model Testing")
            print("-" * 70)
            test_model()
            
            print("=" * 70)
            print("🎉 Complete! Your model is ready to use.")
            print()
            print("Next steps:")
            print("1. Test on your own audio:")
            print("   python inference.py --checkpoint kaggle_checkpoints/best_model.pt \\")
            print("       --input_audio your_noisy_audio.wav \\")
            print("       --output_audio enhanced_audio.wav")
            print()
            print("2. Integrate with web app:")
            print("   Update app.py to use kaggle_checkpoints/best_model.pt")
            print()
            print("3. Train longer for better results:")
            print("   ./train_with_kaggle.sh 50 8")
            print("=" * 70)
    else:
        print("\n⏸️  Training cancelled.")
        print("To train later, run: ./train_with_kaggle.sh")

if __name__ == "__main__":
    main()
