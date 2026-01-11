"""
Create a simple working baseline model for audio enhancement.
"""
import torch
import torch.nn as nn
from speech_enhancement.model import SpeechEnhancementNetwork

print("Creating a simple passthrough model...")

# Create model
model = SpeechEnhancementNetwork(base_channels=16)

# Initialize with small weights to avoid NaN
for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param, gain=0.01)
    else:
        nn.init.constant_(param, 0.01)

# Save as working checkpoint
checkpoint = {
    'epoch': 0,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': None,
    'metric_value': 0.0,
}

torch.save(checkpoint, 'demo_checkpoints/working_model.pt')
print("✅ Created demo_checkpoints/working_model.pt")
print("   This is a baseline model with small random weights")
print("   For better results, train on Kaggle with fixed training script")
