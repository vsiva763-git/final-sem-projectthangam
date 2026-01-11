import torch
from torch.utils.data import DataLoader
from speech_enhancement.dataset import SpeechEnhancementDataset, FramePaddingCollate

clean_dir = '/home/codespace/.cache/kagglehub/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog/versions/1/clean_testset_wav/clean_testset_wav'
noisy_dir = '/home/codespace/.cache/kagglehub/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog/versions/1/noisy_dataset_wav/noisy_dataset_wav'

dataset = SpeechEnhancementDataset(noisy_dir=noisy_dir, clean_dir=clean_dir)
small, _ = torch.utils.data.random_split(dataset, [10, len(dataset)-10])
loader = DataLoader(small, batch_size=4, collate_fn=FramePaddingCollate(), num_workers=0)

print('Testing tuple unpacking...')
for i, batch in enumerate(loader):
    print(f'Batch {i}: type={type(batch)}, len={len(batch) if isinstance(batch, tuple) else "N/A"}')
    try:
        noisy_real, noisy_imag, clean_real, clean_imag = batch
        print(f'  ✓ Unpacking successful: {noisy_real.shape}')
    except Exception as e:
        print(f'  ✗ Error: {e}')
    if i >= 2:
        break
print('Done')
