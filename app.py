"""
Flask Web Application for Speech Enhancement
Upload noisy audio and get cleaned audio back!
"""

import os
import torch
import soundfile as sf
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
import shutil

from speech_enhancement.model import SpeechEnhancementNetwork
from speech_enhancement.data_processing import STFTProcessor


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MODEL_PATH'] = 'demo_checkpoints/checkpoint_epoch_1.pt'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global model (loaded once at startup)
MODEL = None
STFT_PROCESSOR = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    """Load the speech enhancement model"""
    global MODEL, STFT_PROCESSOR
    
    print("🤖 Loading speech enhancement model...")
    MODEL = SpeechEnhancementNetwork(base_channels=16)
    
    checkpoint = torch.load(app.config['MODEL_PATH'], map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        MODEL.load_state_dict(checkpoint['model_state_dict'])
    else:
        MODEL.load_state_dict(checkpoint)
    
    MODEL.to(DEVICE)
    MODEL.eval()
    
    STFT_PROCESSOR = STFTProcessor(
        n_fft=512,
        hop_length=256,
        window="hann",
    )
    
    print(f"✅ Model loaded successfully on {DEVICE}")


def denoise_audio_file(input_path, output_path):
    """
    Process audio file to remove noise
    
    Args:
        input_path: Path to input noisy audio
        output_path: Path to save enhanced audio
        
    Returns:
        dict with processing info
    """
    try:
        print(f"[DENOISE] Starting: {input_path}")
        
        # Load audio
        print(f"[DENOISE] Loading audio...")
        noisy_audio, sample_rate = sf.read(input_path)
        print(f"[DENOISE] Audio loaded: shape={noisy_audio.shape}, sr={sample_rate}")
        
        # Convert stereo to mono if needed
        if len(noisy_audio.shape) > 1:
            print(f"[DENOISE] Converting stereo to mono...")
            noisy_audio = np.mean(noisy_audio, axis=1)
        
        # Ensure proper dtype
        noisy_audio = noisy_audio.astype(np.float32)
        print(f"[DENOISE] Audio shape after conversion: {noisy_audio.shape}")
        
        # Compute STFT
        print(f"[DENOISE] Computing STFT...")
        noisy_audio_tensor = torch.from_numpy(noisy_audio)
        real_part, imag_part = STFT_PROCESSOR.stft(noisy_audio_tensor)
        print(f"[DENOISE] STFT complete: real_part shape={real_part.shape}, imag_part shape={imag_part.shape}")
        
        # Check if STFT output is valid
        if real_part.shape[0] == 0 or real_part.shape[1] == 0:
            raise ValueError(f"Invalid STFT output shape: {real_part.shape}")
        
        # Pad or slice to match expected dimensions (257 freq bins)
        # The model expects 257 frequency bins (n_fft=512 gives 512/2 + 1 = 257)
        expected_freq_bins = 257
        actual_freq_bins = real_part.shape[0]
        
        if actual_freq_bins != expected_freq_bins:
            print(f"[DENOISE] Warning: Expected {expected_freq_bins} freq bins, got {actual_freq_bins}")
            if actual_freq_bins < expected_freq_bins:
                # Pad with zeros
                pad_size = expected_freq_bins - actual_freq_bins
                real_part = torch.nn.functional.pad(real_part, (0, 0, 0, pad_size))
                imag_part = torch.nn.functional.pad(imag_part, (0, 0, 0, pad_size))
            else:
                # Slice
                real_part = real_part[:expected_freq_bins, :]
                imag_part = imag_part[:expected_freq_bins, :]
        
        # Process in chunks if the audio is too long
        max_time_frames = 500  # Process max 500 time frames at once
        num_chunks = int(np.ceil(real_part.shape[1] / max_time_frames))
        
        print(f"[DENOISE] Total time frames: {real_part.shape[1]}, chunks: {num_chunks}")
        
        enhanced_real_list = []
        enhanced_imag_list = []
        
        for chunk_idx in range(num_chunks):
            print(f"[DENOISE] Processing chunk {chunk_idx + 1}/{num_chunks}...")
            
            # Get chunk
            start_frame = chunk_idx * max_time_frames
            end_frame = min((chunk_idx + 1) * max_time_frames, real_part.shape[1])
            
            chunk_real = real_part[:, start_frame:end_frame]
            chunk_imag = imag_part[:, start_frame:end_frame]
            
            # Add batch and channel dimensions
            chunk_real = chunk_real.unsqueeze(0).unsqueeze(0).to(DEVICE)
            chunk_imag = chunk_imag.unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            print(f"[DENOISE] Chunk shape: real={chunk_real.shape}, imag={chunk_imag.shape}")
            
            # Enhance audio
            print(f"[DENOISE] Running model inference for chunk {chunk_idx + 1}/{num_chunks}...")
            with torch.no_grad():
                chunk_enhanced_real, chunk_enhanced_imag = MODEL(chunk_real, chunk_imag)
            
            # Remove batch and channel dimensions
            chunk_enhanced_real = chunk_enhanced_real.squeeze(0).squeeze(0).cpu()
            chunk_enhanced_imag = chunk_enhanced_imag.squeeze(0).squeeze(0).cpu()
            
            enhanced_real_list.append(chunk_enhanced_real)
            enhanced_imag_list.append(chunk_enhanced_imag)
        
        # Concatenate all chunks
        enhanced_real = torch.cat(enhanced_real_list, dim=1)
        enhanced_imag = torch.cat(enhanced_imag_list, dim=1)
        
        print(f"[DENOISE] After concatenation: real={enhanced_real.shape}, imag={enhanced_imag.shape}")
        
        # Inverse STFT
        print(f"[DENOISE] Computing inverse STFT...")
        enhanced_audio = STFT_PROCESSOR.istft(enhanced_real, enhanced_imag)
        print(f"[DENOISE] Enhanced audio shape: {enhanced_audio.shape}")
        
        # Save enhanced audio
        enhanced_audio_np = enhanced_audio.numpy()
        print(f"[DENOISE] Saving to: {output_path}")
        sf.write(output_path, enhanced_audio_np, sample_rate)
        
        # Calculate stats
        duration = len(noisy_audio) / sample_rate
        
        print(f"[DENOISE] Success! Duration: {duration:.2f}s")
        return {
            'success': True,
            'sample_rate': int(sample_rate),
            'duration': float(duration),
            'input_samples': int(len(noisy_audio)),
            'output_samples': int(len(enhanced_audio_np)),
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[DENOISE] ERROR: {error_msg}")
        traceback.print_exc()
        return {
            'success': False,
            'error': error_msg
        }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'success': True,
            'status': 'healthy',
            'model': 'loaded',
            'device': str(DEVICE)
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.wav'):
            return jsonify({'success': False, 'error': 'Only WAV files are supported'}), 400
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_filename = f"{unique_id}_input.wav"
        output_filename = f"{unique_id}_enhanced.wav"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Save uploaded file
        file.save(input_path)
        
        # Process audio
        result = denoise_audio_file(input_path, output_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Audio enhanced successfully!',
                'download_url': url_for('download_file', filename=output_filename),
                'stats': {
                    'sample_rate': int(result['sample_rate']),
                    'duration': f"{result['duration']:.2f} seconds",
                    'input_samples': int(result['input_samples']),
                    'output_samples': int(result['output_samples']),
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download enhanced audio file"""
    try:
        # Validate filename (security check)
        if '..' in filename or '/' in filename:
            return jsonify({'success': False, 'error': 'Invalid filename'}), 400
            
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"enhanced_{filename}",
            mimetype='audio/wav'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up old files"""
    try:
        # Remove files older than 1 hour
        import time
        current_time = time.time()
        
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    if current_time - os.path.getmtime(file_path) > 3600:  # 1 hour
                        os.remove(file_path)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE)
    })


if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    print("\n" + "=" * 60)
    print("🎵 Speech Enhancement Web Application")
    print("=" * 60)
    print(f"🌐 Open your browser and go to: http://localhost:5000")
    print(f"🤖 Model: {app.config['MODEL_PATH']}")
    print(f"💻 Device: {DEVICE}")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
