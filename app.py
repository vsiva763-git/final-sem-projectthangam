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
        # Load audio
        noisy_audio, sample_rate = sf.read(input_path)
        
        # Convert stereo to mono if needed
        if len(noisy_audio.shape) > 1:
            noisy_audio = np.mean(noisy_audio, axis=1)
        
        # Compute STFT
        noisy_audio_tensor = torch.from_numpy(noisy_audio.astype(np.float32))
        real_part, imag_part = STFT_PROCESSOR.stft(noisy_audio_tensor)
        
        # Add batch and channel dimensions
        real_part = real_part.unsqueeze(0).unsqueeze(0).to(DEVICE)
        imag_part = imag_part.unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Enhance audio
        with torch.no_grad():
            enhanced_real, enhanced_imag = MODEL(real_part, imag_part)
        
        # Remove dimensions
        enhanced_real = enhanced_real.squeeze(0).squeeze(0).cpu()
        enhanced_imag = enhanced_imag.squeeze(0).squeeze(0).cpu()
        
        # Inverse STFT
        enhanced_audio = STFT_PROCESSOR.istft(enhanced_real, enhanced_imag)
        
        # Save enhanced audio
        enhanced_audio_np = enhanced_audio.numpy()
        sf.write(output_path, enhanced_audio_np, sample_rate)
        
        # Calculate stats
        duration = len(noisy_audio) / sample_rate
        
        return {
            'success': True,
            'sample_rate': sample_rate,
            'duration': duration,
            'input_samples': len(noisy_audio),
            'output_samples': len(enhanced_audio_np),
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.wav'):
        return jsonify({'error': 'Only WAV files are supported'}), 400
    
    try:
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
                    'sample_rate': result['sample_rate'],
                    'duration': f"{result['duration']:.2f} seconds",
                    'input_samples': result['input_samples'],
                    'output_samples': result['output_samples'],
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download enhanced audio file"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"enhanced_{filename}",
            mimetype='audio/wav'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404


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
    app.run(host='0.0.0.0', port=5000, debug=True)
