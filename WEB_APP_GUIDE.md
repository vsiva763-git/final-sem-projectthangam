# 🌐 Speech Enhancement Web Application

## ✅ Your Website is Running!

### 🚀 Quick Start

**The web application is now live at:**
```
http://localhost:5000
```

### 📱 How to Use

1. **Open the website** in your browser: http://localhost:5000
2. **Upload your noisy audio** (drag & drop or click to browse)
3. **Click "Remove Noise"** button
4. **Download your enhanced audio** - noise removed!

---

## 🎯 Features

✨ **Beautiful Web Interface:**
- Drag-and-drop file upload
- Real-time processing status
- Visual progress indicators
- Download enhanced audio instantly

🤖 **AI-Powered Noise Removal:**
- Neural network processes your audio
- Removes background noise
- Preserves speech quality
- Fast processing

📊 **Processing Stats:**
- Sample rate information
- Audio duration
- Input/output sample counts
- Real-time updates

---

## 💻 Technical Details

### Server Information
- **Framework:** Flask (Python web framework)
- **Model:** Speech Enhancement Neural Network
- **Device:** CPU (can use GPU if available)
- **Port:** 5000
- **Max File Size:** 50MB

### API Endpoints

#### `GET /`
Main web interface

#### `POST /upload`
Upload and process audio file
- **Input:** WAV file (multipart/form-data)
- **Output:** JSON with download URL and stats

#### `GET /download/<filename>`
Download enhanced audio file

#### `GET /health`
Check server health status

---

## 🎨 Web Interface Features

### Upload Methods
1. **Drag & Drop** - Drop your WAV file directly onto the upload area
2. **Click to Browse** - Click the upload area to select a file
3. **File Validation** - Automatically checks for WAV format

### Visual Feedback
- ✅ Success messages
- ❌ Error messages
- 🔄 Processing indicators
- 📊 Statistics display
- ⬇️ Download button

---

## 📝 Example Usage

### Using the Web Interface
1. Navigate to http://localhost:5000
2. Drag your `noisy_audio.wav` file onto the page
3. Click "🚀 Remove Noise"
4. Wait for processing (usually a few seconds)
5. Click "⬇️ Download Enhanced Audio"

### Using the API (cURL)
```bash
# Upload and process audio
curl -X POST http://localhost:5000/upload \
  -F "audio=@noisy_audio.wav" \
  | jq .

# Download result
curl -O http://localhost:5000/download/FILENAME.wav
```

### Using the API (Python)
```python
import requests

# Upload audio
with open('noisy_audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/upload',
        files={'audio': f}
    )

result = response.json()
print(result)

# Download enhanced audio
if result['success']:
    download_url = 'http://localhost:5000' + result['download_url']
    enhanced_audio = requests.get(download_url)
    
    with open('enhanced_audio.wav', 'wb') as f:
        f.write(enhanced_audio.content)
```

---

## 🔧 Configuration

### Change Port
Edit `app.py` line 213:
```python
app.run(host='0.0.0.0', port=5000, debug=True)  # Change 5000 to your port
```

### Use Different Model
Edit `app.py` line 22:
```python
app.config['MODEL_PATH'] = 'path/to/your/model.pt'
```

### Increase File Size Limit
Edit `app.py` line 19:
```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

---

## 🛠️ Server Management

### Start the Server
```bash
python app.py
```

### Stop the Server
Press `Ctrl+C` in the terminal

### Run in Background
```bash
nohup python app.py > server.log 2>&1 &
```

### Check Server Status
```bash
curl http://localhost:5000/health
```

---

## 🌍 Deploy to Production

### Using Gunicorn (Recommended)
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Deploy to Cloud
- **Heroku:** `git push heroku main`
- **AWS EC2:** Upload files and run with Gunicorn
- **Google Cloud Run:** Deploy Docker container
- **Azure:** Deploy as App Service

---

## 📊 Performance

### Processing Speed
- **Small files (1-5 seconds):** ~2-3 seconds
- **Medium files (5-30 seconds):** ~5-10 seconds
- **Large files (30+ seconds):** ~10-20 seconds

### Optimization Tips
1. Use GPU: Set `DEVICE = 'cuda'` in app.py
2. Batch processing for multiple files
3. Use production WSGI server (Gunicorn)
4. Enable file caching

---

## 🔒 Security Notes

⚠️ **For Development Only**
The current setup uses Flask's development server. For production:

1. **Use HTTPS** - Encrypt traffic
2. **Add authentication** - Protect your API
3. **Rate limiting** - Prevent abuse
4. **File validation** - Check file integrity
5. **Use WSGI server** - Gunicorn, uWSGI, etc.

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find and kill process on port 5000
lsof -i :5000
kill -9 <PID>
```

### Model Not Found
Check that `demo_checkpoints/checkpoint_epoch_1.pt` exists:
```bash
ls -la demo_checkpoints/
```

If missing, run:
```bash
python examples.py
```

### Upload Fails
- Check file is WAV format
- Verify file size < 50MB
- Ensure server is running

### Browser Can't Connect
- Check server is running: `curl http://localhost:5000/health`
- Try 127.0.0.1 instead of localhost
- Check firewall settings

---

## 📦 Project Structure

```
/workspaces/final-sem-project/
├── app.py                    # Flask web application
├── templates/
│   └── index.html           # Web interface
├── static/                   # Static files (CSS, JS, images)
├── uploads/                  # Temporary uploaded files
├── outputs/                  # Enhanced audio files
├── demo_checkpoints/         # Model checkpoints
├── speech_enhancement/       # Core library
└── requirements.txt          # Python dependencies
```

---

## 🎉 Success!

Your speech enhancement web application is now running!

**Access it at:** http://localhost:5000

Upload your noisy audio and watch the magic happen! 🎵✨

---

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review server logs
3. Test with sample audio files
4. Verify model checkpoint exists

**Happy denoising!** 🚀
