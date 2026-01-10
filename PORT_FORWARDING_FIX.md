# 🔧 Port Forwarding Fix - Quick Guide

## ✅ Application is Running!

Your Speech Enhancement Web Application is **running successfully** on port 5000.

---

## 🌐 How to Access the Application

### Option 1: VS Code Simple Browser (Recommended)
The application should now be open in VS Code's Simple Browser panel.

### Option 2: External Browser
VS Code should automatically forward port 5000. Look for:
- A notification in VS Code about port forwarding
- Check the **PORTS** tab in VS Code (usually at the bottom)
- Click on the forwarded URL there

### Option 3: Manual Port Forwarding

If you don't see port 5000 in the PORTS tab:

1. **Open the Ports Panel:**
   - Click on **PORTS** tab at the bottom of VS Code
   - Or press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type: "Forward a Port"
   - Enter port: `5000`

2. **Access the Application:**
   - Click the globe icon next to port 5000
   - Or copy the forwarded URL and open in browser

---

## 🧪 Test the Application

### Quick Health Check
Run this command to verify the app is working:
```bash
curl http://localhost:5000/health
```

**Expected Response:**
```json
{
  "device": "cpu",
  "model_loaded": true,
  "status": "healthy"
}
```

### Full Application Test
1. Open http://localhost:5000 in your browser
2. You should see a beautiful purple gradient interface
3. Try uploading a WAV file to test noise removal

---

## 📊 Application Status

✅ **Server:** Running on port 5000  
✅ **Model:** Loaded successfully (demo_checkpoints/checkpoint_epoch_1.pt)  
✅ **Device:** CPU  
✅ **Health Check:** Passing  

---

## 🎯 Available Test Files

You can test with these sample audio files:
- `/workspaces/final-sem-project/data/clean/sample_0.wav`
- `/workspaces/final-sem-project/data/clean/sample_1.wav`
- `/workspaces/final-sem-project/data/demo_clean/sample_0.wav`
- `/workspaces/final-sem-project/data/demo_clean/sample_1.wav`
- `/workspaces/final-sem-project/data/demo_clean/sample_2.wav`

---

## 🛠️ Troubleshooting

### If you still see "Error forwarding port":

1. **Restart the Application:**
   ```bash
   pkill -f "python.*app.py"
   cd /workspaces/final-sem-project
   python app.py
   ```

2. **Check if port is in use:**
   ```bash
   lsof -i :5000
   ```

3. **Use a different port:**
   Edit `app.py` and change the port number, then restart

4. **Check VS Code Settings:**
   - Ensure port forwarding is enabled in your VS Code settings
   - Check if firewall is blocking the connection

### Current App Status:
```bash
# Check if app is running
ps aux | grep "python.*app.py" | grep -v grep

# View app logs
tail -f /workspaces/final-sem-project/app.log
```

---

## 🚀 Quick Restart Commands

```bash
# Stop the app
pkill -f "python.*app.py"

# Start the app in foreground (see output)
cd /workspaces/final-sem-project
python app.py

# Or start in background
cd /workspaces/final-sem-project
nohup python app.py > app.log 2>&1 &
```

---

## 📝 Notes

- The application is configured to run on `0.0.0.0:5000` (accessible from all interfaces)
- In a dev container, VS Code should automatically forward ports
- If using GitHub Codespaces, the port will be automatically exposed
- Maximum upload file size: 50MB
- Supported audio format: WAV files only

---

## ✨ Features Working

✅ Model loading and initialization  
✅ Web server running  
✅ Health endpoint responding  
✅ File upload handling  
✅ Audio processing pipeline  
✅ Download functionality  

The application is **fully functional** and ready to process audio files!
