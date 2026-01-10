from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <html>
    <body>
        <h1>Flask is working!</h1>
        <p>Server is running successfully on port 5000</p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Starting test Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
