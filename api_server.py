from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder=str(Path(__file__).parent / "output" / "predictions_2026"))
CORS(app)

@app.route('/api/config', methods=['GET'])
def get_config():
    """Serve API configuration (Groq API key)"""
    return jsonify({
        'groq_api_key': os.getenv('GROQ_API_KEY', '')
    })

@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    print("🚀 Starting AFCAT API Server...")
    print("📍 API Config: http://localhost:5000/api/config")
    print("📍 Dashboard: http://localhost:5000/")
    app.run(debug=True, port=5000)
