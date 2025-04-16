from flask import Flask, request, jsonify
from flask_cors import CORS
from ocr_service import extract_text
import os

app = Flask(__name__)
CORS(app)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        # Read the image file
        image_bytes = image.read()
        
        # Process the image using our OCR service
        result = extract_text(image_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 