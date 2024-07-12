from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# import cv2
# import numpy as np
import os
from function.function import print11
print11()
# Uncomment these lines to properly initialize the Flask app and CORS
app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return "Welcome to the Flask Server!"
    
@app.route('/upload-image', methods=['GET'])
def handle_image():
    print("upload file success")
    # print11()
    # file = request.files['file']
    # base_path = os.path.dirname(__file__)
    # upload_path = os.path.join(base_path, 'uploads', file.filename)
    # file.save(upload_path)
    # # Add your image processing code here
    # processed_path = os.path.join(base_path, 'processed', 'high_res_' + file.filename)
    # # Dummy image processing: just copy the file for now
    # cv2.imwrite(processed_path, cv2.imread(upload_path))
    # return jsonify({'image_path': f'/processed/high_res_{file.filename}'})
    return jsonify({'message': 'Image upload placeholder'})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5001)
    print("Flask server running on port 5001")  # This will print after the server stops