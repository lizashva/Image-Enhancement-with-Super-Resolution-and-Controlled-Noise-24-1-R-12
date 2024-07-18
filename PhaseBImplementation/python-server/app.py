from flask import Flask, request, jsonify, send_from_directory
import tempfile
import os
import cv2
from flask_cors import CORS
from PIL import Image
import numpy as np              # pkg to read filepaths from the dataset folder       
import tensorflow.keras as K 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer,Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Add, Concatenate
from tensorflow.keras.applications.vgg19 import VGG19
from function.processing_super_resolution import super_resolve_image
app = Flask(__name__)
CORS(app)

# Serve static files from the uploads directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('../public/uploads', filename)




@app.route('/restore-image', methods=['POST'])
def restore_image():
    try:
        print("Received POST request for image restoration.")
        if 'file' not in request.files:
            print("No image found in request.")
            return jsonify({'error': 'No image found in request'}), 400

        image_file = request.files['file']
        
        # Generate a temporary path to save the uploaded image
        temp_dir = "../public/uploads"
        image_path = os.path.join(temp_dir, image_file.filename)
        image_file.save(image_path)
        base_name, extension = os.path.splitext(image_file.filename)
        # Assuming the format is like '2_128.jpg', replace '128' with '256' and add '_created'
        new_filename = 'high_res_'+image_file.filename 
        print(new_filename)
        high_res_image_path = os.path.join(temp_dir, new_filename)
        # Define the new path for high-resolution images in the public folder
        print("low path:",image_path)
        print("high path:",high_res_image_path)
        # Perform super-resolution
        high_res_image_saved_path = super_resolve_image(image_path, high_res_image_path)
        print(f"Super-resolution completed. High-resolution image saved at: {high_res_image_saved_path}")
        data={
            'low_resolution_image_path': image_path.replace("../public/", ""),
            'high_resolution_image_path': high_res_image_saved_path.replace("../public/", "")
        }
        return data

    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
