from flask import Flask, request, jsonify, send_from_directory
import tempfile
import os
import json
import shutil
from flask_cors import CORS
from PIL import Image
import numpy as np              # pkg to read filepaths from the dataset folder       
import tensorflow.keras as K 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer,Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Add, Concatenate
from tensorflow.keras.applications.vgg19 import VGG19
from function.processing_super_resolution import *
from function.N2N.config import noise2noise
app = Flask(__name__)
CORS(app)

# Serve static files from the uploads directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('../public/uploads', filename)

@app.route('/denoised/<filename>')
def denoised_file(filename):
    return send_from_directory('../public/denoised', filename)


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
        model=load_model_super_resolution()
        high_res_image_saved_path = super_resolve_single_image(model,image_path, high_res_image_path)
        print(f"Super-resolution completed. High-resolution image saved at: {high_res_image_saved_path}")
        data={
            'low_resolution_image_path': image_path.replace("../public/", ""),
            'high_resolution_image_path': high_res_image_saved_path.replace("../public/", "")
        }
        return data


    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500
    
    
@app.route('/remove-noise', methods=['POST'])
def remove_noise():
    try:
        print("Received POST request for remove Noise.")
        if 'file' not in request.files:
            print("No image found in request.")
            return jsonify({'error': 'No image found in request'}), 400

        image_file = request.files['file']
        # Get the noise type from the request form data
        noise_type = request.form.get('noise_type', 'gaussian')  # Default to 'gaussian' if not provided
        if noise_type == 'Random Text':
            noise_type = 'text'
        if noise_type == 'Multiplicative Bernoulli':
            noise_type = 'bernoulli'
        # Split the base filename into name and extension
        name, ext = os.path.splitext(image_file.filename)

        # Generate a temporary path to save the uploaded image
        temp_dir = "../public/uploads"
        image_path = os.path.join(temp_dir, image_file.filename)
        image_file.save(image_path)

        # Perform super-resolution
        denoised_image_saved_path = noise2noise(noise_type, temp_dir)
        print(f"after test: {denoised_image_saved_path}")
        # Define the new path for denoise image in the public folder
        new_filename = f"{name}-{noise_type}-denoised.png"
        clean_image_path = os.path.join(denoised_image_saved_path, new_filename)
        clean_image_path = clean_image_path.replace('/public','')
        print(f"Noise2Noise completed. Denoised image saved at: {clean_image_path}")
        data={
            'noise_removed_image_path': clean_image_path
        }
        return data

    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500

@app.route('/train-model', methods=['POST'])
def train_model():
    try:
        print("Received POST request for model training.")

        # Get the JSON data sent with the request
        train_data = request.json
        if not train_data:
            print("No training data received.")
            return jsonify({'error': 'No training data provided'}), 400

        # Extract the model type from the data
        model_type = train_data.get('model')
        if not model_type:
            return jsonify({'error': 'Model type not specified'}), 400

        # Define the directory based on the model type
        if model_type == "Noise2Noise":
            train_dir = "../public/trainrequests/noise2noise"
        elif model_type == "Super resolution":
            train_dir = "../public/trainrequests/super_resolution"
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        os.makedirs(train_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(train_dir, "train_request.json")

        # Load existing data if the file exists- ??
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = []

        # Append the new data to the existing data
        existing_data.append(train_data)
        # Save the updated data
        with open(file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
        
        # Generate the batch file for N2N
        if model_type == "Noise2Noise":
            batch_file_path = os.path.join(train_dir, "run_noise2noise.bat")
            with open(batch_file_path, 'w') as batch_file:
                batch_file.write("@echo off\n")
                batch_file.write(f" python ./train.py ^\n")
                batch_file.write(f"    --train-dir \"{train_data.get('dataset')}/train\" ^\n")
                batch_file.write(f"    --valid-dir \"{train_data.get('dataset')}/valid\" ^\n")
                batch_file.write(f"    --ckpt-save-path \"./ckpts\" ^\n")
                batch_file.write(f"    --ckpt-overwrite ^\n")
                batch_file.write(f"    --train-size {train_data.get('trainSize')} ^\n")
                batch_file.write(f"    --valid-size {train_data.get('validSize')} ^\n")
                batch_file.write(f"    --nb-epochs {train_data.get('nbEpochs')} ^\n")
                batch_file.write(f"    --loss {train_data.get('loss').lower()} ^\n")
                batch_file.write(f"    --noise-type {train_data.get('noiseType').lower()} ^\n")
                batch_file.write(f"    --noise-param {train_data.get('noiseParam')} ^\n")
                batch_file.write(f"    --crop-size {train_data.get('cropSize')}\n")
            print(f"Batch file created successfully at: {batch_file_path}")

        print(f"Training data saved successfully at: {file_path}")
        return jsonify({'message': 'Training request received and saved successfully.'})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred processing your request'}),500

        
if __name__ == '__main__':
    app.run(debug=True, port=5000)
