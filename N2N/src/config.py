from flask import Flask, request, jsonify
from argparse import Namespace
import json
from test import *
from train import *


app = Flask(__name__)


class TrainConfig:
    def __init__(self, noise_type, noise_param):
        self.train_dir = './data/train'
        self.train_size = 500
        self.valid_dir = './data/valid'
        self.valid_size = 100
        self.ckpt_save_path = './ckpts'
        self.ckpt_overwrite = False
        self.learning_rate = 0.001
        self.adam = [0.9, 0.99, 1e-8]
        self.report_interval = 25
        self.batch_size = 4
        self.nb_epochs = 5
        self.loss = 'l2'
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.crop_size = 128
        self.seed = 42
        self.plot_stats = False
        self.cuda = True
        self.clean_targets = False


class TestConfig:
    def __init__(self, noise_type, noise_param):
        self.data = './data/test'
        self.noise_type = noise_type
        self.load_ckpt = f'./ckpts/{noise_type}/n2n-{noise_type}.pt'
        self.noise_param = noise_param
        self.seed = 1
        self.crop_size = 256
        self.show_output = 1


def set_configuration(config):
    valid_noise_types = ['gaussian', 'poisson', 'text']
    if config['noise_type'] not in valid_noise_types:
        raise ValueError(f"Unsupported noise type '{config['noise_type']}'. Choose from {valid_noise_types}.")

    if config['status'] == 'train':
        params = TrainConfig(config['noise_type'], config['noise_param'])
    else:
        params = TestConfig(config['noise_type'], config['noise_param'])

    return params


def json_to_namespace(json_data):
    config = Namespace()
    config.__dict__.update(json_data)
    return config

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        config = json_to_namespace(data)
        run_training(config)
        return jsonify({"status": "Training started"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_training(config):
    # Train/Test datasets - adding noise to the image

    params = set_configuration(config)
    test_N2N(params)

if __name__ == '__main__':
    # app.run(debug=True)
    params = TestConfig('poisson', 1.2)
    #params = TestConfig('gaussian', 25)
    test_N2N(params)