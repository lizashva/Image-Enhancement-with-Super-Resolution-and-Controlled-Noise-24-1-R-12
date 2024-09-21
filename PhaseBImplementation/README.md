# Image Enhancement with Super Resolution and Controlled Noise

## Project Overview

The **Pixel Pro GUI** is designed to offer a comprehensive interface for a sophisticated image processing model developed within this project. It allows users to apply super-resolution and noise removal techniques to images through an easy-to-use graphical interface.

### Features:

- **Training Section**: Enables users to configure the model’s parameters and select appropriate datasets for training.
- **Inference Section**: Allows users to apply pre-trained models to new images and test various enhancement methods.

---

## Requirements

- Python 3.8 or later
- Node.js (for the client-side interface)
- TensorFlow and other Python dependencies (listed in `requirements.txt`)

---

## Instructions

### 1. Clone the Project

To begin, clone the project repository from GitHub:

```bash
git clone https://github.com/lizashva/Image-Enhancement-with-Super-Resolution-and-Controlled-Noise-24-1-R-12.git
```

### 2. Install Dependencies

#### Backend (Python Server)

Navigate to the `python-server` folder and install the Python dependencies:

```bash
cd python-server
pip install -r requirements.txt
```

#### Frontend (Client)

Navigate to the `client` folder and install the npm packages:

```bash
cd ../client
npm install
```

---

## Models Setup

### Super Resolution Weights

Download the [Super Resolution Weights (super_resolution_weights.h5)](https://drive.google.com/drive/folders/1SGRqb7oJHWqFRinaHBjdj3pGn_Jsgnau) from the provided Google Drive link.

- **Location in Drive**: `Trained Models/Super Resolution/super_resolution_weights.h5`
- **Steps**:
  1. Create a folder named `models` inside the `python-server` directory if it doesn’t already exist.
  2. Move or copy the `super_resolution_weights.h5` file into this newly created `models` folder.

```bash
# Example structure after moving the file:
python-server/
├── models/
│   └── super_resolution_weights.h5
```

### Noise2Noise Weights and Models

Download the [Noise2Noise models and weights](https://drive.google.com/drive/folders/1SGRqb7oJHWqFRinaHBjdj3pGn_Jsgnau).

- **Location in Drive**: `Trained Models/Noise2Noise/*`
- This includes multiple subfolders like `bernoulli`, `brownGaussian`, `gaussian`, `poisson`, `text`.

- **Steps**:
  1. Navigate to `python-server/`.
  2. Create a folder named `models` inside the `N2N` directory if it doesn’t already exist.
  3. Download and move the folders (e.g., `bernoulli`, `brownGaussian`, etc.) into this `models` folder.

```bash
# Example structure after moving the folders:
python-server/
└──
    └── models/
            ├── bernoulli/
            ├── brownGaussian/
            ├── gaussian/
            ├── poisson/
            └── text/
```

---

## Getting Started

### Running the Client

To start the client-side interface, run the following commands:

```bash
cd client
npm run start-client
```

### Running the Backend Server

In a separate terminal, navigate to the `python-server` directory and start the server:

```bash
cd ../python-server
python app.py
```

The server will start on **port 5000**, and the client will start on **port 3000**.

---

## Using the Application

1. **Super Resolution Tab**: Users can upload a low-resolution image and apply the super-resolution model to enhance its quality.
2. **Noise Removal Tab**: Users can upload a noisy image, select a noise type, and apply the noise removal model.
3. **Training Section**: Configure and start training models on user-specified datasets.
