# Image Classification with PyTorch

This project trains a deep learning model to classify flower images using transfer learning. It was built as part of the Udacity AI Programming with Python Nanodegree and demonstrates skills in PyTorch, data preprocessing, model training, and inference via command-line tools.

---

## Project Overview

The goal of this project is to develop an AI image classifier that can predict the species of a flower based on a given image. This includes:

- Building and training a neural network using **PyTorch**
- Using **transfer learning** with a pretrained CNN (VGG16)
- Creating a command-line interface to train models and make predictions
- Saving and loading model checkpoints for reuse

---

## Skills Applied

- Transfer learning with **pretrained CNNs**
- Training custom classifiers with **PyTorch**
- Data augmentation & normalization using **torchvision.transforms**
- Building CLI tools with **argparse**

---

## What I Built

The project is divided into two parts:

1. **Jupyter Notebook (`ImageClassifier.ipynb`)**
   - Loads and augments flower image data
   - Loads pretrained VGG16 model and attaches a custom classifier
   - Trains the model and tests it on unseen data
   - Saves the trained model checkpoint

2. **Command Line Application**
   - `train.py`: Trains a model from scratch or resumes from a checkpoint
   - `predict.py`: Uses a saved model to predict flower species from new images

---

## Results

- Achieved **72.66% test accuracy** on a 102-category flower dataset.
- Trained for **5 epochs** using frozen VGG16 convolutional layers and a custom feedforward classifier.
- Implemented training with **GPU support**, data augmentation, and loss/accuracy tracking per epoch.
- Successfully outputs **top 5 most probable classes** with confidence scores and category names.

---

## Example Usage

**Training:**
```bash
python train.py flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.003 --hidden_units 512 --epochs 5 --gpu
```
**Prediction:**
```bash
python predict.py input.jpg checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

---

## Project Structure
```text
image-classification-udacity/
├── train.py                 # Train a new network and save checkpoint
├── predict.py              # Predict image class using trained model
├── ImageClassifier.ipynb   # Jupyter Notebook for development and training
├── cat_to_name.json        # Mapping from category label to flower name
├── checkpoint.pth          # Saved trained model
└── README.md               # Project documentation
```

---

## 📚 Dataset
[Oxford 102 Category Flower Dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102)

