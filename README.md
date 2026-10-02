# CIFAR10-CNN

A simple CNN image classifier trained on the CIFAR-10 dataset using TensorFlow/Keras.

## Overview

This project uses a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories:

- Plane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Model

The CNN contains:

- Convolutional layers for feature extraction
- Max pooling layers for downsampling
- A fully connected layer for classification
- Softmax output for 10 classes

## Dataset

CIFAR-10 contains 60,000 RGB images with a resolution of 32×32 pixels.

- 50,000 training images
- 10,000 test images
- 10 classes

## Usage

Train the model:

```bash
python train.py
