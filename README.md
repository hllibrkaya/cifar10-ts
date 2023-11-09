# Convolutional Neural Network (CNN) Training on CIFAR-10 Dataset

This project is related to training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset for image classification. The CIFAR-10 dataset contains 60,000 32x32 color images categorized into 10 different classes, making it a suitable benchmark for object recognition tasks.

## Project Workflow

In this project, I employ a CNN architecture to tackle the image classification task. CNNs are well-suited for image-related tasks due to their ability to learn hierarchical features from images. The main components of the CNN model used in this project are as follows:

- **Convolutional Layers (Conv2D)**: Convolutional layers play a crucial role in feature extraction from input images. These layers utilize learnable filters to convolve over the input images, detecting various patterns, edges, and more complex features.

- **Max-Pooling Layers (MaxPooling2D)**: Max-pooling layers are utilized to reduce the spatial dimensions of the feature maps produced by the convolutional layers. This helps in decreasing computational complexity and mitigating overfitting.

- **Flatten Layer**: The Flatten layer transforms the outputs from convolutional and max-pooling layers into a flattened vector, suitable for input into dense neural network layers.

- **Dense Layers**: Dense layers, also known as fully connected layers, are responsible for making predictions. In this project, multiple dense layers with various activation functions are employed to learn and predict class labels.

- **Dropout**: Dropout layers are incorporated to prevent overfitting by randomly deactivating a fraction of input units during training.

The specific architecture of the model in this project comprises multiple convolutional layers with ReLU activation, max-pooling layers, dropout layers, and densely connected layers. The model is trained to minimize the sparse categorical cross-entropy loss function and maximize classification accuracy.

The utilization of convolutional layers and max-pooling layers allows the model to capture and abstract meaningful features from input images, making it suitable for image classification tasks.

## Test Accuracy

After training the model on the CIFAR-10 dataset, I achieved a test accuracy of approximately 73.37%. This accuracy represents the model's ability to correctly classify images from the test set into their respective classes. The test accuracy is an important metric for evaluating the model's performance.

The test accuracy result indicates that the model has learned to recognize objects in the CIFAR-10 dataset with a reasonable level of accuracy. Please note that the specific accuracy may vary based on the training process and hyperparameters used.


## Requirements

To run this project, you will need the following libraries:

- TensorFlow
- Keras
- Matplotlib
- Scikit-learn

You can install these libraries using the following commands:

```bash
pip install tensorflow
pip install matplotlib
pip install scikit-learn
