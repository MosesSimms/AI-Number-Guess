# AI-Number-Guess
Neural Network trained on the MNIST dataset to guess user drawn number


# How It Works
Using a raylib interface, users can draw a singe digit (0-9) for the computer to guess.
The computer loads a pretrained Pytorch Neural Network to process and guess the number.
As of now all outputs are sent to the terminal.

The Network model was trained using the MNIST dataset to an accuracy of 97%,
although the accuracy seems to be a little lower for user drawn digits.

# Current Issues
1. The UI is ugly.

2. Due to differences between images from the user and images from the dataset
the accuracy is a little lower than I would like.
The computer still struggles sometimes with 8's, 4's, and some other number I can't recall.

3. I don't like having to save the canvas as an image into the directory for process, 
but due to the way the python version of raylib works I didn't see another option.
raylib saves the texture image as a C structure, which pillow isnt able to work with.
Therefore I have to export the image first and then open with pillow.

4. Other things that I don't remember

# How To Run
## Prerequisites
* pytorch
* torchvision
* gzip
* idx2numpy
* numpy
* raylib (python)
* pillow

## Raylib Interface
As the repo already includes a model that I trained, simply run draw.py

Draw using the Left Mouse Button  
To generate computer guess press 'Enter'  
To clear drawing canvas press 'C'  

## Training Network
If you would like to retrain the network for better accuracy, run network.py  

There are clear variables for Batch Size, Learning Rate, Epochs, The Loss Function, and The Optimizer.

Make sure when altering Batch Size to also change the Batch Size parameter of the Dataloaders found near the top of the file as train_ds_loader and test_ds_loader

# To Do List
* Better UI
* Find A Better Solution For The Images
* Perhaps Improve Accuracy
* Compile As Executible

# Example Screenshots
<img src="https://raw.githubusercontent.com/MosesSimms/AI-Number-Guess/main/examples/window.png" width=256 height=256 alt="Example Screenshot"> <img src="https://raw.githubusercontent.com/MosesSimms/AI-Number-Guess/main/examples/terminal.png" width=400 height=256 alt="Example Screenshot">
