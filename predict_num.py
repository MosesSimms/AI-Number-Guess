from network import *
import torch

# Defaults to CPU if NVIDIA GPU isn't found 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the network and model
network = NeuralNetwork()

if torch.cuda.is_available():
    network.load_state_dict(torch.load("models/model.pth"))
else:
    network.load_state_dict(torch.load("models/model.pth", map_location=torch.device("cpu")))

# Sets the network to evaluation mode
network.eval()

network.to(device)
