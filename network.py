import torch
from torch import nn
from torch import Tensor
import torchvision
import torchvision.transforms.functional
import torch.utils.data as data_utils
import numpy as np
from data import *

# Setting GPU as Device if NVIDIA GPU found
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Network Class
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512,512),
			nn.ReLU(),
			nn.Linear(512, 10),
		)

	def forward(self, x: Tensor):

		# Was having issues with flatten when running without batch size
		# Checks if tensor shape doesnt have batch and changes flatten start and end params
		if len(x.shape) == 2:
			flatten = nn.Flatten(0, -1)
			x = flatten(x)
			logits = self.linear_relu_stack(x)
			return logits
		else:
			x = self.flatten(x)
			logits = self.linear_relu_stack(x)
			return logits
	

# Importing MNIST Dataset

	# Importing Training Data
train_image_data = torchvision.transforms.functional.to_tensor(np.array(train_images).T).to(device)
train_labels_data = torch.tensor(train_labels).to(device)
	# Images from dataset weren't orientated correctly
train_image_data = torch.rot90(train_image_data)
train_image_data = train_image_data.permute(1, 2, 0)


train_ds = data_utils.TensorDataset(train_image_data, train_labels_data)
train_ds_loader = data_utils.DataLoader(train_ds, batch_size=16, shuffle=True)


	# Importing Testing Data
test_image_data = torchvision.transforms.functional.to_tensor(np.array(test_images).T).to(device)
test_labels_data = torch.tensor(test_labels).to(device)
test_image_data = torch.rot90(test_image_data)

test_image_data = test_image_data.permute(1, 2, 0)

test_ds = data_utils.TensorDataset(test_image_data, test_labels_data)
test_ds_loader = data_utils.DataLoader(test_ds, batch_size=16, shuffle=True)


if __name__ == "__main__":

	# Honestly I wrote the network awhile ago following the Pytorch docs and I dont remember the use of this line
	# I think its just loading the next Image and Label set from the data
	train_features, train_labels = next(iter(train_ds_loader))

	# Creating Network model
	model = NeuralNetwork().to(device)

	# Training Network

	# Was able to get 97% accuracy with these parameters on images from the dataset
	# Not sure of exact accuracy on User drawn images
	learning_rate = 1e-2
	batch_size = 16

	def train_loop(dataloader, model, loss_fn, optimizer):
		size = len(dataloader.dataset)
		for batch, (X, y) in enumerate(dataloader):

			pred = model(X)
			loss = loss_fn(pred, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch % 100 == 0:
				loss, current = loss.item(), batch * len(X)
				print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


	def test_loop(dataloader, model, loss_fn):
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		test_loss, correct = 0, 0

		with torch.no_grad():
			for X, y in dataloader:
				pred = model(X)
				test_loss += loss_fn(pred, y).item()
				correct += (pred.argmax(1) == y).type(torch.float).sum().item()

		test_loss /= num_batches
		correct /= size
		print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

	epochs = 15

	for x in range(epochs):
		print(f"Epoch {x+1}\n-------------------------------")
		train_loop(train_ds_loader, model, loss_fn, optimizer)
		test_loop(test_ds_loader, model, loss_fn)
	print("Done")

	# Saving Network Parameters
	FILE = "./models/model.pth"
	torch.save(model.state_dict(), FILE)



