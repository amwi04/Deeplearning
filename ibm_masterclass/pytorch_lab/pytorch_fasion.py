# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://cocl.us/DL0320EN_TOP_IMAGE">
#     <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0320EN/Assets/Images/Top.png" width="750" alt="IBM 10TB Storage" />
# </a>
# %% [markdown]
# <h1>Fashion-MNIST Project </h1>
# %% [markdown]
# <h2>Table of Contents</h2>
# %% [markdown]
# <p>In this project, you will classify  Fashion-MNIST dataset using convolutional neural networks.</p>
# <ul>
#
# <ul>
# <li><a href="#Preparation">Preparation</a></li>
# <li><a href="#Q1">Questions 1: Create a Dataset Class</li>
# <li><a href="#Train">Define Softmax, Criterion function, Optimizer and Train the Model</a></li>
#
# </ul>
#
#
# </ul>
#
# <p>Estimated Time Needed: <b>30 min</b></p>
# <hr>
# %% [markdown]
# <h2 id="Preparation" >Preparation</h2>
# %% [markdown]
# Download the datasets you needed for this lab.
# %% [markdown]
# The following are the PyTorch modules you are going to need

# %%
# PyTorch Modules you need for this lab
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
torch.manual_seed(0)

# %% [markdown]
# Import Non-PyTorch Modules

# %%
# Other non-PyTorch Modules

from matplotlib.pyplot import imshow
import matplotlib.pylab as plt

from PIL import Image


# %%
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = '+ str(data_sample[1]))


# %%
from tqdm.notebook import trange, tqdm

# %% [markdown]
# <hr>
# %% [markdown]
# <hr>
# %% [markdown]
# <h2 id="Questions 1">Questions 1: Create a Dataset Class</h2>
# %% [markdown]
# In this section, you will load a Dataset object, but first you must transform the dataset. Use the <code>Compose</code> function perform the following transforms:.
# <ol>
#     <li>use the transforms object to<code> Resize </code> to resize the image.</li>
#     <li>use the transforms object to<code> ToTensor </code> to concert the image to a tensor.</li>
# </ol>
#
# You will then take a screen shot of your validation data.
# %% [markdown]
# Use the compose function ot compse the

# %%
#Hint:

IMAGE_SIZE = 16

transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
transforms.ToTensor()#
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

# %% [markdown]
# <hr>
# %% [markdown]
# Create two dataset objects for the Fashion MNIST  dataset. One for training data called <code> dataset_train </code> and one for validation data <code>dataset_val</code>. You will be asked to take a screenshot of several samples.
# %% [markdown]
# <b>Hint:</b>
# <code>dsets.FashionMNIST(root= './data', train=???, transform=composed,  download=True)</code>

# %%

dataset_train=dsets.FashionMNIST(root= '.fashion/data', train=True, transform=composed,  download=True)
dataset_val=dsets.FashionMNIST(root= '.fashion/data', train=False, transform=composed, download=True)


# %%
for n,data_sample in enumerate(dataset_val):

    show_data(data_sample)
    plt.show()
    if n==2:
        break

# %% [markdown]
# <h2 id="Q2">Questions 2</h2>
# Create a Convolutional Neural Network class using ONE of the following constructors.  Train the network using the provided code then provide a screenshot of your training cost and accuracy with your validation data.
# %% [markdown]
# Constructor  using Batch Norm

# %%
class CNN_batch(nn.Module):

    # Contructor
    def __init__(self, out_1=16, out_2=32,number_of_classes=10):
        super(CNN_batch, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)

        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)

        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)
        self.bn_fc1 = nn.BatchNorm1d(10)

    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x=self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x=self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x=self.bn_fc1(x)
        return x

# %% [markdown]
# Constructor  for regular Convolutional Neural Network

# %%
class CNN(nn.Module):

    # Contructor
    def __init__(self, out_1=16, out_2=32,number_of_classes=10):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)

    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# %% [markdown]
# train loader  and validation loader

# %%
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=100 )
test_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=100 )

# %% [markdown]
# Convolutional Neural Network object

# %%
#model = CNN(out_1=16, out_2=32,number_of_classes=10)
model =CNN_batch(out_1=16, out_2=32,number_of_classes=10)

# %% [markdown]
# Code used to train the model

# %%
import time
start_time = time.time()

cost_list=[]
accuracy_list=[]
N_test=len(dataset_val)
learning_rate =0.1
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
n_epochs=10




# %%
for epoch in tqdm(range(n_epochs)):
    print(epoch)
    cost=0
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        cost+=loss.item()
    correct=0
    #perform a prediction on the validation  data
    model.eval()
    for x_test, y_test in test_loader:
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
    accuracy = correct / N_test
    accuracy_list.append(accuracy)
    cost_list.append(cost)

# %% [markdown]
# You will use the following to plot the Cost and accuracy for each epoch for the training and testing data, respectively.

# %%
import pickle
with open('cost_list.pkl', 'wb') as f:
    pickle.dump(cost_list, f)

with open('accuracy_list.pkl', 'wb') as f:
    pickle.dump(accuracy_list, f)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(cost_list, color=color)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('Cost', color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.set_xlabel('epoch', color=color)
ax2.plot( accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()

# %% [markdown]
# dataset: https://github.com/zalandoresearch/fashion-mnist
# %% [markdown]
# <h2>About the Authors:</h2>
#
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# %% [markdown]
# Other contributors: <a href="https://www.linkedin.com/in/michelleccarey/">Michelle Carey</a>, <a href="www.linkedin.com/in/jiahui-mavis-zhou-a4537814a">Mavis Zhou</a>
# %% [markdown]
# <hr>
# %% [markdown]
# Copyright &copy; 2018 <a href="cognitiveclass.ai?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu">cognitiveclass.ai</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.

