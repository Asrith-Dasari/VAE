import numpy as np 
import torch as torch
import torch.nn as nn
import torch.optim as optim
import pdb
import os 
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.utils import save_image
from torch.nn import functional as F




# Path of the train images
images_dataset_path = "./monet_jpg/"

# Creating dataset class
class monet_dataset(Dataset):
    """
    Monet pictures dataset
    """

    def __init__(self,path):

        self.path = path
        self.elements_to_dict ={}

        for index,file in enumerate(os.listdir(self.path)):
            self.elements_to_dict[index] = file


    def __len__(self):

        return(len(os.listdir(self.path)))

    def __getitem__(self,idx):

        img = cv2.imread(os.path.join(self.path,self.elements_to_dict[idx]))
        trans = transforms.ToTensor()
        tensor = trans(img)
        return tensor

#Creating the dataset object
data = monet_dataset(path = images_dataset_path)

batch_size = 5

# Creating Dataset Loader object
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

num_batches = len(data_loader)

class VAE_Encoder(torch.nn.Module):
	"""
	Create the VAE Model based Architecture
	"""

	def __init__(self):
		super(VAE_Encoder,self).__init__()

		#Encoder Part of the Network
		self.conv1 = nn.Conv2d(in_channels =3,out_channels = 30,kernel_size = 7, stride = 2)
		self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 1)
		self.bc1 = nn.BatchNorm2d(30)
		self.LeakyRelu = nn.LeakyReLU(0.2)

		self.conv2 = nn.Conv2d(in_channels = 30, out_channels = 50, kernel_size = 7, stride=2)
		self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride= 1)
		self.bc2 = nn.BatchNorm2d(50)

		self.conv3 = nn.Conv2d(in_channels = 50, out_channels = 100, kernel_size = 5, stride=2)
		self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 1)
		self.bc3 = nn.BatchNorm2d(100)


		self.conv4 = nn.Conv2d(in_channels = 100, out_channels = 150, kernel_size = 3, stride=2)
		self.max_pool4 = nn.MaxPool2d(kernel_size = 2, stride = 1)
		self.bc4 = nn.BatchNorm2d(150)

		#Output dimension after flattening is [10, 100, 237, 237]

		#Mean and Standard deviation
		self.mean = nn.Linear(18150,500)
		self.log_variance = nn.Linear(18150,500)



	def forward(self, x):

		# Encoder part of the network
		x = self.conv1(x)
		x = self.max_pool1(x)
		x = self.bc1(x)
		x = self.LeakyRelu(x)

		x = self.conv2(x)
		x = self.max_pool2(x)
		x = self.bc2(x)
		x = self.LeakyRelu(x)

		x = self.conv3(x)
		x = self.max_pool3(x)
		x = self.bc3(x)
		x = self.LeakyRelu(x)

		x = self.conv4(x)
		x = self.max_pool4(x)
		x = self.bc4(x)
		x = self.LeakyRelu(x)

		x = torch.flatten(x, start_dim = 1)

		#Mean and Standar deviation Nodes
		mean = self.mean(x)
		log_variance = self.log_variance(x)

		return mean, log_variance


class VAE_Decoder(nn.Module):
	"""docstring for VAE_Decoder"""
	def __init__(self):
		super(VAE_Decoder, self).__init__()

		#Decoder Part of the network
		#Wanted to build 2 types of the Decoder Network
		#1.Use only NN's 
		#2.Use Conv2dTranspose
		
		#Building Decoder using Vanilla NN

		self.LeakyRelu = nn.LeakyReLU(0.3)
		self.Relu = nn.ReLU()
		#self.Linear1  = nn.Linear(1500, 500)
		self.Linear2  = nn.Linear(500, 100)
		self.Linear3  = nn.Linear(100, 1000)
		self.Linear4 = nn.Linear(1000, 256*256*3)

	def forward(self,mean,log_variance):

		noise = torch.tensor(np.random.normal(0,1, (mean.shape[0],mean.shape[1])))
		#Constructing the re_parameterization
		x = mean+ torch.exp(0.5*log_variance)*noise.float()

		#x = self.Linear1(x)
		#x = self.LeakyRelu(x)

		x = self.Linear2(x)
		x = self.LeakyRelu(x)

		x = self.Linear3(x)
		x = self.LeakyRelu(x)

		x = self.Linear4(x)
		x = self.Relu(x)

		x = torch.reshape(x,(x.shape[0],3,256,256))

		return x

		

def CustomVariationalLoss(real, generated, mean, log_variance):
	"""
	VAE loss = || real- generated|| ^2 + KL divergence(real,generated)

	VAE loss = MSELOSS(real,generated) + -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
	"""

	mse_loss = nn.MSELoss()
	MSE = mse_loss(generated,real)

	#BCE = nn.BCELoss()
	#bce_loss = BCE(generated, real)
	#BCE = F.binary_cross_entropy(generated, real, reduction='mean')
	
	kl_loss = torch.mean(-0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()),dim=0)

	return MSE+kl_loss



Encoder = VAE_Encoder()
Decoder = VAE_Decoder()

#optim.Adam(discriminator.parameters(), lr=0.002)
encoder_optimizer = optim.Adam(Encoder.parameters(), lr = 0.0005)
decoder_optimizer = optim.Adam(Decoder.parameters(), lr = 0.0005)

num_epochs = 100
epoch_loss = []
avg_loss = []

for epoch in range(num_epochs):

    for n_batch, batch_data in enumerate(data_loader):

    	mean,log_variance =  Encoder(batch_data)
    	generated_image = Decoder(mean,log_variance)

    	loss = CustomVariationalLoss(batch_data, generated_image, mean, log_variance)

    	

    	loss.backward()

    	encoder_optimizer.step()
    	decoder_optimizer.step()

    	encoder_optimizer.zero_grad()
    	decoder_optimizer.zero_grad()

    	avg_loss.append(loss.item())

    	if (n_batch%5 == 0):
    		x = Decoder(mean,log_variance)
    		for i in range(x.shape[0]):
    			if not os.path.exists(os.path.join(os.getcwd(),'Generated_Images')):
    				os.mkdir('Generated_Images')
    			if (i==0): 
    				save_image(x[i], './generated_images/epoch_{}_batch_{}_{}.png'.format(epoch,n_batch,i))

    print("Epoch {}---Loss {}".format(epoch,sum(avg_loss)/len(avg_loss)))
