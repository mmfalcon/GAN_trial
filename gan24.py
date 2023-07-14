import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import normal
from numpy.random import uniform
import torch.nn.functional as F
from scipy.stats import norm
from torch.autograd import Variable
import math
import random
import time

# Real data distribution
class RealDistribution:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_samples):
        samples = np.random.normal(self.mu, self.sigma, num_samples)
        #samples.sort()
        
        return samples

# Noise data distribution as inputs for the generator
class NoiseDistribution:
    def __init__(self, data_range):
        self.data_range = data_range

    def sample(self, num_samples):
        samples = np.random.uniform(low=-1.0, high=1.0, size=num_samples)
        
        return samples

class Generator(nn.Module):
    def __init__(self, g_input_size, hidden_size, g_output_size):
        super(Generator, self).__init__()

	#Layer1
        fc1 = nn.Linear(g_input_size, hidden_size, bias=True)
        self.hidden_layer1 = nn.Sequential(fc1, nn.ReLU())
    # Output layer
        self.output_layer = nn.Linear(hidden_size, g_output_size, bias=True)

    def forward(self, x):
        h1 = self.hidden_layer1(x)
#        h2 = self.hidden_layer2(h1)
#        h3 = self.hidden_layer3(h2)
        out = self.output_layer(h1)
        
        return out
     
class Discriminator(nn.Module):
    def __init__(self, d_input_size, hidden_size, d_output_size):
        super(Discriminator, self).__init__()

        # Layer1
        fc1 = nn.Linear(d_input_size, hidden_size, bias=True)
        self.hidden_layer1 = nn.Sequential(fc1, nn.ReLU())

        # Output Layer 
        fc4 = nn.Linear(hidden_size, d_output_size, bias=True)
        self.output_layer = nn.Sequential(fc4, nn.Sigmoid())

    def forward(self, x):
        h1 = self.hidden_layer1(x)
#        h2 = self.hidden_layer2(h1)
#        h3 = self.hidden_layer3(h2)
        out = self.output_layer(h1)
        
        return out
        
# Test samples
class TestSample: 
    def __init__(self, discriminator, generator, realD, noiseD, data_range, batch_size, num_samples, num_bins):
        self.D = discriminator
        self.G = generator
        self.realD = realD
        self.noiseD = noiseD
        self.bs = batch_size
        self.num_samples = num_samples
        self.num_bins = num_bins
        self.xs = np.linspace(-data_range, data_range, num_samples)
        self.bins = np.linspace(-data_range, data_range, num_bins)

    def real_distribution(self):
        d = self.realD.sample(self.num_samples)
        p_data, _ = np.histogram(d, self.bins, density=True)
        p_data = p_data
        return p_data

    def gen_distribution(self):
        n_batches = int(self.num_samples / self.bs)
        z_ = self.noiseD.sample(self.num_samples)
        z_ = torch.from_numpy(z_).type(torch.FloatTensor)
        z = torch.reshape(z_, (n_batches, self.bs))  
        g =  self.G(z)   
        g = torch.reshape(g, (self.num_samples, ))
        p_gen, _ = np.histogram(g.detach().numpy(), self.bins, density=True)
        return p_gen

    def decision_boundary(self):
        db = np.zeros((self.num_samples, 1)) #array of size( num_sample) = bs*nbatches
        for i in range(self.num_samples // self.bs): #no of batches(int), i->each batch no 
            x_ = self.xs[self.bs*i:self.bs*(i+1)]
            x_ = Variable(torch.FloatTensor(np.reshape(x_, [self.bs, 1]))) #x variable tensor for each batch

            db[self.bs*i:self.bs*(i+1)] = self.D(x_).detach().numpy() #probability of each value in the sample domain [-range  to +range]

        return db 


    def transformation_G(self):
        z_ = self.xs
        n_batches = int(self.num_samples / self.bs)
        z_ = torch.from_numpy(z_).type(torch.FloatTensor)
        z = torch.reshape(z_, (n_batches, self.bs)) 
        g = self.G(z)
        g = torch.reshape(g, (self.num_samples, ))
        trans = g.detach().numpy()
        return z_, trans
    
class Display:
    def __init__(self, num_samples, num_bins, data_range, mu, sigma):
        self.num_samples = num_samples
        self.num_bins = num_bins
        self.data_range = data_range
        self.mu = mu
        self.sigma = sigma
    
    def plot_result(self, p_data, p_gen, epoch):
        p_x = np.linspace(-self.data_range, self.data_range, len(p_data))

        f, ax = plt.subplots(1)
        ax.set_ylim(0, max(1, np.max(p_data)*1.1))
        ax.set_xlim(-1.2, 1.2)
        plt.plot(p_x, p_data, label='Real data')
        plt.plot(p_x, p_gen, label='Generated data')
        plt.title('Gaussian Approximation using GAN: ' + '(epoch = %3g)' % epoch)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend(loc=1)
        plt.grid(True)

        # Save plot
        save_dir = "result/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt. savefig(save_dir + 'Gaussian' + '_mu_%g' % self.mu +
                    '_sigma_%g' % self.sigma + '_ep_' + str(epoch).zfill(7) + '.png')
        plt.show()        

    def plot_gtransform(self, z_, trans, epoch):
        
        f_trans, ax_trans = plt.subplots()
        plt.plot(z_, trans, label='G transformation')
        plt.title('G mapping: Uniform[-1,1] to Normal[-1,1] : ' + '(epoch = %3g)' % epoch)
        plt.xlabel('z')
        plt.ylabel('z_g')
        plt.legend(loc=1)
        plt.grid(True)
        
        # Save plot
        save_dir = "G_transform/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig(save_dir + 'gtransform' + '_mu_%g' % self.mu +
                    '_sigma_%g' % self.sigma + '_ep_%g' % epoch + '.png')
        #plt.close()
        plt.show()        

###################################################
# Hyper-parameters#################################
###################################################
mu = 0.0
sigma = 0.3
data_range = 1

hidden_size = 64
G_learning_rate = 0.001
D_learning_rate = 0.001

d_input_size = 64
d_output_size = 1
g_input_size = 64
g_output_size = 64

# Samples
realData = RealDistribution(mu, sigma)    
noiseData = NoiseDistribution(data_range) 

# Create models
G = Generator(g_input_size, hidden_size, g_output_size)
D = Discriminator(d_input_size, hidden_size, d_output_size)

PRE_TRAIN = True
# Load models
if (os.path.isfile('./SavedModel/G_train.pt')):
    G.load_state_dict(torch.load('./SavedModel/G_train.pt'))
    D.load_state_dict(torch.load('./SavedModel/D_train.pt'))
    PRE_TRAIN = False
    print('Loaded D and G models..')
else :
    PRE_TRAIN = True
    pass

#################################################
############ PRE TRAIN DISCRIMINATOR ############
#################################################

if(PRE_TRAIN == True):
    
    bceloss = torch.nn.BCELoss()
    D_optimizer = torch.optim.SGD(D.parameters(), lr=D_learning_rate)
    num_samples_pre = 64000
    batch_size_pre = 64
    n_batches_pre = int(num_samples_pre/batch_size_pre)
    num_bins_pre = 50 # number of equal-width bins in a histogram
    num_epoch_pre = 5000
    D_pre_losses = []
    for epoch in range(num_epoch_pre): 
        
        x_ = realData.sample(num_samples_pre)
        x_ = torch.from_numpy(x_).type(torch.FloatTensor)
        x = torch.reshape(x_, (n_batches_pre, batch_size_pre))
        #x.requires_grad_()
        y_real_pre = torch.ones(n_batches_pre, 1)
        
        z_ = noiseData.sample(num_samples_pre)
        z_ = torch.from_numpy(z_).type(torch.FloatTensor)  
        z = torch.reshape(z_, (n_batches_pre, batch_size_pre))
        #z.requires_grad_()
        y_fake_pre = torch.zeros(n_batches_pre, 1)

        # Train D model
        D_optimizer.zero_grad()
        D_pre_decision = D(x)
        D_pre_loss = bceloss(D(x), y_real_pre) + bceloss(D(z), y_fake_pre)
        
        # Back propagation
        D.zero_grad()
        D_pre_loss.backward()
        D_optimizer.step()
        
        # Save loss values for plot
        D_pre_losses.append(D_pre_loss.item())
        
        if epoch % 100 == 0:
            print(epoch, D_pre_loss.item())
            
            # Plot loss
            fig, ax = plt.subplots()
            losses = np.array(D_pre_losses)
            plt.plot(losses, label='Pre-train loss')
            plt.title("Pre-training Loss")
            plt.legend()
            plt.show()

# Save generator, discriminator
torch.save(G.state_dict(), './SavedModel/G_pretrain.pt')
torch.save(D.state_dict(), './SavedModel/D_pretrain.pt')

print("G's state_dict:")
for param_tensor in G.state_dict():
    print(param_tensor, "\t", G.state_dict()[param_tensor].size())
    
print("D's state_dict:")
for param_tensor in D.state_dict():
    print(param_tensor, "\t", D.state_dict()[param_tensor].size())

print("Pre_Training D complete.")

##########################################
############## TRAIN GAN #################
##########################################
start_gan = time.time()

# Optimizers
D_optimizer = torch.optim.SGD(D.parameters(), lr=D_learning_rate, momentum = 0.5)
G_optimizer = torch.optim.SGD(G.parameters(), lr=G_learning_rate, momentum = 0.5)

D_scheduler = torch.optim.lr_scheduler.StepLR(D_optimizer, step_size=100000, gamma=0.5)
G_scheduler = torch.optim.lr_scheduler.StepLR(G_optimizer, step_size=100000, gamma=0.5)	


# Loss function
bceloss = torch.nn.BCELoss()

D_losses = []
G_losses = []
D_real_losses = []
D_fake_losses = []

num_epochs = 1000000
num_samples = 64000 #for testing
batch_size = 64
n_batches = int(num_samples/batch_size)
num_bins = 50 #for test
test_batch_size = batch_size
test_num_samples = 32000

for epoch in range(num_epochs):
    # Generate real/fake samples
    x_ = realData.sample(num_samples)
    x_ = torch.from_numpy(x_).type(torch.FloatTensor)
    x = torch.reshape(x_, (n_batches, batch_size))  
    x.requires_grad_()
    y_real_ = torch.ones(n_batches, 1) 

    z_ = noiseData.sample(num_samples)
    z_ = torch.from_numpy(z_).type(torch.FloatTensor)
    z = torch.reshape(z_, (n_batches, batch_size)) 
    z.requires_grad_()
    z = G(z)
    y_fake_ = torch.zeros(n_batches, 1) 

    D.zero_grad()
    # Train discriminator with real data
    D_real_decision = D(x)
    D_real_loss = bceloss(D_real_decision, y_real_)
    #D_real_loss.backward()
    # Train discriminator with fake data
    D_fake_decision = D(z)
    D_fake_loss = bceloss(D_fake_decision, y_fake_)
    #D_fake_loss.backward()
    D_loss = D_real_loss + D_fake_loss
    # Back propagation
    D_loss.backward()
    D_optimizer.step()

    # Train generator
    z_ = noiseData.sample(num_samples)
    z_ = torch.from_numpy(z_).type(torch.FloatTensor)
    z = torch.reshape(z_, (n_batches, g_input_size))
    z = G(z)
    G.zero_grad()
    D_fake_decision = D(z)
    G_loss = bceloss(D_fake_decision, y_real_) 

    # Back propagation
    G_loss.backward()
    G_optimizer.step()

    D_scheduler.step()
    G_scheduler.step()

    # Save loss values for plot
    D_losses.append(D_loss.item())
    G_losses.append(G_loss.item())
    D_real_losses.append(D_real_loss.item())
    D_fake_losses.append(D_fake_loss.item())

    if epoch % 1000 == 0:
        print('epoch:{}, D loss:{}, G loss:{}, G_lr:{}'.format(epoch, round(D_loss.item(),4), round(G_loss.item(),4), G_optimizer.param_groups[0]['lr']))
        # Test sample after GAN-training
        sample = TestSample(D, G, realData, noiseData, data_range, test_batch_size, test_num_samples, num_bins)
        p_data = sample.real_distribution()
        p_gen = sample.gen_distribution()
        latent, gen = sample.transformation_G()

        if epoch % 1000 == 0:
        # Display result
	        display = Display(num_samples, num_bins, data_range, mu, sigma)
	        display.plot_result(p_data, p_gen, epoch)

        if epoch % 10000 == 0:
           torch.save(G.state_dict(), './SavedModel/G_train_temp.pt')
           torch.save(D.state_dict(), './SavedModel/D_train_temp.pt')
           display.plot_gtransform(latent, gen, epoch)
           np.savetxt('g_transform_latest.dat', np.c_[ latent, gen ])

end_gan = time.time()
print('Time elapsed in GAN training = ' + str(end_gan - start_gan))

# Save losses into text file
np.savetxt('losses.txt', np.c_[ D_real_losses, D_fake_losses, D_losses, G_losses])

# Save generator, discriminator
torch.save(G.state_dict(), './SavedModel/G_train.pt')
torch.save(D.state_dict(), './SavedModel/D_train.pt')

# Save animation
import imageio.v2 as imageio

png_dir = './result/'
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))

imageio.mimsave('./training.gif', images, fps=5)

print('End.')