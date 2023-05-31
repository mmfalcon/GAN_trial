import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import normal
import torch.nn.functional as F

loadModel = True
#attributes of data to be generated
Series_Length = 512
mini_batch_size = 15

#attributes of generator-io
g_input_size = Series_Length #can be anything, preferable the series length    
g_hidden_size = 16  
g_output_size = Series_Length

#attributes of discriminator-io
d_input_size = Series_Length
d_hidden_size = 16  
d_output_size = 1

d_minibatch_size = 15
g_minibatch_size = 15
num_epochs = 100000
print_interval = 1000

d_learning_rate = 2e-4
g_learning_rate = 2e-4

#make this example reproducible-reg random generations
#seed(1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU device : ' + str(device))

def generate_nGaussian(total_samples,Series_length, mini_batch_size): #batch_size is not used, total samples = batchsize for now
	#create a datapool of training data
	mean = np.random.default_rng().uniform(-1,1,total_samples)
	stddev = np.random.default_rng().uniform(0,0.5,total_samples)
	data_matrix = []
	data_sample = []
	for i in range(0,mean.size):
		data_sample = normal(loc=mean[i], scale=stddev[i], size=Series_length)
		data_matrix.append(data_sample)
	data_matrix = np.stack(data_matrix, axis=0)
	output = torch.from_numpy(data_matrix) #convert nd numpy array to torch tensor
	output = torch.tensor(data_matrix, dtype=torch.float32) ##by default it is type(torch.float64)
	output.requires_grad_() 
	return output

def get_noise_sampler():
    return lambda m, n: torch.rand(m, n).requires_grad_()  # Uniform-dist data into generator, _NOT_ Gaussian 

noise_data  = get_noise_sampler()

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size*16)
        self.map2 = nn.Linear(hidden_size*16, hidden_size*4)
        self.map3 = nn.Linear(hidden_size*4, hidden_size*8)
        self.map4 = nn.Linear(hidden_size*8, output_size)
        self.xfer = torch.nn.SELU()    
    def forward(self, x):
        x = self.xfer( self.map1(x) )
        x = self.xfer( self.map2(x) )
        x = self.xfer( self.map3(x) )
        x = self.map4( x )
        return self.xfer( x )
        
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size*16)
        self.map2 = nn.Linear(hidden_size*16, hidden_size*4)
        self.map3 = nn.Linear(hidden_size*4, hidden_size)
        self.map4 = nn.Linear(hidden_size, output_size)
        self.elu = torch.nn.LeakyReLU()
    def forward(self, x):
        x = self.elu(self.map1(x))
        x = self.elu(self.map2(x))
        x = self.elu(self.map3(x))
        x = self.map4(x)
        return torch.sigmoid( x )

def train_D_on_actual() :
    real_data = generate_nGaussian(d_minibatch_size,d_input_size,5)
    real_decision = D( real_data )
    real_error = criterion( real_decision, torch.ones( d_minibatch_size, 1 ))  # ones = true
    real_error.backward()
    
def train_D_on_generated() :
    noise = noise_data( d_minibatch_size, g_input_size )
    fake_data = G( noise ) 
    fake_decision = D( fake_data )
    fake_error = criterion( fake_decision, torch.zeros( d_minibatch_size, 1 ))  # zeros = fake
    fake_error.backward()

def train_G():
    noise = noise_data( g_minibatch_size, g_input_size )
    fake_data = G( noise )
    fake_decision = D( fake_data )
    error = criterion( fake_decision, torch.ones( g_minibatch_size, 1 ) )
    error.backward()
    return error.item(), fake_data #returns a scalar and the fake dataset

def test_G(fixed_latent):
     fake_data = G(fixed_latent)
     fake_decision = D( fake_data )
     error = criterion( fake_decision, torch.ones( g_minibatch_size, 1 ) )
     return error.item(), fake_data

def draw( data ) :    
    plt.figure()
    d = data.tolist() if isinstance(data, torch.Tensor ) else data
    plt.plot( d ) 
    plt.show()

def write_progress(real_sample,fake_sample,number):
	real_sample = real_sample.detach().numpy() 
	fake_sample = fake_sample.detach().numpy()
	realFile = './data/real_'+ str(number).zfill(10) + '.dat'
	fakeFile = './data/fake_'+ str(number).zfill(10) + '.dat'
	with open(realFile, 'w') as f:
		for i in real_sample[0,:]:
		    f.write(str(i) + '\n')
	f.close()
	with open(fakeFile, 'w') as f:
		for i in fake_sample[0,:]:
		    f.write(str(i) + '\n')
	f.close()
	
def write_errors(losses):
	number = len(losses)
	lossFile = './data/losses_'+ str(number).zfill(10) + '.dat'
	with open(lossFile, 'w') as f:
		for i in range(0,number,1):
		    f.write(str(i) + '\t' + str(losses[i]) + '\n')
	f.close()

def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch

#################################################################################
########################## M A I N ##############################################
#################################################################################
#def main():
if ((loadModel == True) and (os.path.isfile('./models/generator_model.pth'))):
	#torch : Load model
	G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
	D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
	criterion = nn.BCELoss()
	d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate ) #adjusting weights, , momentum = 0.1 ?
	g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate ) 
#	G.load_state_dict(torch.load('./models/generator_model'))
#	D.load_state_dict(torch.load('./models/discriminator_model'))
#	G.eval()
#	D.eval()
	G, g_optimizer, g_start_epoch = load_checkpoint(G, g_optimizer, './models/generator_model.pth' )
	D, d_optimizer, d_start_epoch = load_checkpoint(D, d_optimizer, './models/discriminator_model.pth' )
	d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate ) #adjusting weights
	g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate ) 
	G.eval()
	D.eval()
	#G = G.to(device)
	# (For GPU) now individually transfer the optimizer parts...
	#for state in g_optimizer.state.values():
	#	for k, v in state.items():
	#		if isinstance(v, torch.Tensor):
	#			state[k] = v.to(device)
	
else:
	G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
	D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
	criterion = nn.BCELoss()
	d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate ) #adjusting weights
	g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate ) 

print(d_optimizer)
print(g_optimizer)	
#get a fixed latent vector
fixed_latent  = noise_data( d_minibatch_size, g_input_size )

#open file to write progress
if not os.path.exists('./data'):
   os.makedirs('./data')

#TRAINING PROCESS
losses = []
real_err = []
fake_err = []
for epoch in range(num_epochs):
    D.zero_grad()
    train_D_on_actual()     
    train_D_on_generated() 
    d_optimizer.step()
    G.zero_grad()
    loss,generated = train_G()
    g_optimizer.step()
    losses.append( loss )
    if( epoch % print_interval) == (print_interval-1) :
        print( "Epoch %6d. Loss %5.3f" % ( epoch+1, loss ) )
        test_loss,test_generated = test_G(fixed_latent)
        test_real = generate_nGaussian(d_minibatch_size,d_input_size,5)
        write_progress(test_real,test_generated,epoch+1)
        
print( "Training complete" )
write_errors(losses)

#torch : Save model 
if not os.path.exists('./models'):
   os.makedirs('./models')
   
g_checkpoint = { 
    'epoch': epoch+1,
    'model': G.state_dict(),
    'optimizer': g_optimizer.state_dict()}
torch.save(g_checkpoint, './models/generator_model.pth')

d_checkpoint = { 
    'epoch': epoch+1,
    'model': D.state_dict(),
    'optimizer': d_optimizer.state_dict()}
torch.save(d_checkpoint, './models/discriminator_model.pth')

test_loss,test_generated = test_G(fixed_latent)
######display updated generations
d1 = torch.empty( test_generated.size(0), 63 ) 
for i in range( 0, d1.size(0) ) :
	d1[i] = torch.histc( test_generated[i], min=-5, max=5, bins= 63 )
draw( d1.t() )

#DISPLAY LOSS
plt.figure()
plt.plot(losses) 
plt.show()
print("End.")

#end main()

#if __name__ == "__main__":
#    main()


