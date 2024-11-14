import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

# hyperparams/ dimensions
d1 = 100
d2 = 100


class VariationalAutoEncoder(nn.Module):
    def __init__(self,d1 = d1, d2 = d2, latent_dimension = 56):
        super().__init__()
        
        self.d1 = d1
        self.d2 = d2
        self.latentd = latent_dimension
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # be able to add hidden dims...
        modules = []
        hidden_dims = [32,64,128,256,512,512] # extra for dim-reduction
        
        # Encoder
        # Conv2d takes in a dimension, outputs this dimension...
        
        '''Convolutional Layer (
            Channels_in: how many input kernel
            channels_out: how many kernels we will have (subkernels for each input)
            kernel_size: how large our kernels do their scans, and end up
            stride
            padding)
            
            At end, we will have 512 2x2 feature-maps given 64x64 images to start'''
        
        in_channels = 3
        for h in hidden_dims:
            modules.append(
                nn.Sequential(
                    # convolutional layer
                    nn.Conv2d(in_channels,h,kernel_size=3,stride=2,padding=1),
                    # batchnorm for stablization
                    nn.BatchNorm2d(h),
                    # Non-linearity
                    nn.LeakyReLU(),
                )
            )
            
            # setting last dim equal...
            in_channels = h
            
        # just have a list of sequentials, unpack, and do in sequence!
        self.encoder_modules = nn.Sequential(*modules)
        
        # multiply by 4 because of spacial dimensions.
        self.l_mu = nn.Linear(6144,latent_dimension)
        self.l_logvar = nn.Linear(6144,latent_dimension)
        
        '''
        *Building Decoder*
        Reverse our list, and create the inverse convolutional layers.
        
        Final layer is special: it has an added convolutional layer,
        in order for us to be able to output 3 channels
        
        '''
        modules = []
        
        hidden_dims.reverse()
        
        # kernel 3, stride 2, padding 1, output padding 1
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],hidden_dims[i+1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU(),
                )
            )
            
        self.decoder_modules = nn.Sequential(*modules)
        
        
        # first part of reconstruction
        self.decoder_input = nn.Linear(latent_dimension,6144)
        
        # final layer to translate into color
        self.final_adj = nn.AdaptiveAvgPool2d([self.d1,self.d2]) # need this for shapes!
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]), # 2d over 4d input
            nn.LeakyReLU(),
            # make to shape of images match here!
            nn.Conv2d(hidden_dims[-1],3,kernel_size=3,padding=1,stride=1), 
            nn.Tanh(),
        )
        
    def encoder(self,x):
        # encoding to latent vectors
        x = self.encoder_modules(x)
        x = x.flatten(1) # flatten all dimensions starting at 1
        
        mu = self.l_mu(x)
        logvar = self.l_logvar(x)
        
        return mu, logvar
    
    def reparameterization(self,mu,logvar):
        # sample from z, pass to decoder
        epsilon = torch.randn_like(mu)
        
        sampled = epsilon * torch.exp(0.5*logvar) + mu
        
        return sampled, mu, logvar
        
    def decoder(self,x):
        # reconstructing sample from z
        # B,in --> B,C,out
        
        # decodes a latent-vector...
        x = self.decoder_input(x)
        
        x = x.view(x.shape[0],512,4,3) # B, C, kernel
        
        x = self.decoder_modules(x)
        x = self.final_layer(x)
        x = self.final_adj(x)
        
        x = self.tanh(x)
        
        return x
    
    def get_latent(self,x):
        # get sampled latent vector for an image
        # encoder
        mu,logvar = self.encoder(x)
        
        # sampling
        latent,mu,logvar = self.reparameterization(mu,logvar)

        return latent, mu, logvar
    
    def forward(self,x):
        # checking for shape errors
        assert(self.d1*self.d2 == x.shape[-2]*x.shape[-1])
        if x.dim() <= 3:
            x = x.unsqueeze(0)

        # encoder
        mu,logvar = self.encoder(x)
        
        # sampling
        latent,mu,logvar = self.reparameterization(mu,logvar)
        
        # decoder
        out = self.decoder(latent)
        
        return out, mu, logvar
    
    

    



# IMAGES MUST BE 218,178
def run_test():
    test = torch.randn((3,218,178))
    model = VariationalAutoEncoder(218,178,3)
    out,mu,sigma = model(test)
    print("***Results***")
    print(out.shape,mu.shape,sigma.shape)
    


# TO-DO
'''
1. Understand importance of batchnorm 
2. Implement residual connections?
3. Reason better with conv. layers!

'''

# our adaptive pool 256,192 --> 218,178
# will take average by doing some kernel, same
# as normal images..