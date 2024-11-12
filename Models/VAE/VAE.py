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
        
        # encoder takes in B,C,H*W --> B,C,Out
        self.h1 = nn.Linear(self.d1*self.d2*3, 300)
        self.h2 = nn.Linear(300, 256)
        self.h3 = nn.Linear(256, 128)
        self.mu1 = nn.Linear(128, latent_dimension)
        self.sigma1 = nn.Linear(128, latent_dimension)
        
        # decoder
        self.h4inv = nn.Linear(latent_dimension,128)
        self.h3inv = nn.Linear(128,256)
        self.h2inv = nn.Linear(256,300)
        self.h1inv = nn.Linear(300,self.d1*self.d2*3)
        
    def encoder(self,x):
        # encoding to latent vectors
        h = self.relu(self.h1(x))
        h = self.relu(self.h2(h))
        h = self.relu(self.h3(h))
        
        # parameterizing z
        mu = self.mu1(h)
        sigma = self.sigma1(h)
        
        return mu, sigma
    
    def reparameterization(self,mu,logvar):
        # sample from z, pass to decoder
        epsilon = torch.randn_like(mu)
        
        sampled = epsilon * torch.exp(0.5*logvar) + mu
        
        return sampled, mu, logvar
        
    def decoder(self,x):
        # reconstructing sample from z
        # B,C,in --> B,C,out
        h = self.relu(self.h4inv(x))
        h = self.relu(self.h3inv(h))
        h = self.relu(self.h2inv(h))
        out = self.sigmoid(self.h1inv(h)) # to constrain values
        
        return out
    
    def forward(self,x):
        assert(x.shape[2]*x.shape[3]*3 == self.d1*self.d2*3)
        x = x.view(-1,self.d1*self.d2*3)
        
        # encoder
        mu,logvar = self.encoder(x)
        
        # sampling
        latent,mu,logvar = self.reparameterization(mu,logvar)
        
        # decoder
        out = self.decoder(latent).view(-1,3,self.d1,self.d2)

        return out, mu, logvar
    

