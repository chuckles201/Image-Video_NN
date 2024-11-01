import torch
from torch import nn
import torch.nn.functional as F


# input --> hidden --> mean,std --> sample --> decode/output
class VAE(nn.Module):
    def __init__(self,input_dim,latent_dim=20,h_dim=200):
        super().__init__()
        self.input_dim = input_dim
        
        # encoder
        self.img_2hid = nn.Linear(input_dim,h_dim)
        self.hid2mu = nn.Linear(h_dim,latent_dim)
        self.hid2sigma = nn.Linear(h_dim,latent_dim)
        
        self.lin_extra = nn.Linear(input_dim,input_dim)
            
        # decoder (reverse of encoder) 
        self.z_2hid = nn.Linear(latent_dim,h_dim)
        self.hid_2img = nn.Linear(h_dim,input_dim)
        
        self.relu = nn.ReLU()
        
    def encode(self,x):
        x = x.view(-1,self.input_dim)
        # takes in flat image
        h = self.relu(self.img_2hid(x))
        mu = self.hid2mu(h)
        sigma = self.hid2sigma(h)
        
        return mu,sigma # sigma is std
        
        
    def decode(self,x):
        h = self.relu(self.z_2hid(x)) # potential problems?
        h =self.hid_2img(h)
        h = self.lin_extra(h)
        return torch.sigmoid(h) # bet 0,1
    
    def forward(self,x):
        mu, sigma = self.encode(x)
        eps = torch.randn_like(mu)
        z_reparam = eps*sigma + mu
        
        x_recon = self.decode(z_reparam)
       
        
        return x_recon, mu, sigma
    
    

model = VAE(28*28)
tst = torch.randn(4,28,28)
x,mu,sigma = model(tst)
print(x.shape,mu.shape,sigma.shape)