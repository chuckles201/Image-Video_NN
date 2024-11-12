import torch
from torch import nn
import torch.nn.functional as F
import VAE

# building our training loop with loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loss function: KL and reconstr.
def loss_function(target,output,mu,logvar):
    mse = F.mse_loss(output,target)
    
    # sigma and mu added across single sample, take mean acr batch
    # B,latent
    KL = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=-1),dim=0)
    
    return KL + mse
    
# getting a batch
def get_batch(data,size,d1,d2):
    # return size,3,h*w batches of data
    # data is list of tensors
    ind = torch.randint(0,len(data),size=(size,),device=device).tolist()
    batch = torch.stack([data[i] for i in ind]).to(device=device)
    return batch.view(size,3,d1,d2).to(device)
    

def train_VAE(d1,d2,latent,epochs,batch_size,data):
    # initializing model
    model = VAE.VariationalAutoEncoder(d1,d2,latent)
    model.to(device)
    
    # passing in parameters to optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
    
    model.train()
    # training loop
    for i in range(epochs):
        x = get_batch(data,batch_size,d1,d2)
        x_reconstructed, mu, logvar = model(x)
        
        
        # going backward
        optimizer.zero_grad()
        loss = loss_function(x,x_reconstructed,mu,logvar)
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Epoch {i}, loss: {loss.item()}")
            
    return model
        