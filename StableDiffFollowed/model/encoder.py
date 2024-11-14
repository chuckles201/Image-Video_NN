import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


# steadily increasing number of features, decreasing image
class VAE_Encoder(nn.module):
    def __init__(self):
        super().__init__(
            # convolutions, residual convs, attention, norms

            # (B,C=3,H=512,W=512) --> (B,128,H,W)
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            
            # residual, convolution and normalization
            #(B,128,H,W) --> #(B,128,H,W)
            VAE_ResidualBlock(128,128),
            
            #(B,128,H,W) --> #(B,128,H,W)
            VAE_ResidualBlock(128,128),
            
            #(B,128,H,W) --> #(B,128,H / 2,W / 2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            
            
            #(B,128,H/2,W/2) --> #(B,256,H/2,W/2)
            VAE_ResidualBlock(128,256),
            #(B,256,H/2,W/2) --> #(B,256,H/2,W/2)
            VAE_ResidualBlock(256,256),  
            
            #(B,256,H/2,W/2) --> #(B,256,H / 4,W / 4)
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0), 
            
            #(B,256,H / 4,W / 4) --> #(B,512,H / 4,W / 4)
            VAE_ResidualBlock(256,512),
            VAE_ResidualBlock(512,512), # just think of as convolution
            
            #(B,512,H / 4,W / 4) --> #(B,512,H / 8,W / 8)
            # Now we have 512 channels h/8, w/8
            # 512 x 512/8 x 512/8 feature maps
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0), 
            
            
            # final residual blocks!
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            # self-attention!
            # B, C, H/8, W/8
            VAE_AttentionBlock(512),
            
            # what are these coming from?
            VAE_ResidualBlock(512,512),
            
            # normalization
            nn.GroupNorm(32,512),
            # just RELU that empircally works better
            nn.SiLU(),
            
            # changes number of feature maps
            # this is our bottleneck
            # (B,512,H / 8,W/8) --> (B,512,H/8,W/8) 
            nn.Conv2d(512,8,kernel_size=3,padding=1),
            
            # just scanning over image
            nn.Conv2d(8,8,kernel_size=1,padding=0)
        )
    
    def forward(self,x : torch.Tensor,noise: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        # noise: (B, 8, H/8, W/8) (same as output)
        for module in self:
            if getattr(module,'stride',None) == (2,2):
                # padding is only applied to the right and bottom
                # L,R,T,B
                x = F.pad(x,(0,1,0,1)) 
            x = module(x)
            
        
        
        # CHUNK: (B,8,H,W) -> (B,4,H,W) x 2
        mean, log_var = torch.chunk(x,2,dim=1)
        
        # making sure doesn't explode
        log_var = torch.clamp(log_var,-30,20)
        
        variance = log_var.exp() # getting true variance
        std_dev = variance.sqrt()
        
        # Sampling from gaussian, we receive noise as arg.
        sample = noise*std_dev + mean
        
        # B, 4, H/8,W/8 <==> Final shape
        sample *= 0.18215
        # this scaling was to make sure that the variance was 1
        # this was acheived expirimentally by averaging output variance
        # from autoencoder
        return sample # B,