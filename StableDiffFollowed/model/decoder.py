import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


'''
Class preforms simple attention mechanism with groupnorm and residual
connection,
every pixel attends to every other! Pixels know where other pixels
are, and can communicate with pixels to share relative values.

1. Projection from pixel value to what it should look for and store
2. Communication with other pixels, and adding-up relevant info!

'''
# attention block, uses attention with 1 head
# does number of channels necessary
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        
        # group normalization
        # Shape B,C,H,W --> normalize in part of both
        # 
        self.groupnorm = nn.GroupNorm(32,channels)
        
        # self attention K,Q,V
        # 1 head, num channels == embedding dim
        self.attention = SelfAttention(1,channels)
        
    def forward(self, x:torch.Tensor):
        # B,C,H,W = x
        residue = x
        
        # self-attention operation for speed
        n,c,h,w = x.shape
        x = x.view(n,c,h*w) # flattening
        
        # (B,h*w,C)
        # need it like this, 
        # each 'pixel' has channels, that are
        # added acccording to K
        x = x.transpose(-2,-1)
        
        # preforming full attention operation
        # channels are embeddings
        x = self.attention(x)
        
        # restoring shaoes
        x = x.transpose(-1,-2)
        x = x.view(n,c,h,w)
        
        x += residue # residual after attention!
        return x
            


'''
Simple residual and convolutional connections,
may change output channels.
'''
# building VAE residual block
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        
        # norm and convolution
        # B,C,H,W --> B,out_c,H,W
        self.groupnorm_1 = nn.GroupNorm(32,in_channels)
        self.conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        
        # norm and convolution 2
        self.groupnorm_2 = nn.GroupNorm(32,in_channels)
        self.conv_2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        
        
        # residual layers/skip connections
        # just apply to x, change shape if needed
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # B,C,H,W --> B,out_c,H,W
            # basically just summing up parts... (still resiudal)
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
            
            
    def forward(self,x: torch.Tensor):
        h = self.conv_1(x)
        h = F.silu(h)
        h = self.conv_1(h)
        
        # adding residual layer
        x = self.conv_2(F.silu(self.groupnorm_2(h))) + self.residual_layer(x)
        
        return x

'''
Coding our decoder, which should simply be expressive,
while being able to reverse the shapes of our encoder.
Decoder has hard task of inferring meaning from latent-vector.

Strategy of decoder, is to create a lot of channels
from our B,4,H,W vector, and slowly distill down these channels.
We basically allow complex patterns to be slowly learned, by
first growing up our image, and then scaling it down.

More research necessary on why this works.
'''
# sequential class that simply grows-up the dimensions
# should be basically exact opposite of encoder
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        # input is B,4,H/8,W/8
        super().__init__(
            # does nothing convolution
            nn.Conv2d(4,4,kernel_size=1,padding=0),
            
            # B,4,H/8,W/8 --> B,512,H/8,W/8
            nn.Conv2d(4,512,kernel_size=3,padding=1),
            
            VAE_ResidualBlock(512,512),
            # give channels, determines h/w
            VAE_AttentionBlock(512),
            
            # stacking residual blocks
            # B,512,H/8,W/8 --> B,512,H/8,W/8
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            # increasing spatial-size
            # B,512,H/8,W/8 --> B,512,H/4,W/4
            nn.Upsample(scale_factor=2),
            
            # just learning about pixels....
            # UNDERSTAND WHY THIS IS EXPRESSIVE!
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            # increasing spatial-size
            # B,512,H/4,W/4 --> B,512,H/2,W/2
            nn.Upsample(scale_factor=2),
            
            # more stacking...
            # scaling down the number of channels
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            
            # B,256,H/2,W/2 --> B,512,H,W
            nn.Upsample(scale_factor=2),
            
            # more stacking...
            # scaling down the number of channels
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            
            # grouping features in 32, with 128 channels
            # 4 groups of 32
            nn.GroupNorm(32,128),
            nn.SiLU(),
            
            # final convultion for image
            # B,128,H,W --> B,3,H,W (New image)!
            nn.Conv2d(128,3,kernel_size=3,padding=1)
            
        )
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # B,4,H/8,W/8 -> B,3,H,W
        
        # we re-scale our result into what
        # the VAE is used to working with (original scale)
        x /= 0.18215 # we re-scale our input

        for module in self:
            x = module(x)

        # B,3,H,W
        return x