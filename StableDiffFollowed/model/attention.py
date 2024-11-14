import torch
from torch import nn
from torch.nn import functional as F 
import math


# implementing self-attention
'''
Takes in number of channels
'''
class SelfAttention(nn.Module):
    def __init__(self,n_heads : int,d_embd: int,in_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embd,3*d_embd,bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embd,d_embd)
        self.soft = nn.Softmax()
        
        # shapes for working with
        self.n_heads = n_heads
        self.d_embd = d_embd
        self.d_head = n_heads // d_embd
        assert(n_heads % d_embd == 0)
        
        
    # no mask used
    def forward(self,x,mask=False):
        # input is: B,C,D_embd=H*W
        # C = Sequence length
        
        # create B,C,Emb*heads
        k,q,v = self.in_proj(x).chunk(3,dim=-1) # splitting
        
        # change dimensionality
        batch_size,sequence_length,d_embd = x.shape
        
        # splitting into heads
        # transposing so we can compute with heads as batch
        # --> (B,Heads,Seq,Emb)
        q = q.view(batch_size,sequence_length,self.n_heads,self.d_head).transpose(2,1)
        k = k.view(batch_size,sequence_length,self.n_heads,self.d_head).transpose(2,1)
        v = v.view(batch_size,sequence_length,self.n_heads,self.d_head).transpose(2,1)
        
        # preform along same dimension
        w = k @ q.T 
        w /= math.sqrt(w.shape[-1]) # normalizing by number vals
        
        # mask if needed (not in our case)
        if mask:
            mask = torch.ones_like(w,dtype=torch.bool).triu()
            mask = torch.masked_fill(w,mask,0)
            soft = self.soft(mask,dim=-1)
        else:
            soft = self.soft(w,dim=-1)
            
        # normalizing weights (auto-done in linear layers usually)
           
        # final computation, and viewing as in original form 
        # B, H, C, C @ B, H, C, D --> B,H,C,D --> B,C,D*H
        out = soft @ v
        out = out.transpose(2,1).view(batch_size,sequence_length,d_embd)
        # finally done! 
        return  self.out_proj(out)
        
        
        