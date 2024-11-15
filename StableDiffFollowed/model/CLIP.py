import torch
from torch import nn
import torch.nn.functional as F 
from attention import SelfAttention



'''
Embeddings for our clip model.
Simply take the embeddings for the text, and 
add a positional embedding.

arbitrary embedding for a given vocab and embd size

'''
class CLIPEmbedding(nn.Module):
    
    def __init__(self,n_vocab:int,n_embd:int,n_tokens:int):
        super().__init__()
        
        # for each vocabulary item, a token
        self.token_embedding = nn.Embedding(n_vocab,n_embd)
        
        # position embeddding: learned start at zero
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens,n_embd))
        
    def forward(self, tokens):
        # (B,seq) - > (B,seq,Dim)
        x = self.token_embedding(tokens)
        
        # adding in position embeddings
        # adds in the information
        x += self.position_embedding
        
        return x

'''
CLIP LAYER

- These are just simply our transformer blocks,
that we will feed our tokem embeddings into in order to
generate embeddings for our model.
- This is simply the encoder for the text that is alligned with 
the encoder for the images.


'''
class CLIPLayer(nn.Module):
    def __init__(self,n_head: int, n_embd: int):
        super().__init__()
        
        # layernorm and attention
        self.layernorm_1 = nn.LayerNorm()
        self.attention = SelfAttention(n_head,n_embd)
        self.layernorm_2 = nn.LayerNorm()
        self.linear_1 = nn.Linear(n_embd,4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd,n_embd)
        
    def forward(self,x):
        # B, Seq, Dim - > same
        residue = x
        
        # self attention and norm
        # causal mask is on because we want to
        # train the model to learn text
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue
        
        # feed-forward w/ residual connection
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        # non-linearity --> special QuickGeLU!
        # in practice, this works better.
        x = x * torch.sigmoid(x*1.702)
        x = self.linear_2(x) + x
        
        return x
        
        
        
        




'''
CLIP:
- Learns a shared-embedding for images and text,
we need a pre-trained moddel for CLIP (encoder and decoder)

- We will only be using the text encoder for our purposes

The text encoder will literally just be an encoder transformer.
'''
class CLIP(nn.Module):
    
    def __init__(self):
        super().__init__()
        # pre-set sizes for our embeddings Vocab,Embd,SeqMax
        self.embedding = CLIPEmbedding(49408,768,77)
        
        self.layers = nn.Module([
            # attention heads, embd
            CLIPLayer(12, 768) for i in range(12)
        ])
        
        self.layernorm = nn.LayerNorm(768)

    # text inputds
    def forward(self,tokens: torch.LongTensor) -> torch.FloatTensor:
        
        # tokens from text/image
        tokens = tokens.type(torch.long)
        
        # B,Seq -> B,Seq,D = 768
        state = self.embedding(tokens)
        
        # all CLIPLayers
        for layer in self.layers:
            state = layer(state)
            
        # output with layernorm
        output = self.layernorm(state)
        
        return output
    