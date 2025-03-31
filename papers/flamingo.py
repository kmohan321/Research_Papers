import torch
import torch.nn as nn

class GatedAttention(nn.Module):
  def __init__(self,hidden_dim,heads):
    super().__init__()
    
    #learnable parameters for gated tanh
    self.alpha_cross = nn.Parameter(torch.tensor(0.0))
    self.alpha_ffw = nn.Parameter(torch.tensor(0.0))
    
    self.cross = nn.MultiheadAttention(embed_dim= hidden_dim,num_heads=heads,
                                       batch_first=True)
    self.mha = nn.MultiheadAttention(embed_dim= hidden_dim,num_heads=heads,
                                       batch_first=True)
    
    self.cross_ffw = nn.Sequential(
      nn.Linear(hidden_dim,4*hidden_dim),
      nn.GELU(),
      nn.Linear(4*hidden_dim,hidden_dim)
    )
    self.mha_ffw = nn.Sequential(
      nn.Linear(hidden_dim,4*hidden_dim),
      nn.GELU(),
      nn.Linear(4*hidden_dim,hidden_dim)
    )
  def forward(self,x,y):
    
    # y -> text embeddings (from language model) (b,s,d)
    # x -> images embeddings (from perceiver model) (b,s,d)
    cross_out,_ = self.cross(y,x,x)
    y = y + torch.tanh(self.alpha_cross) * cross_out   # gated for taking relevant info from cross attention 
    y = y + torch.tanh(self.alpha_ffw) * self.cross_ffw(y) # gated for taking relevant info from feed forward net
    
    attn_out,_ = self.mha(y,y,y)
    y = y + attn_out
    y = y + self.mha_ffw(y)
    return y 
