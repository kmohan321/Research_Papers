from sympy import false
import torch
import torch.nn as nn


class Frequencies(nn.Module):
  def __init__(self, max_seq_length,head_dim):
    super().__init__()
    
    assert head_dim%2==0, 'head_dim should be even'
    
    # m values for different positions in sequence
    m = torch.arange(0,max_seq_length)
    
    # theta values for different index in token vector
    theta = 1/(10000**(2*torch.arange(0,head_dim//2))/head_dim)
    
    #all possible combinations for m and theta
    freq = torch.outer(m,theta)
    
    #converting freq to polar
    complex_freq = torch.polar(torch.ones_like(freq),freq)
    
    self.register_buffer('complex_freq', complex_freq.unsqueeze(0).unsqueeze(2))
    
  def forward(self):
    return self.complex_freq
  

def rope(x,complex_freq):
    b ,s, h, d = x.shape
    x = x.view(b, s, h, -1, 2)
    x = torch.view_as_complex(x)
 
    x = x * complex_freq[:,s,:,:]
    x = torch.view_as_real(x)
    x = x.view(b,s,h,d)
    return x


class TPA(nn.Module):
  def __init__(self, 
               hidden_dim: int,
               rank_query: int,
               rank_key: int,
               rank_value: int,
               heads: int,
               head_dim: int,
               ):
    super().__init__()
    
    self.hidden_dim = hidden_dim
    self.heads = heads
    self.head_dim = head_dim
    self.rank_key = rank_key
    self.rank_query = rank_query
    self.rank_value = rank_value
    
    self.W_Aq = nn.Linear(hidden_dim, rank_query * heads,bias=False)
    self.W_Ak = nn.Linear(hidden_dim, rank_key * heads,bias=False)
    self.W_Av = nn.Linear(hidden_dim, rank_value * heads,bias=False)
    
    self.W_Bq = nn.Linear(hidden_dim,rank_query * head_dim,bias=False)
    self.W_Bk = nn.Linear(hidden_dim,rank_key * head_dim,bias=False)
    self.W_Bv = nn.Linear(hidden_dim,rank_value * head_dim,bias=False)
    
    self.Wo = nn.Linear(hidden_dim,hidden_dim,bias=False)
    
    self.scale = self.head_dim ** -0.5
    self.query_scale = rank_query ** -1
    self.key_scale = rank_key ** -1
    self.value_scale = rank_value ** -1
    
  def forward(self, x, complex_freq, mask = None):
    b,s,d = x.shape
    
    # (b,s,d) -> (b,s,r*h)
    Aq,Ak,Av = self.W_Aq(x),self.W_Ak(x),self.W_Av(x)
    #(b,s,d) -> (b,s,r*h_d)
    Bq,Bk,Bv = self.W_Bq(x),self.W_Bk(x),self.W_Bv(x)
    
    #(b,s,r*h) -> (b,s,r,h), (b,s,r*h_d) -> (b,s,r,h_d)
    Aq,Bq = Aq.view(-1,s,self.rank_query,self.heads),Bq.view(-1,s,self.rank_query,self.head_dim)
    Ak,Bk = Ak.view(-1,s,self.rank_key,self.heads),Bk.view(-1,s,self.rank_key,self.head_dim)
    Av,Bv = Av.view(-1,s,self.rank_value,self.heads),Bv.view(-1,s,self.rank_value,self.head_dim)
    
    Bq,Bk = rope(Bq,complex_freq), rope(Bk,complex_freq)
    
    #(b,s,h,r) @ (b,s,r,h_d) -> (b,s,h,h_d)
    Q = (Aq.transpose(2,3) @ Bq) * self.query_scale
    K = (Ak.transpose(2,3) @ Bk) * self.key_scale
    V = (Av.transpose(2,3) @ Bv) * self.value_scale
    
    # applying the attention
    #(b,s,h,h_d) -> (b,h,s,h_d)
    Q,K,V = Q.transpose(1,2),K.transpose(1,2),V.transpose(1,2)
    
    attention_scores = (Q @ K.transpose(2,3)) * self.scale
    if mask:
      attention_scores = attention_scores.masked_fill(mask==0,-torch.inf)
    attention_weights = torch.softmax(attention_scores,dim=-1)
    
    #(b,h,s,h_d) -> (b,s,h,h_d)
    out = (attention_weights @ V).transpose(1,2)
    out = out.contiguous().view(-1,s,self.hidden_dim)
    return self.Wo(out)
  


    
    
    
    
    
    
    
    
    
    