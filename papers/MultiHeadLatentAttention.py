import torch
import torch.nn as nn

class Frequencies(nn.Module):
  def __init__(self, max_seq_length,head_dim):
    super().__init__()
    
    assert head_dim%2==0, 'head_dim should be even'
    
    # m values for different positions in sequence
    m = torch.arange(0,max_seq_length)
    
    # theta values for different index in token vector
    theta = 1/(10000**2*torch.arange(0,head_dim//2)/head_dim)
    
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


class MLA(nn.Module):
  
  def __init__(self,
               hidden_dim:int,
               heads:int,
               v_head_dim:int,
               kv_rank:int,
               query_rank:int,
               rope_qk_dim:int,
               ):
               
    super().__init__()
    
    self.heads = heads
    self.hidden_dim = hidden_dim
    self.rope_qk_dim = rope_qk_dim
    self.kv_rank = kv_rank
    self.query_rank = query_rank
    self.v_head_dim = v_head_dim
    
    assert self.hidden_dim % self.heads == 0 ,"hidden_dim must be divisible by heads"
    self.head_dim = self.hidden_dim // self.heads 
    self.qk_head_dim = self.head_dim + self.rope_qk_dim # head_dim for qk
    
    # down and up projection matrix (query)
    self.wq_d = nn.Linear(self.hidden_dim,self.query_rank,bias=False)
    self.q_norm = nn.RMSNorm(self.query_rank) 
    self.wq_u = nn.Linear(self.query_rank, self.heads * self.qk_head_dim,bias=False)
    
    # down and up projection matrix (key_value)
    self.wkv_d = nn.Linear(self.hidden_dim, (self.kv_rank + self.rope_qk_dim) ,bias=False)
    self.kv_norm = nn.RMSNorm(self.kv_rank)
    self.wkv_u = nn.Linear(self.kv_rank, self.heads * (self.head_dim + self.v_head_dim) ,bias=False)
    
    #output_linear_layer
    self.wo = nn.Linear(self.heads * self.v_head_dim ,self.hidden_dim,bias=False)
    self.scale = self.qk_head_dim ** -0.5
    
  def forward(self, x: torch.Tensor, rope_freq: torch.Tensor, mask: torch.Tensor = None):
    b,s,d = x.shape
    # (b,s,d) -> (b,s, n_h * qk_dim)
    q = self.wq_u(self.q_norm(self.wq_d(x)))
    q = q.view(-1,s,self.heads,self.qk_head_dim)
    
    #(b,s, n_h * qk_dim) -> (b,s,n_h,d_h) , (b,s,n_h, d_r) 
    q , q_rope = torch.split(q, [self.head_dim,self.rope_qk_dim],dim=-1)
    q_rope = rope(q_rope,rope_freq)
    q = torch.cat([q , q_rope],dim=-1)
    
    kv_c = self.wkv_d(x)
    
    #(b,s, n_h * qk_dim) -> (b,s,n_h * d_h) , (b,s,n_h * d_r) 
    kv_c, k_rope = torch.split(kv_c,[self.kv_rank,self.rope_qk_dim],dim=-1)
    k_rope = rope(k_rope.unsqueeze(2),rope_freq)
    k_rope = k_rope.expand(-1,-1,self.heads,-1)
    
    kv = self.wkv_u(self.kv_norm(kv_c))
    kv = kv.view(-1,s,self.heads,(self.head_dim + self.v_head_dim))
    k ,v = torch.split(kv,[self.head_dim , self.v_head_dim],dim=-1)
    
    # (b, s ,n_h, qk_dim)
    k = torch.cat([k , k_rope], dim=-1) 
    
    # attention mechanism
    q,k,v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    attention_scores = q @ k.transpose(2,3) * self.scale
    # causal_masking
    if mask is not None :
        attention_scores = attention_scores.masked_fill(mask==0,-torch.inf)
        
    attention_weights = torch.softmax(attention_scores,dim=-1)
    out = (attention_weights @ v).transpose(1,2)
    out  = out.contiguous().view(-1,s,self.heads * self.v_head_dim)
    return self.wo(out)
    