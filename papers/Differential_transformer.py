import torch
import torch.nn as nn


class Differntial_Attention(nn.Module):
  def __init__(self, hidden_dims,heads,lambda_init=0.8):
    super().__init__()
    
    self.heads = heads
    self.lambda_init = lambda_init
    assert hidden_dims%heads==0,'hidden_dims must be divisible by num_heads'
    
    self.head_dim = hidden_dims//self.heads
    
    self.wq = nn.Linear(hidden_dims,self.head_dim*self.heads,bias=False)
    self.wk = nn.Linear(hidden_dims,self.head_dim*self.heads,bias=False)
    self.wv = nn.Linear(hidden_dims,self.head_dim*self.heads,bias=False)
    self.wo = nn.Linear(hidden_dims,hidden_dims,bias=False)
    
    # RMSNORM for stablizing gradients, independently for each head
    self.RMSModules = nn.ModuleList([
      nn.RMSNorm(self.head_dim,elementwise_affine=False) for _ in range(self.heads)
    ])
    
    # learnable lambdas 
    self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim))
    self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim))
    self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim))
    self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim))
    
    
  def forward(self,x):
    
    b,s,d = x.shape
    
    query = self.wq(x).view(b,s,self.heads,self.head_dim)    
    key = self.wk(x).view(b,s,self.heads,self.head_dim)  
    value = self.wv(x).view(b,s,self.heads,self.head_dim)
    
    query = query.transpose(1,2) 
    key = key.transpose(1,2) 
    value = value.transpose(1,2)
    
    # splitting query and keys
    q1 , q2 = torch.chunk(query,2,dim=-1)
    k1 , k2 = torch.chunk(key,2,dim=-1)
    
    atten_score1 , atten_score2 = q1 @ k1.transpose(2,3) , q2 @ k2.transpose(2,3)
    atten_weight1 , atten_weight2 = torch.softmax((atten_score1/self.head_dim**0.5),dim=-1),\
          torch.softmax((atten_score2/self.head_dim**0.5),dim=-1)

    lambda_learnable = torch.exp(torch.dot(self.lambda_q1,self.lambda_k1)) - torch.exp(torch.dot(self.lambda_q2,self.lambda_k2))\
      + self.lambda_init
    
    attention_weights = atten_weight1 - lambda_learnable * atten_weight2
    
    out = attention_weights @ value
    
    # normalizing each heads
    self.norms_heads = []
    for i,norm in enumerate(self.RMSModules):
      self.norms_heads.append(norm(out[:,i]) * (1 - self.lambda_init)) 
      
    out = torch.cat(self.norms_heads,dim=1)
  
    out = out.transpose(1,2)
    outputs = out.contiguous().view(b,s,d)
    return self.wo(outputs)
    
      
  
x = torch.randn(size=(2,8,256))
atten = Differntial_Attention(256,4)
print(atten(x).shape)
    
    
        
    
    
    
    
    