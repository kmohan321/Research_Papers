import torch
import torch.nn as nn

class MinGRU(nn.Module):
  def __init__(self,
               input_dim:int,
               hidden_dim:int,
               ):
      super().__init__()
      
      self.hidden_dim = hidden_dim
      self.w_h = nn.Linear(input_dim,hidden_dim) # weights for h_tilde 
      self.w_z = nn.Linear(input_dim,hidden_dim) # weights for z
   
  #function for sequential implementation
  def forward_sequential(self, x_t, h_prev):
    
      #x_t: shape(b,1,input_size)
      #h_prev: shape(b,1,hidden_size)
      
      #(b,1,input_dim) -> (b,1,hidden_dim)
      z_t = torch.sigmoid(self.w_z(x_t))
      h_tilde = self.w_h(x_t)
      h_t = (1 - z_t) * h_prev + z_t * h_tilde
      return h_t
    
  def forward_parallel(self, x_t : torch.Tensor, ht : torch.Tensor = None):
      
      z_t = torch.sigmoid(self.w_z(x_t))
      h_tilde = self.w_h(x_t)
      
      h0 = ht[:,0].unsqueeze(1)
      at = 1 - z_t 
      bt = z_t * h_tilde #shape ->(b,s,d) * (b,s,d)
      
      #parallel scan for h_t = at * h_t-1 + bt, at = (1-zt) and bt = zt * ht
      #this recurrence relation is defined as -> at_star * (h0 + bt_star)
      at_star = torch.cumprod(at , dim=1)
      bt_star = torch.cumsum(bt/at_star, dim=1)
      
      ht = at_star * (h0 + bt_star)
      return torch.cat([h0,ht],dim=1)
      
    
  def forward(self, x : torch.Tensor, h0 : torch.Tensor = None, parallel = False):
    
      #x: shape(b,s,input_dim)
      #h0: shape(b,1,hidden_dim)
      
      b,s,input_dim = x.shape
      
      if h0 is None:
        h0 = torch.zeros(size=(b,1,self.hidden_dim),device=x.device)

      h_t = torch.zeros(size=(b,s,self.hidden_dim),device=x.device)
      h_t = torch.cat([h0 , h_t], dim=1)
      
      if not parallel:
        for s_idx in range(s):
          h_t[:,s_idx + 1]  = self.forward_sequential(x[:,s_idx],h_t[:,s_idx])
      else:
        h_t  = self.forward_parallel(x,h_t)
        
      return h_t[:,1:] #returning only actual states 


