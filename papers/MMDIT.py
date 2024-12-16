import torch
import torch.nn as nn

class Joint_Attention(nn.Module):
  def __init__(self,heads,hidden_dim,qk_norm=False):
    super().__init__()
    
    assert hidden_dim%heads == 0, 'hidden_dim must be divisible by heads'
    
    self.qk_norm = qk_norm
    self.heads = heads
    self.head_dim = hidden_dim//self.heads
    
    # K Q V for labels
    self.label_q = nn.Linear(hidden_dim,self.head_dim*self.heads,bias=False)
    self.label_k = nn.Linear(hidden_dim,self.head_dim*self.heads,bias=False)
    self.label_v = nn.Linear(hidden_dim,self.head_dim*self.heads,bias=False)
    self.label_out = nn.Linear(hidden_dim,hidden_dim,bias=False)
    
    # K Q V for patches
    self.patch_q = nn.Linear(hidden_dim,self.head_dim*self.heads,bias=False)
    self.patch_k = nn.Linear(hidden_dim,self.head_dim*self.heads,bias=False)
    self.patch_v = nn.Linear(hidden_dim,self.head_dim*self.heads,bias=False)
    self.patch_out = nn.Linear(hidden_dim,hidden_dim,bias=False)
    
    self.rms_q_c = nn.RMSNorm(hidden_dim)
    self.rms_k_c = nn.RMSNorm(hidden_dim)
    
    self.rms_q_x = nn.RMSNorm(hidden_dim)
    self.rms_k_x = nn.RMSNorm(hidden_dim)
    
    self.scale = self.head_dim ** -0.5
  def forward(self,labels_embed,patches_embed):
    
    b,s_label,d = labels_embed.shape
    b,s_patch,d = patches_embed.shape
    
    c, x  = labels_embed , patches_embed
    
    q_c,k_c,v_c =  self.label_q(c), self.label_k(c),self.label_v(c)
    q_x,k_x,v_x =  self.patch_q(x), self.patch_k(x),self.patch_v(x)
    
    if self.qk_norm:
      q_c,k_c = self.rms_q_c(q_c), self.rms_k_c(k_c)
      q_x,k_x = self.rms_q_x(q_x), self.rms_k_x(k_x)
     
     # Concatenation of Q K V over sequence length
    query = torch.cat([q_c,q_x],dim=1)
    key = torch.cat([k_c,k_x],dim=1)
    value = torch.cat([v_c,v_x],dim=1)
    
    query = query.view(b,-1,self.heads,self.head_dim).transpose(1,2)
    key = key.view(b,-1,self.heads,self.head_dim).transpose(1,2)
    value = value.view(b,-1,self.heads,self.head_dim).transpose(1,2)
    
    attention_score = torch.softmax((query @ key.transpose(2,3)) * self.scale ,dim=-1)
    out = (attention_score @ value).transpose(1,2)
    out = out.contiguous().view(b,-1,self.heads*self.head_dim)
    c , x = torch.split(out,[s_label,s_patch],dim=1)
    
    return self.label_out(c) , self.patch_out(x) 
  
class MM_DiT(nn.Module):
  def __init__(self, hidden_dim,heads,qk_norm=False ):
    super().__init__()
    
    self.label_affines = nn.Sequential(
      nn.SiLU(),
      nn.Linear(hidden_dim,6*hidden_dim)
    )
    
    self.patch_affines = nn.Sequential(
      nn.SiLU(),
      nn.Linear(hidden_dim, 6*hidden_dim)
    )
    
    self.norm_c = nn.LayerNorm(hidden_dim,elementwise_affine=False,eps=1e-6)
    self.norm_x = nn.LayerNorm(hidden_dim,elementwise_affine=False,eps=1e-6)
    
    self.norm_c_mlp = nn.LayerNorm(hidden_dim,elementwise_affine=False,eps=1e-6)
    self.norm_x_mlp = nn.LayerNorm(hidden_dim,elementwise_affine=False,eps=1e-6)
    
    self.joint_atten = Joint_Attention(heads,hidden_dim,qk_norm=qk_norm)

    self.mlp_label = nn.Sequential(
      nn.Linear(hidden_dim,4*hidden_dim),
      nn.GELU(approximate='tanh'),
      nn.Linear(4*hidden_dim,hidden_dim)
    )
    
    self.mlp_patch = nn.Sequential(
      nn.Linear(hidden_dim,4*hidden_dim),
      nn.GELU(approximate='tanh'),
      nn.Linear(4*hidden_dim,hidden_dim)
    )
    
  def forward(self,timestep_embed,label_embed,patch_embed):
    
    timestep_embed = timestep_embed.unsqueeze(dim=1)
    
    alpha_c , beta_c, gamma_c,scale_c_mlp, shift_c_mlp, gate_c_mlp = torch.chunk(self.label_affines(timestep_embed),6,dim=-1) 
    alpha_x , beta_x, gamma_x, scale_x_mlp, shift_x_mlp, gate_x_mlp = torch.chunk(self.label_affines(timestep_embed),6,dim=-1)
    
    label_embed,patch_embed = self.norm_c(label_embed), self.norm_x(patch_embed)
    c,x = alpha_c * label_embed + beta_c , alpha_x * patch_embed + beta_x 
    
    c,x = self.joint_atten(c,x)
    c,x = gamma_c * c + label_embed , gamma_x * x + patch_embed
    
    c,x = self.norm_c_mlp(c), self.norm_x_mlp(x)
    c,x = scale_c_mlp * c + shift_c_mlp , scale_x_mlp * x + shift_x_mlp
    
    c,x = self.mlp_label(c) , self.mlp_patch(x)
    c,x = gate_c_mlp * c + label_embed , gate_x_mlp * x + patch_embed
    return c,x 

    
    
    
    
