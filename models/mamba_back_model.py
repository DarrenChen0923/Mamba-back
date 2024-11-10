import torch.nn as nn
from mamba_ssm import Mamba
from .layers import FFWRelativeSelfAttentionModule


class MambaModel(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, mlp_hidden_dim=32, output_dim=1):
        super(MambaModel, self).__init__()
        self.mlp1 = nn.Linear(1,d_model)
        self.self_atten = FFWRelativeSelfAttentionModule(d_model,2,2)
        self.mamba = Mamba(
            d_model=d_model, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        self.mlp2= nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, output_dim)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(9,9),
            nn.ReLU(),
            nn.Linear(9,1)
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = self.self_atten(x.transpose(0,1), diff_ts=None,
                query_pos=None, context=None, context_pos=None,pad_mask=None)[-1].transpose(0,1)
        x = self.mamba(x)
        output = self.mlp2(x)
        output = output.view(x.shape[0],-1)
        output = self.mlp3(output)
        return output