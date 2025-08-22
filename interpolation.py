import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd

class ExtendedMappingA(nn.Module):
    """
    A PyTorch module that extends token representations by applying a modulation factor 
    and appending interpolated tokens to the sequence.

    Attributes:
        seq_len (int): The length of the input sequence.
        p (int): The total desired sequence length after appending interpolated tokens.
        hidden_dim (int): The dimensionality of the hidden representation.
        token_proj (nn.Linear): A linear layer to project input tokens to the hidden dimension.
        modulation (nn.Parameter): A learnable modulation factor applied to each token in the sequence.
        interp_tokens (nn.Parameter): Learnable interpolated tokens to extend the sequence.

    Args:
        input_dim (int): The dimensionality of the input tokens.
        hidden_dim (int): The dimensionality of the hidden representation.
        seq_len (int, optional): The length of the input sequence. Default is 20.
        p (int, optional): The total desired sequence length after appending interpolated tokens. Default is 30.

    Methods:
        forward(x):
            Computes the extended token representation by applying modulation to the input tokens 
            and appending interpolated tokens.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, p, hidden_dim), where the first 
                `seq_len` tokens are modulated projections of the input tokens, and the remaining 
                `p - seq_len` tokens are interpolated tokens.
    """

    def __init__(self, input_dim, hidden_dim, seq_len=20, p=30):
        super().__init__()
        self.seq_len = seq_len
        self.p = p
        self.hidden_dim = hidden_dim
        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.modulation = nn.Parameter(torch.ones(seq_len, hidden_dim))
        extra = p - seq_len
        self.interp_tokens = nn.Parameter(
            torch.linspace(-1.0, 1.0, extra).unsqueeze(1).expand(extra, hidden_dim).clone()
        )

    def forward(self, x):
        batch_size = x.size(0)
        token_repr = self.token_proj(x) # (batch_size, seq_len, hidden_dim)                   
        token_repr = token_repr * self.modulation.unsqueeze(0)  
        interp_repr = self.interp_tokens.unsqueeze(0).expand(batch_size, -1, -1)  
        out = torch.cat([token_repr, interp_repr], dim=1)   

        # A(x) = [v_1*f(x_1), v_2*f(x_2), ..., v_n*f(x_n), \hat{x_n+1}, ..., \hat{x_p}]
        # where v_i is the modulation factor for the i-th token, f(x_i) is the projected token representation,
        # and \hat{x_n+1}, ..., \hat{x_p} are the interpolated tokens.
        return out
    
class ExtendedMappingA_statistical(nn.Module):
    """
    A PyTorch module that extends token representations by applying a modulation factor 
    and appending interpolated tokens to the sequence.

    Attributes:
        seq_len (int): The length of the input sequence.
        p (int): The total desired sequence length after appending interpolated tokens.
        hidden_dim (int): The dimensionality of the hidden representation.
        token_proj (nn.Linear): A linear layer to project input tokens to the hidden dimension.
        modulation (nn.Parameter): A learnable modulation factor applied to each token in the sequence.
        interp_tokens (nn.Parameter): Learnable interpolated tokens to extend the sequence.

    Args:
        input_dim (int): The dimensionality of the input tokens.
        hidden_dim (int): The dimensionality of the hidden representation.
        seq_len (int, optional): The length of the input sequence. Default is 20.
        p (int, optional): The total desired sequence length after appending interpolated tokens. Default is 30.

    Methods:
        forward(x):
            Computes the extended token representation by applying modulation to the input tokens 
            and appending interpolated tokens.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, p, hidden_dim), where the first 
                `seq_len` tokens are modulated projections of the input tokens, and the remaining 
                `p - seq_len` tokens are interpolated tokens.
    """

    def __init__(self, input_dim, prompt_dim, hidden_dim, seq_len=20, p=30):
        super().__init__()
        self.seq_len = seq_len
        self.p = p
        self.hidden_dim = hidden_dim
        self.token_proj = nn.Linear(input_dim + prompt_dim, hidden_dim)
        self.modulation = nn.Parameter(torch.ones(seq_len, hidden_dim))
        extra = p - seq_len
        self.interp_tokens = nn.Parameter(
            torch.linspace(-1.0, 1.0, extra).unsqueeze(1).expand(extra, hidden_dim).clone()
        )

    def forward(self, x):
        batch_size = x.size(0)
        token_repr = self.token_proj(x) # (batch_size, seq_len, hidden_dim)                   
        token_repr = token_repr * self.modulation.unsqueeze(0)  
        interp_repr = self.interp_tokens.unsqueeze(0).expand(batch_size, -1, -1)  
        out = torch.cat([token_repr, interp_repr], dim=1)   

        # A(x) = [v_1*f(x_1), v_2*f(x_2), ..., v_n*f(x_n), \hat{x_n+1}, ..., \hat{x_p}]
        # where v_i is the modulation factor for the i-th token, f(x_i) is the projected token representation,
        # and \hat{x_n+1}, ..., \hat{x_p} are the interpolated tokens.
        return out

## NO USE FOR NOW
class ExtendedMapping_manual(nn.Module):
    """
    Replace interpolated tokens with specified values based on the input tensor and the weights.
    The weights can be wk, wq, or wv, and are used to compute the new values for the interpolated tokens.
    """

    def __init__(self, input_dim, hidden_dim, seq_len=20, p=30):
        super().__init__()
        self.seq_len = seq_len
        self.p = p
        self.hidden_dim = hidden_dim
        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.modulation = nn.Parameter(torch.ones(seq_len, hidden_dim))
        extra = p - seq_len
        self.interp_tokens = nn.Parameter(
            torch.linspace(-1.0, 1.0, extra).unsqueeze(1).expand(extra, hidden_dim).clone()
        )

        self.delta_l = 2 * input_dim / self.p
        self.k_range = torch.arange(0, self.p, dtype=torch.float32).unsqueeze(1) # (p, 1)
        Ls = self.delta_l * self.k_range - input_dim
        self.register_buffer("magic_l", Ls)

    def forward(self, x, w, indx):
        # wk: (batch_size, seq_len, d)
        batch_size = x.size(0)
        token_repr = self.token_proj(x) # (batch_size, seq_len, hidden_dim)                   
        token_repr = token_repr * self.modulation.unsqueeze(0)  
        interp_repr = self.interp_tokens.unsqueeze(0).expand(batch_size, -1, -1)  
        out = torch.cat([token_repr, interp_repr], dim=1)   # (batch_size, p, hidden_dim)

        # A(x) = [v_1*f(x_1), v_2*f(x_2), ..., v_n*f(x_n), \hat{x_n+1}, ..., \hat{x_p}]
        # where v_i is the modulation factor for the i-th token, f(x_i) is the projected token representation,
        # and \hat{x_n+1}, ..., \hat{x_p} are the interpolated tokens.

        new_values = 2 * self.magic_l * w[:, indx, :].unsqueeze(1) # (batch_size, p, d)
        out[:, :, x.size(-1): 2 * x.size(-1)] = new_values

        out[:, :, -1] = (self.magic_l**2).squeeze(1)
        out[:, :, -2] = self.magic_l.squeeze(1)
        return out