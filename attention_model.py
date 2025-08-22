# Interpolation
from interpolation import ExtendedMappingA, ExtendedMapping_manual, ExtendedMappingA_statistical

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class MultiHeadAttentionLayer_attn(nn.Module):
    """
    A PyTorch implementation of a Multi-Head Attention Layer.

    This layer performs multi-head attention, which is a key component of the 
    Transformer architecture. It projects the input into query, key, and value 
    spaces, computes scaled dot-product attention for each head, and combines 
    the results.

    Attributes:
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the hidden representation.
        head_dim (int): Dimension of each attention head (hidden_dim // num_heads).
        q_proj (nn.Linear): Linear layer to project input into query space.
        k_proj (nn.Linear): Linear layer to project input into key space.
        v_proj (nn.Linear): Linear layer to project input into value space.
        out_proj (nn.Linear): Linear layer to project the concatenated output 
                              of all heads into the output space.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden representation.
        output_dim (int): Dimension of the output features.
        num_heads (int): Number of attention heads. Must divide hidden_dim evenly.

    Methods:
        forward(x, return_weights=False):
            Computes the multi-head attention output for the input tensor.

            Args:
                x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
                return_weights (bool, optional): If True, returns the attention weights 
                                                 along with the output. Defaults to False.

            Returns:
                torch.Tensor: Output tensor of shape [batch_size, seq_len, output_dim].
                torch.Tensor (optional): Attention weights of shape 
                                         [batch_size, num_heads, seq_len, seq_len] 
                                         if return_weights is True.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_weights=False):
        batch_size, seq_len, _ = x.size() # shape [batch_size, p, hidden_dim]
        Q = self.q_proj(x) # shape [batch_size, p, hidden_dim]
        K = self.k_proj(x) # shape [batch_size, p, hidden_dim]
        V = self.v_proj(x) 
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) 
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(scores, dim=-1) # 
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.out_proj(attn_output) # shape [batch_size, p, output_dim]
        if return_weights:
            return attn_output, attn_weights
        return attn_output

class onelayer_MultiLayerAttentionModel_attn(nn.Module):
    """
    A PyTorch module implementing a one-layer multi-head attention model with an extended mapping layer.

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden layer.
        output_dim (int): The dimensionality of the output features.
        seq_len (int, optional): The sequence length of the input. Defaults to 20.
        p (int, optional): The dimensionality of the extended mapping layer. Defaults to 30.
        num_heads (int, optional): The number of attention heads in the multi-head attention layer. Defaults to 1.

    Attributes:
        seq_len (int): The sequence length of the input.
        p (int): The dimensionality of the extended mapping layer.
        mapping_A (ExtendedMappingA): A module that maps the input to an extended representation.
        attn1 (MultiHeadAttentionLayer_attn): A multi-head attention layer.

    Methods:
        forward(x, return_weights=False):
            Performs a forward pass through the model.

            Args:
                x (torch.Tensor): The input tensor of shape [batch_size, seq_len, input_dim].
                return_weights (bool, optional): If True, returns the attention weights. Defaults to False.

            Returns:
                torch.Tensor: The output tensor of shape [batch_size, seq_len, output_dim].
                list (optional): A list containing the attention weights if `return_weights` is True.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, seq_len=20, p=30, num_heads=1):

        super().__init__()
        self.seq_len = seq_len
        self.p = p

        self.mapping_A = ExtendedMappingA(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len, p=p) # [batch_size, p, hidden_dim]
        self.attn1 = MultiHeadAttentionLayer_attn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads) # [batch_size, p, output_dim]

    def forward(self, x, return_weights=False):
        x_extended = self.mapping_A(x)  
        x_attn1, attn1_weights = self.attn1(x_extended, return_weights=True)
        x_out = x_attn1[:, :self.seq_len, :]
        if return_weights:
            return x_out.transpose(-2,-1), [attn1_weights]
        return x_out.transpose(-2,-1)


class onelayer_MultiLayerAttentionModel_attn_stats(nn.Module):
    """
    A PyTorch module implementing a one-layer multi-head attention model with an extended mapping layer.

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden layer.
        output_dim (int): The dimensionality of the output features.
        seq_len (int, optional): The sequence length of the input. Defaults to 20.
        p (int, optional): The dimensionality of the extended mapping layer. Defaults to 30.
        num_heads (int, optional): The number of attention heads in the multi-head attention layer. Defaults to 1.

    Attributes:
        seq_len (int): The sequence length of the input.
        p (int): The dimensionality of the extended mapping layer.
        mapping_A (ExtendedMappingA): A module that maps the input to an extended representation.
        attn1 (MultiHeadAttentionLayer_attn): A multi-head attention layer.

    Methods:
        forward(x, return_weights=False):
            Performs a forward pass through the model.

            Args:
                x (torch.Tensor): The input tensor of shape [batch_size, seq_len, input_dim].
                return_weights (bool, optional): If True, returns the attention weights. Defaults to False.

            Returns:
                torch.Tensor: The output tensor of shape [batch_size, seq_len, output_dim].
                list (optional): A list containing the attention weights if `return_weights` is True.
    """

    def __init__(self, input_dim, prompt_dim, hidden_dim, output_dim, seq_len=20, p=30, num_heads=1):

        super().__init__()
        self.seq_len = seq_len
        self.p = p

        self.mapping_A = ExtendedMappingA_statistical(input_dim=input_dim, prompt_dim=prompt_dim, hidden_dim=hidden_dim, seq_len=seq_len, p=p) # [batch_size, p, hidden_dim]
        self.attn1 = MultiHeadAttentionLayer_attn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads) # [batch_size, p, output_dim]

    def forward(self, x, return_weights=False):
        x_extended = self.mapping_A(x)  
        x_attn1, attn1_weights = self.attn1(x_extended, return_weights=True)
        x_out = x_attn1[:, :self.seq_len, :]
        if return_weights:
            return x_out, [attn1_weights]
        return x_out

class SingleHeadAttentionModel_f(nn.Module):
    """
    SingleHeadAttentionModel implements a single-head attention mechanism with an extended mapping layer.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden representation after mapping.
        output_dim (int): Dimensionality of the output features from the attention mechanism.
        seq_len (int, optional): Sequence length for the output. Defaults to 20.
        p (int, optional): Number of projected tokens/features after mapping. Defaults to 30.

    Attributes:
        seq_len (int): Sequence length for the output.
        p (int): Number of projected tokens/features after mapping.
        mapping_A (nn.Module): Extended mapping layer to project input to hidden_dim.
        q_proj (nn.Linear): Linear layer to project input to queries.
        k_proj (nn.Linear): Linear layer to project input to keys.
        v_proj (nn.Linear): Linear layer to project input to values.

    Forward Args:
        x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
        return_weights (bool, optional): If True, also returns attention weights. Defaults to False.

    Returns:
        torch.Tensor: Attention output of shape [batch_size, seq_len, output_dim].
        torch.Tensor (optional): Attention weights of shape [batch_size, p, p] if return_weights is True.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len=20, p=30):
        super(SingleHeadAttentionModel_f, self).__init__()
        self.seq_len = seq_len
        self.p = p

        # Define the extended mapping layer
        self.mapping_A = ExtendedMappingA(input_dim=input_dim+1, hidden_dim=hidden_dim, seq_len=seq_len, p=p)
        self.mapping_B = ExtendedMappingA(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len, p=p)
        # Define the single-head attention layer
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, output_dim)
        # self.out_proj = nn.Linear(output_dim, output_dim)
    def forward(self, x, w, return_weights=False):
        x = self.mapping_A(x)  # shape [batch_size, p, hidden_dim]
        w = self.mapping_B(w)  # shape [batch_size, p, hidden_dim]
        Q = self.q_proj(w) # shape [batch_size, p, hidden_dim]
        K = self.k_proj(x) # shape [batch_size, p, hidden_dim]
        V = self.v_proj(x)  # shape [batch_size, p, output_dim]
        scores = torch.matmul(K, Q.transpose(-2,-1)) # shape [batch_size, p, p]
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V) # shape [batch_size, p, output_dim]
        # Reshape the output to match the expected output shape
        # attn_output = self.out_proj(attn_output)
        attn_output = attn_output[:, :self.seq_len, :]
        if return_weights:
            return attn_output, attn_weights
        return attn_output


class SingleHeadAttentionModel(nn.Module):
    """
    SingleHeadAttentionModel implements a single-head attention mechanism with an extended mapping layer.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden representation after mapping.
        output_dim (int): Dimensionality of the output features from the attention mechanism.
        seq_len (int, optional): Sequence length for the output. Defaults to 20.
        p (int, optional): Number of projected tokens/features after mapping. Defaults to 30.

    Attributes:
        seq_len (int): Sequence length for the output.
        p (int): Number of projected tokens/features after mapping.
        mapping_A (nn.Module): Extended mapping layer to project input to hidden_dim.
        q_proj (nn.Linear): Linear layer to project input to queries.
        k_proj (nn.Linear): Linear layer to project input to keys.
        v_proj (nn.Linear): Linear layer to project input to values.

    Forward Args:
        x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
        return_weights (bool, optional): If True, also returns attention weights. Defaults to False.

    Returns:
        torch.Tensor: Attention output of shape [batch_size, seq_len, output_dim].
        torch.Tensor (optional): Attention weights of shape [batch_size, p, p] if return_weights is True.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len=20, p=30):
        super(SingleHeadAttentionModel, self).__init__()
        self.seq_len = seq_len
        self.p = p

        # Define the extended mapping layer
        self.mapping_A = ExtendedMappingA(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len, p=p)
        # Define the single-head attention layer
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, output_dim)
        # self.out_proj = nn.Linear(output_dim, output_dim)
    def forward(self, x, return_weights=False):
        x = self.mapping_A(x)  # shape [batch_size, p, hidden_dim]
        batch_size, seq_len, _ = x.size() # shape [batch_size, p, hidden_dim]
        Q = self.q_proj(x) # shape [batch_size, p, hidden_dim]
        K = self.k_proj(x) # shape [batch_size, p, hidden_dim]
        V = self.v_proj(x)  # shape [batch_size, p, output_dim]
        scores = torch.matmul(Q, K.transpose(-2,-1)) # shape [batch_size, p, p]
        attn_weights = F.softmax(scores, dim=-1) 
        attn_output = torch.matmul(attn_weights, V) # shape [batch_size, p, output_dim]
        # Reshape the output to match the expected output shape
        # attn_output = self.out_proj(attn_output)
        attn_output = attn_output[:, :self.seq_len, :]
        if return_weights:
            return attn_output, attn_weights
        return attn_output
    
