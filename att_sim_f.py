# In-context Dataset
# We want to sample the dataset on the fly
# Z being training data X
# T being target data f()X


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from attention_model import SingleHeadAttentionModel_f

class SimFDataset(Dataset):
    """
    A PyTorch Dataset for generating synthetic sequence data with associated targets.
    Each sample consists of:
        - Input tensor `x`: Concatenation of a non-linear transformation of the data and a broadcasted target vector.
        - Target tensor: Non-linear transformation of the data.
        - Weight matrix `w`: The randomly initialized weight matrix used in the transformation.
    Args:
        num_samples (int): Number of samples in the dataset.
        seq_len (int): Length of each sequence.
        input_dim (int): Dimensionality of each input vector in the sequence.
    Attributes:
        data (Tensor): Randomly generated input data of shape (num_samples, seq_len, input_dim).
        w (Tensor): Randomly initialized weight matrix of shape (seq_len, input_dim).
        y (Tensor): Randomly initialized target vector of shape (seq_len, 1).
        targets (Tensor): Non-linear transformation of `data` using `w` and `y`, shape (num_samples, seq_len, input_dim).
        x (Tensor): Concatenation of `targets` and broadcasted `y`, shape (num_samples, seq_len, input_dim + 1).
    Methods:
        f(x): Applies a non-linear transformation to input `x` using `w` and `y`.
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the input tensor, target tensor, and weight matrix for the sample at index `idx`.
    """
    def __init__(self, num_samples, seq_len, input_dim):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.data = 10 * torch.rand(num_samples, seq_len, input_dim) - 5 # N x n x d
        self.w = torch.randn(seq_len, input_dim)  # n x d
        # broadcast w to N x n x d
        # self.w = self.w.unsqueeze(0).expand(num_samples, -1, -1)  # N x n x d
        self.y = torch.randn(seq_len, 1)  # n x 1
        self.targets = self.f(self.data) @ self.data  # N x n x d
        y_broadcast = self.y.unsqueeze(0).expand(num_samples, -1, -1) # N x n x 1
        # concatenate self.data and self.y in column dimension
        self.x = torch.cat((self.targets, y_broadcast), dim=2) # N x n x (d+1)

    def f(self, x):
        y = torch.tanh(x @ self.w.transpose(-2, -1)+self.y)
        return y
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # x [N x n x (d+1)], y [N x n x d], w [N x n x d]
        return self.x[idx], self.targets[idx], self.w


# Train the model
def train_model(p, dataset, sub_epochs = 10, batch_size=32, learning_rate=0.001, seq_len=20, hidden_dim=32):
    model = SingleHeadAttentionModel_f(
        input_dim=dataset.input_dim,
        hidden_dim=hidden_dim,
        output_dim = dataset.input_dim,
        seq_len=seq_len,
        p=p,
    )
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Create DataLoader
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(sub_epochs):
        avg_loss = 0
        for i, (x, y, w) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x, w)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
        avg_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{sub_epochs}, Loss: {avg_loss:.4f}")

        # Testing loop
        with torch.no_grad():
            test_loss = 0
            for x, y, w in test_loader:
                output = model(x, w)
                loss = criterion(output, y)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    # Hyperparameters
    num_samples = 50000
    seq_len = 20
    input_dim = 24
    hidden_dim = 64
    p = 60
    sub_epochs = 100
    batch_size = 32
    learning_rate = 0.001

    # Create dataset and train model
    dataset = SimFDataset(num_samples, seq_len, input_dim)
    train_model(p, dataset, sub_epochs, batch_size, learning_rate, seq_len, hidden_dim)