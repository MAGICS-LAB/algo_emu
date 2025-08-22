import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model
from attention_model import onelayer_MultiLayerAttentionModel_attn_stats, onelayer_MultiLayerAttentionModel_attn

# In-context Dataset
class SimStatsDataset(Dataset):
    """
    A PyTorch Dataset class for simulating attention-based data.
    """

    def __init__(self, num_samples, seq_len, input_dim, prompt_dim, algorithm, noise_std =0.05):
        """
        Args:
            num_samples (int): The number of samples in the dataset.
            seq_len (int): The length of each sequence.
            input_dim (int): The dimensionality of the input tokens.
            hidden_dim (int): The dimensionality of the hidden layer.
            noise_std (float): Standard deviation of the noise added to the target.
            algorithm (list): List of algorithms to simulate, e.g., ['linear', 'lasso'].
        """
        self.sample = []
        self.seq_len = seq_len # n
        self.input_dim = input_dim # d
        self.prompt_dim = prompt_dim # p
        self.algorithm = algorithm
        self.noise_std = noise_std
       
        # populate the data with random values
        for n in range(num_samples):
            # raw each entry of X from a uniform distribution from -1 to 1
            x = 2 * torch.randn(seq_len, input_dim) - 1 # N x n x d

            if algorithm == 'mixed':
                algo = random.choice(['linear', 'ridge', 'lasso'])
            else:
                algo = algorithm

            w = torch.randn(prompt_dim, 1) # N x p x 1
            if algo == 'lasso':
                mask = torch.rand_like(w) > 0.5
                w = w * mask.float()  # Apply sparsity: dimension being p x 1

            if algo == 'ridge':
                lam = 5
                y = x @ w + noise_std * torch.randn(seq_len, 1)  # N x n x 1
                xtx = x.T @ x  # N x d x d
                xty = x.T @ y
                w = torch.linalg.solve(xtx + lam * torch.eye(input_dim), xty)  # N x d x 1

            y = x @ w + noise_std * torch.randn(seq_len, 1)
            
            self.sample.append((x, y, w.flatten())) # N x n x d, N x n x 1, N x p

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]
    

class OneLayerAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, prompt_dim, seq_len, number_of_heads=1, p=60):
        super().__init__()
        self.attnm = onelayer_MultiLayerAttentionModel_attn_stats(
            input_dim=input_dim, prompt_dim=prompt_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
            seq_len=seq_len, p=p, num_heads=number_of_heads
        )
        # self.attns = onelayer_MultiLayerAttentionModel_attn(
        #     input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim,
        #     seq_len=seq_len, p=p, num_heads= number_of_heads
        # )
        self.linear = nn.Linear(hidden_dim, 1)  # Final linear layer to output the prediction

    def forward(self, x, w_prompt):
        # Simulate prepending W^a to x, flatten w and tile across tokens
        w_tokens = w_prompt.unsqueeze(1).repeat(1, x.size(1), 1) # N x n x p
        x_aug = torch.cat([x, w_tokens], dim=-1)  # N x n x (d+p) 
        # h = self.attns(self.attnm(x_aug))
        h = self.attnm(x_aug)  # N x p x hidden_dim
        y_pred = self.linear(h)  # N x n x 1
        return y_pred

# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load frozen model for inference
def load_frozen_model(path, input_dim, hidden_dim, prompt_dim, seq_len, num_head, p):
    model = OneLayerAttention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        prompt_dim=prompt_dim,
        seq_len=seq_len,
        number_of_heads=num_head,
        p=p,
    )
    model.load_state_dict(torch.load(path))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

# Evaluate fixed model on new algorithm
def evaluate_on_task(frozen_model, dataset_name, num_samples, input_dim, prompt_dim, seq_len):
    test_dataset = SimStatsDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        input_dim=input_dim,
        prompt_dim=prompt_dim,
        algorithm=dataset_name,
        noise_std=0.05
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    mse = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for x, y, w in test_loader:
            y_pred = frozen_model(x, w)
            loss = mse(y_pred, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Frozen model → {dataset_name} Test Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_frozen_model_multiple_seeds(
    model_path, seeds, num_samples, input_dim, prompt_dim, seq_len,
    hidden_dim, num_head, p_fixed
):
    results = {"lasso": [], "linear": [], "ridge": []}

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load frozen model
        model = load_frozen_model(
            path=model_path,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            prompt_dim=prompt_dim,
            seq_len=seq_len,
            num_head=num_head,
            p=p_fixed
        )

        for task in ["lasso", "linear", "ridge"]:
            test_loss = evaluate_on_task(
                frozen_model=model,
                dataset_name=task,
                num_samples=num_samples,
                input_dim=input_dim,
                prompt_dim=prompt_dim,
                seq_len=seq_len
            )
            results[task].append(test_loss)

    # Print summary
    for task in ["lasso", "linear", "ridge"]:
        losses = np.array(results[task])
        mean_loss = losses.mean()
        std_loss = losses.std()
        print(f"{task.capitalize()} → Mean ± Std = {mean_loss:.4f} ± {std_loss:.4f}")


# Extend main() to include freezing and testing
def main():
    # same as before ...
    num_samples = 50000
    seq_len = 20
    input_dim = 24
    hidden_dim = 2 * input_dim
    batch_size = 32
    learning_rate = 0.001
    sub_epochs = 300
    prompt_dim = input_dim
    p_fixed = 60
    num_head_fixed = 6

    li_dataset = SimStatsDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        input_dim=input_dim,
        prompt_dim=prompt_dim,
        algorithm='mixed',  # 'mixed' to include all algorithms
        noise_std=0.05
    )

    print("Training attention model on mixed data...")
    model = OneLayerAttention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        prompt_dim=prompt_dim,
        seq_len=seq_len,
        number_of_heads=num_head_fixed,
        p=p_fixed
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    loader = DataLoader(li_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(sub_epochs):
        model.train()
        epoch_loss = 0
        for x, y, w in loader:
            optimizer.zero_grad()
            loss = mse(model(x, w), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[Mixed Train] Epoch {epoch+1}/{sub_epochs} - Loss: {epoch_loss/len(loader):.4f}")

    # Save trained model
    model_path = "/home/ubuntu/att_sim_att/mixed_trained_attention.pth"
    save_model(model, model_path)

    # Evaluate frozen model on different tasks
    seeds = [42, 43, 44, 45, 46]
    print("\nEvaluating frozen model on different tasks...")
    evaluate_frozen_model_multiple_seeds(
        model_path=model_path,
        seeds=seeds,
        num_samples=1000,
        input_dim=input_dim,
        prompt_dim=prompt_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_head=num_head_fixed,
        p_fixed=p_fixed
    )

if __name__ == "__main__":
    main()