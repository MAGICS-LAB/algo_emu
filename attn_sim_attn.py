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
from attention_model import onelayer_MultiLayerAttentionModel_attn

# In-context Dataset
class SimAttentionDataset(Dataset):
    """
    A PyTorch Dataset class for simulating attention-based data.

    Args:
        num_samples (int): The number of samples in the dataset.
        seq_len (int): The length of each sequence.
        input_dim (int): The dimensionality of the input tokens.
    """

    def __init__(self, num_samples, seq_len, input_dim, hidden_dim):
        self.num_samples = num_samples # N
        self.seq_len = seq_len # n
        self.input_dim = input_dim # d
        # raw each entry of X from a uniform distribution from -1 to 1
        self.data = 2 * torch.randn(num_samples, seq_len, input_dim) - 1 # N x n x d
        self.wk = torch.randn(hidden_dim, input_dim) # 1 x h x d
        self.wq = torch.randn(hidden_dim, input_dim) # 1 x h x d
        self.wv = torch.randn(input_dim, input_dim) # 1 x d x d

    def target(self, X):
        # Simlulate the target data
        K = self.wk @ X.transpose(-2, -1) # N x h x n
        Q = self.wq @ X.transpose(-2, -1) # N x h x n
        V = self.wv @ X.transpose(-2, -1) # N x d x n
        attn_weights = torch.softmax(K.transpose(-2, -1) @ Q , dim=-1) # N x n x n
        Y = torch.matmul(V, attn_weights) # N x d x n
        return Y, K, Q, V

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        y, k, q, v = self.target(x)
        # x [N x n x d], y [N x d x n], k [N x h x n], q [N x h x n], v [N x d x n]
        return x, y, k, q, v

# Train the model
def train_model(num_head, p, dataset, sub_epochs = 10, batch_size=32, learning_rate=0.001, seq_len=20, hidden_dim=32):
    model_k = onelayer_MultiLayerAttentionModel_attn(
        input_dim=dataset.input_dim,
        hidden_dim=hidden_dim,
        output_dim = hidden_dim,
        seq_len=seq_len,
        p=p,
        num_heads=num_head
    ) # n x h

    model_q = onelayer_MultiLayerAttentionModel_attn(
        input_dim=dataset.input_dim,
        hidden_dim=hidden_dim,
        output_dim = hidden_dim,
        seq_len=seq_len,
        p=p,
        num_heads=num_head
    )
    model_v = onelayer_MultiLayerAttentionModel_attn(
        input_dim=dataset.input_dim,
        hidden_dim=hidden_dim,
        output_dim = dataset.input_dim,
        seq_len=seq_len,
        p=p,
        num_heads=num_head
    )

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    mse = nn.MSELoss()

    # ── Phase 1: Train model_k,q,v separately ──
    optim_k = optim.Adam(model_k.parameters(), lr=learning_rate)
    optim_q = optim.Adam(model_q.parameters(), lr=learning_rate)
    optim_v = optim.Adam(model_v.parameters(), lr=learning_rate)

    for epoch in range(sub_epochs):
        epoch_loss_k = epoch_loss_q = epoch_loss_v = 0
        for x, y, k_target, q_target, v_target in train_loader:
            # x [N x n x d], y [N x d x n], k [N x h x n], q [N x h x n], v [N x d x n]
            # --- k ---
            optim_k.zero_grad()
            k_prime = model_k(x) 
            loss_k = mse(k_prime, k_target)
            loss_k.backward()
            optim_k.step()

            # --- q ---
            optim_q.zero_grad()
            q_prime = model_q(x)
            loss_q = mse(q_prime, q_target)
            loss_q.backward()
            optim_q.step()

            # --- v ---
            optim_v.zero_grad()
            v_prime = model_v(x)
            loss_v = mse(v_prime, v_target)
            loss_v.backward()
            optim_v.step()

            epoch_loss_k += loss_k.item()
            epoch_loss_q += loss_q.item()
            epoch_loss_v += loss_v.item()

        n = len(train_loader)
        print(f"[Submodels] Epoch {epoch+1}/{sub_epochs}  "
              f"Loss_k={epoch_loss_k/n:.4f}  "
              f"Loss_q={epoch_loss_q/n:.4f}  "
              f"Loss_v={epoch_loss_v/n:.4f}")

    # # Optionally freeze submodels before Phase 2:
    # for param in model_k.parameters(): param.requires_grad = False
    # for param in model_q.parameters(): param.requires_grad = False
    # for param in model_v.parameters(): param.requires_grad = False

    # evaluation...
    model_k.eval()
    model_q.eval()
    model_v.eval()

    with torch.no_grad():
        # Evaluate the model on the test set
        test_loss_k = test_loss_q = test_loss_v = 0
        for x, y, k_target, q_target, v_target in test_loader:
            k_prime = model_k(x)
            loss_k = mse(k_prime, k_target)

            q_prime = model_q(x)
            loss_q = mse(q_prime, q_target)

            v_prime = model_v(x)
            loss_v = mse(v_prime, v_target)

            test_loss_k += loss_k.item()
            test_loss_q += loss_q.item()
            test_loss_v += loss_v.item()
        n = len(test_loader)
        print(f"[Submodels] Test Loss_k={test_loss_k/n:.4f}  "
              f"Test Loss_q={test_loss_q/n:.4f}  "
              f"Test Loss_v={test_loss_v/n:.4f}")
        test_loss = 0
        for x, y, _, _, _ in test_loader:
            k_prime = model_k(x)
            q_prime = model_q(x)
            v_prime = model_v(x)
            attn_weights = torch.softmax(k_prime.transpose(-2, -1) @ q_prime, dim=-1)
            y_prime = torch.matmul(v_prime, attn_weights)
            loss = mse(y_prime, y)
            test_loss += loss.item()
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    return test_loss / len(test_loader)

def p_experiment(p_list, seeds, dataset, seq_len, hidden_dim,
                 batch_size, learning_rate, sub_epochs):
    p_results = []
    for p in p_list:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            loss = train_model(
                num_head=1,
                p=p,
                dataset=dataset,
                sub_epochs=sub_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seq_len=seq_len,
                hidden_dim=hidden_dim
            )

            print(f"[p Exp] seed={seed} p={p} → Test Loss={loss:.4f}")
            p_results.append({"seed": seed, "p": p, "test_loss": loss})

    df_ps = pd.DataFrame(p_results)
    df_ps.to_csv("p_experiment_results.csv", index=False)
    summary_ps = df_ps.groupby("p")["test_loss"].agg(["mean","std"]).reset_index()
    print(summary_ps)

def head_experiment(head_list, seeds, dataset, seq_len, hidden_dim,
                    batch_size, learning_rate, sub_epochs):
    head_results = []
    for num_head in head_list:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            loss = train_model(
                num_head=num_head,
                p=60,
                dataset=dataset,
                sub_epochs=sub_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seq_len=seq_len,
                hidden_dim=hidden_dim
            )

            print(f"[head Exp] seed={seed} num_head={num_head} → Test Loss={loss:.4f}")
            head_results.append({"seed": seed, "num_head": num_head, "test_loss": loss})

    df_heads = pd.DataFrame(head_results)
    df_heads.to_csv("head_experiment_results.csv", index=False)
    summary_heads = df_heads.groupby("num_head")["test_loss"].agg(["mean","std"]).reset_index()
    print(summary_heads)


# Main function
def main():
    # Hyperparameters
    num_samples = 50000
    seq_len = 20
    input_dim = 24
    hidden_dim = 2 * input_dim
    batch_size = 32
    learning_rate = 0.001
    sub_epochs = 50
    # Create dataset
    dataset = SimAttentionDataset(num_samples, seq_len, input_dim, hidden_dim)
    # Train the model
    # train_model(num_head_fixed, p_fixed, dataset, sub_epochs=sub_epochs, num_epochs=num_epochs,
    #             batch_size=batch_size, learning_rate=learning_rate,
    #             seq_len=seq_len, hidden_dim=hidden_dim)
    
    seeds = [1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244]
    #### p-value experiment
    p_list  = [20,30,40,50,60,70,80,90,100,110]
    p_experiment(
        p_list=p_list,
        seeds=seeds,
        dataset=dataset,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sub_epochs=sub_epochs
    )

    #### head experiment
    head_list = [1,2,4,6,8,12]
    head_experiment(
        head_list=head_list,
        seeds=seeds,
        dataset=dataset,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sub_epochs=sub_epochs
    )

if __name__ == "__main__":
    main()