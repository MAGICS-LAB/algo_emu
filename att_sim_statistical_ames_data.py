import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split


# Model
from attention_model import onelayer_MultiLayerAttentionModel_attn_stats, onelayer_MultiLayerAttentionModel_attn

# In-context Dataset
class AmesStatsData(Dataset):
    """
    A PyTorch Dataset class for Ames housing data with attention-based simulation.
    """

    def __init__(self, algorithm='mixed', alpha=1.0):
        self.algorithm = algorithm
        self.alpha = alpha

        # Load Ames dataset
        housing = fetch_openml(name="house_prices", as_frame=True)
        df = housing.frame

        # Log-transform target (SalePrice)
        y = np.log1p(df['SalePrice'])
        X = df.drop(columns='SalePrice')

        # Handle missing values
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna('Missing')
        for col in X.select_dtypes(exclude=['object']).columns:
            X[col] = X[col].fillna(X[col].median())

        # One-hot encode categoricals
        X = pd.get_dummies(X, drop_first=True)

        # Standardize numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit regression models for prompts
        self.models = {
            'linear': LinearRegression().fit(X_scaled, y),
            'lasso': Lasso(alpha=alpha).fit(X_scaled, y),
            'ridge': Ridge(alpha=alpha).fit(X_scaled, y)
        }

        self.weights = {
            name: torch.tensor(model.coef_, dtype=torch.float32)
            for name, model in self.models.items()
        }

        self.features = torch.tensor(X_scaled, dtype=torch.float32)
        self.targets = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].unsqueeze(0)  # (1, d)
        y = self.targets[idx].unsqueeze(0)   # (1, 1)
        algo = random.choice(['linear', 'lasso', 'ridge']) if self.algorithm == 'mixed' else self.algorithm
        w = self.weights[algo]
        return x, y, w

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

def evaluate_on_task(frozen_model, dataset_name, indices):
    test_dataset = AmesStatsData(algorithm=dataset_name)
    test_dataset.features = test_dataset.features[indices]
    test_dataset.targets = test_dataset.targets[indices]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
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
    model_path, seeds, input_dim, prompt_dim, seq_len,
    hidden_dim, num_head, p_fixed, test_indexes
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
                indices=test_indexes
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
    seq_len = 1
    input_dim = 262
    hidden_dim = 2 * input_dim
    batch_size = 32
    learning_rate = 0.001
    sub_epochs = 300
    prompt_dim = input_dim
    p_fixed = 100
    num_head_fixed = 8

    li_dataset = AmesStatsData(algorithm='mixed')
    # split dataset into train and test
    train_size = int(0.8 * len(li_dataset))
    test_size = len(li_dataset) - train_size
    train_dataset, test_dataset = random_split(li_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

    for epoch in range(sub_epochs):
        model.train()
        epoch_loss = 0
        for x, y, w in train_loader:
            optimizer.zero_grad()
            loss = mse(model(x, w), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[Mixed Train] Epoch {epoch+1}/{sub_epochs} - Loss: {epoch_loss/len(train_loader):.4f}")

    # Save trained model
    model_path = "Attn_Sim_Attn/mixed_trained_attention_ames.pth"
    save_model(model, model_path)

    # Evaluate frozen model on different tasks
    seeds = [42, 43, 44, 45, 46]
    print("\nEvaluating frozen model on different tasks...")
    evaluate_frozen_model_multiple_seeds(
        model_path=model_path,
        seeds=seeds,
        input_dim=input_dim,
        prompt_dim=prompt_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_head=num_head_fixed,
        p_fixed=p_fixed,
        test_indexes=test_dataset.indices
    )

if __name__ == "__main__":
    main()