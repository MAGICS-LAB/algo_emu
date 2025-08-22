import matplotlib.pyplot as plt
import numpy as np

# Data
num_heads = np.array([1, 2, 4, 6, 8, 12])
positions = np.arange(len(num_heads))  # [0, 1, 2, 3, 4, 5] for equal spacing

# Ridge
mean_loss_ridge = np.array([0.112659, 0.161929, 0.147661, 0.132578, 0.144022, 0.151732])
std_loss_ridge = np.array([0.015297, 0.067028, 0.083245, 0.057615, 0.085316, 0.037979])

# Lasso
mean_loss_lasso = np.array([0.1286, 0.0863, 0.0853, 0.0948, 0.0756, 0.0857])
std_loss_lasso = np.array([0.0368, 0.0336, 0.0249, 0.0280, 0.0216, 0.0194])

# Linear
mean_loss_linear = np.array([0.147226, 0.131675, 0.168894, 0.135289, 0.158500, 0.157613])
std_loss_linear = np.array([0.021029, 0.053331, 0.092925, 0.062859, 0.078760, 0.049089])

# Set up subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # wider for more space between

# Plot configs
models = [
    ("Lasso", mean_loss_lasso, std_loss_lasso),
    ("Ridge", mean_loss_ridge, std_loss_ridge),
    ("Linear", mean_loss_linear, std_loss_linear)
]

for i, (name, mean, std) in enumerate(models):
    ax = axes[i]
    ax.plot(positions, mean, color='darkgreen', marker='o', linewidth=3)
    ax.fill_between(positions, mean - std, mean + std, color='gray', alpha=0.2)
    ax.set_title(name, fontsize=26, pad=12)
    ax.set_xlabel("Number of Heads", fontsize=26, labelpad=10)    
    # Ensure only integer x-ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(num_heads, fontsize=22)
    ax.tick_params(axis='both', labelsize=22)

    if i == 0:
        ax.set_ylabel("Mean Loss", fontsize=26, labelpad=10)


plt.tight_layout()
plt.savefig("plot_heads_all_models_independent_y.pdf", format="pdf", bbox_inches='tight')
plt.show()
