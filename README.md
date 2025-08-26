# In‑Context Algorithm Emulation in Fixed‑Weight Transformers

This is the code for the paper: [**In-Context Algorithm Emulation in Fixed-Weight Transformers**](https://arxiv.org/abs/2508.17550). You can use this repo to reproduce the results in the paper.

## Environmental Setup

1. Clone the repository
    ```bash
    git clone <your-repo-url>
    cd algo_emu
    ```
2. Create and activate a virtual environment
    ```bash
    conda create -n algo_emu python=3.10
    conda activate algo_emu
    ```
3. Install required packages
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Attention Emulating Continuous Function (Theorem 2.1)
Run ```att_sim_f.py```.

### Attention Emulating Attention Head (Theorem 3.1/3.2)
Run ```attn_sim_attn.py```.

### Attention Emulating Statistical Models (Corollary 3.2.1)
Run ```att_sim_statistical.py``` using synthetic data, and ```att_sim_statistical_ames_data.py``` using [Ames Housing Data](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset).

## Citation
