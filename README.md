# Linear Transformers as VAR Models:  Aligning Autoregressive Attention Mechanisms with Autoregressive Forecasting

This repository provides the official implementation of [SAMoVAR](https://arxiv.org/abs/2502.07244). The visualization in the paper can be found at `visualization/`. The main model can be found in `models/AutoregressiveAlignment.py`, where the core codes of SAMoVAR structure are as follows:

```python
# SAMoVAR Attention
class LinearAttention(nn.Module):
    def __init__(self, config):
        super(LinearAttention, self).__init__()
        self.D = config.n_embd
        self.n_head = config.n_head
        assert self.D % self.n_head == 0
        self.d = self.D // self.n_head
        self.q_proj = nn.Linear(self.D, self.D, bias=config.bias)
        self.v_proj = nn.Linear(self.D, self.D, bias=config.bias)
        self.q_norm = RMSNormMultiHead(self.n_head, self.d)
        self.v_norm = RMSNormMultiHead(self.n_head, self.d)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, X, qv_input=None):
        B, L, D = X.shape
        K = X.reshape(B, L, self.n_head, -1)
        Q = self.q_proj(X if qv_input is None else qv_input).reshape(B, L, self.n_head, -1)
        Q = self.q_norm(Q)
        V = self.v_proj(X if qv_input is None else qv_input).reshape(B, L, self.n_head, -1) 
        V = self.v_norm(V)
        W = torch.einsum('blhd,blhe->blhde', K, V)         # B L H d d
        W = torch.cumsum(W, dim=1)
        X = torch.einsum('blhd,blhde->blhe', Q, W)
        X = X.reshape(B, L, D)
        X = self.dropout(X)
        return X

# SAMoVAR Transformer
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layer = config.n_layer
        self.n_layer = n_layer
        self.predictor = config.predictor
        if self.predictor == 'LinearAttention':
            self.attn = nn.ModuleList([LinearAttention(config) for i in range(n_layer)])
        elif self.predictor == 'FixedVAR':
            self.attn = nn.ModuleList([FixedVAR(config) for i in range(n_layer)])
        self.pns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.lns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.mlps = nn.ModuleList([MLP(config) for i in range(n_layer)])

        self.use_AR_structure = True
        self.VAR_input_norm = NORM(config.n_embd)
        
        self.n_head = config.n_head
        self.d = config.n_embd // self.n_head
        self.generate_S = BatchedInvertibleMatrix(h=self.n_head, d=self.d)

    def forward(self, x):
        B, L, D = x.shape
        if not self.use_AR_structure:
            for attn, pn, ln, mlp in zip(self.attn, self.pns, self.lns, self.mlps):
                x = x + attn(pn(x))
                x = x + mlp(ln(x))
            return x
        else:
            for ln, mlp in zip(self.lns, self.mlps):
                x = x + mlp(ln(x))

            BC, N, D = x.shape
            x = self.VAR_input_norm(x)
            x_orig = x.clone()
            attn_res = []
            index = len(self.attn)

            S = self.generate_S(get_inverse=False)  # structural matrix D in the paper
            for attn in self.attn:
                index -= 1
                x = attn(x, qv_input=x_orig)
                attn_res.append(x.view(BC, N//2, 2, D)[:, :, 1, :])

            attn_res = sum(attn_res)
            attn_res = torch.einsum('blhd,hde->blhe', attn_res.reshape(BC, N//2, self.n_head, self.d), S).reshape(BC, N//2, D)
            attn_res = x_orig.view(BC, N//2, 2, D)[:, :, 1, :] + attn_res
            return attn_res
```

### 1. Install the Required Packages

Begin by installing PyTorch with GPU support according to the instructions on [PyTorch's official website](https://pytorch.org/get-started/locally/). After that, install the remaining dependencies with:

```bash
pip install -r requirements.txt
```

### 2. Download the Datasets

Download the 12 datasets used in the paper from the link provided by [itransformer](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view) [1]. Once downloaded, place the dataset files in the `dataset/` directory.

### 3. Run the Training Scripts

To train the SAMoVAR, LinTrans, or FixedVAR models, execute:

```bash
bash samovar.sh
```

For the baseline models mentioned in the paper, run:

```bash
bash baseline.sh
```

### 4. Monitor the Training Process

You can use TensorBoard to visualize the training process. Launch TensorBoard with:

```bash
nohup tensorboard --logdir runs --port 6006 --bind_all > tensorb.log 2>&1 &
```

### 5. Training on Custom Data

If you wish to train the models on your own dataset, ensure your CSV file is formatted with the first column labeled `date` (containing timestamps) and the remaining columns holding the time series values. Place your dataset file in the `dataset/` folder.

Afterwards, update the following arrays in the scripts to include your dataset information:

```bash
data_names=("PEMS/PEMS03.npz" "PEMS/PEMS04.npz" "PEMS/PEMS07.npz" "PEMS/PEMS08.npz" "ETT-small/ETTm1.csv" "ETT-small/ETTm2.csv" "ETT-small/ETTh1.csv" "ETT-small/ETTh2.csv" "weather/weather.csv" "Solar/solar_AL.txt" "electricity/electricity.csv" "traffic/traffic.csv")
data_alias=("PEMS03" "PEMS04" "PEMS07" "PEMS08" "ETTm1" "ETTm2" "ETTh1" "ETTh2" "Weather" "Solar" "ECL" "Traffic")
data_types=("PEMS" "PEMS" "PEMS" "PEMS" "ETTm1" "ETTm2" "ETTh1" "ETTh2" "custom" "Solar" "custom" "custom")
enc_ins=(358 307 883 170 7 7 7 7 21 137 321 862)  # Number of time series in each dataset
batch_sizes=(8 8 8 8 32 32 32 32 32 8 8 8)         # Batch sizes for each dataset
grad_accums=(4 4 4 4 1 1 1 1 1 4 4 4)               # Gradient accumulation steps
```

Adjust these lists as needed to match the configuration of your custom dataset.

## References

[1] Liu, Yong, et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." arXiv preprint arXiv:2310.06625 (2023).
