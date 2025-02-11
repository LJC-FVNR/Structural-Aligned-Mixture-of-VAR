import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from types import SimpleNamespace

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        From: https://github.com/meta-llama/llama/blob/main/llama/model.py
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @classmethod
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    def forward(self, x):
        output = self.norm(x.float()).type_as(x)
        return output * self.weight

NORM = RMSNorm

class LinearAttention(nn.Module):
    def __init__(self, config):
        super(LinearAttention, self).__init__()
        self.D = config.n_embd
        self.activation = nn.SiLU()
        self.n_head = config.n_head

        assert self.D % self.n_head == 0
        self.d = self.D // self.n_head

        self.c_proj = nn.Linear(self.D, self.D, bias=config.bias)
        self.k1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.q1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.v1 = nn.Linear(self.D, self.D, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, X):
        B, L, D = X.shape
        device = X.device
        
        Q = self.q1(X).reshape(B, L, self.n_head, -1)                        
        K = self.k1(self.dropout(X)).reshape(B, L, self.n_head, -1).unsqueeze(-1)     # B L H d 1
        V = self.v1(X).reshape(B, L, self.n_head, -1).unsqueeze(-1).permute(0, 1, 2, 4, 3)   # B L H 1 d
        W = torch.einsum('blhdk,blhke->blhde', K, V)         # B L H d d
        W = torch.cumsum(W, dim=1)
        X = torch.einsum('blhd,blhde->blhe', Q, W)
        X = X.reshape(B, L, D)
        X = self.dropout(self.c_proj(X))
        return X
    
    def generate_weights(self, X):
        B, L, D = X.shape
        device = X.device
        
        Q = self.q1(X).reshape(B, L, self.n_head, -1)                        
        K = self.k1(self.dropout(X)).reshape(B, L, self.n_head, -1)     # B L H d
        QKT = torch.einsum('blhd,bihd->bhli', Q, K)
        QKT = torch.tril(QKT)
        return QKT

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd) # , bias=config.bias
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layer = config.n_layer
        self.n_layer = n_layer
        self.predictor = config.predictor
        if self.predictor == 'LinearAttention':
            self.attn = nn.ModuleList([LinearAttention(config) for i in range(n_layer)])
        else:
            raise
        self.pns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.lns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.mlps = nn.ModuleList([MLP(config) for i in range(n_layer)])

        self.layer_attn = nn.Linear(config.n_embd, n_layer*2)

    def forward(self, x):
        for attn, pn, ln, mlp in zip(self.attn, self.pns, self.lns, self.mlps):
            x = x + attn(pn(x))
            x = x + mlp(ln(x))
        return x

    def get_attn_input(self, x):
        B, L, D = x.shape
        attn_input = []
        x_shortcut = []
        for attn, pn, ln, mlp in zip(self.attn, self.pns, self.lns, self.mlps):
            attn_input.append(pn(x))
            x_shortcut.append(x)
            x = x + attn(pn(x))
            x = x + mlp(ln(x))
        return attn_input, x_shortcut

class MultiSegmentTokenizer(nn.Module):
    def __init__(self, s: int, n_channels: int, d_model: int, number_of_targets=0, univariate=False):
        super(MultiSegmentTokenizer, self).__init__()
        self.s = s
        self.tokenizer = nn.Linear(s, d_model)
        self.d_model = d_model
        n_targets = number_of_targets if number_of_targets else n_channels
        
        self.context_channel_projection = nn.Linear(n_channels, n_targets)
        
        self.context_temporal_projection = nn.Linear(s, d_model)
        self.tokenizer_output = nn.Linear(d_model, s)
        self.C = n_channels
        self.number_of_targets = number_of_targets
        self.input_norm_main = NORM(d_model)
        self.input_norm_ctx = NORM(d_model)
        self.univariate = univariate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B C L -> # B C N d
        B, C, L = x.shape
        pad_len = (self.s - (L % self.s)) % self.s
        x_padded = F.pad(x, (pad_len, 0), 'constant', 0)
        L_padded = L + pad_len
        N = L_padded // self.s
        x_segmented = x_padded.view(B, C, N, self.s) # B C N s
        return x_segmented

    def tokenize(self, x_segmented):
        # x_segmented: B C N s
        x_ctx = x_segmented.clone()
        B, C, N, s = x_ctx.shape
        x_ctx = x_ctx.permute(0, 2, 3, 1) # B N s C
        x_ctx = self.context_channel_projection(x_ctx)             # B N s Ct
        x_ctx = x_ctx.permute(0, 3, 1, 2)                          # B Ct N s
        x_ctx = self.context_temporal_projection(x_ctx)            # B Ct N d
        if self.univariate:
            x_ctx = torch.zeros_like(x_ctx)
        
        x_main = x_segmented[:, -self.number_of_targets:]          # B Ct N s
        x_main = self.tokenizer(x_main)                            # B Ct N d
        x_main = torch.stack([self.input_norm_ctx(x_ctx), self.input_norm_main(x_main)], dim=3).reshape(B, -1, N*2, self.d_model)    # B Ct N*2 d
        return x_main

    def inverse_tokenize(self, x):
        # B Ct N d -> B Ct N*s
        B, C, N, d = x.shape
        x = self.tokenizer_output(x)
        return x

    def process_single_token(self, x):
        # x: (B, C, s)
        x = self.tokenizer(x)
        x = x.unsqueeze(-2)  # B C 1 d
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        pad_len = (self.pred_len - (self.seq_len % self.pred_len)) % self.pred_len
        self.max_len = 2*(self.seq_len + pad_len) // self.pred_len #! 2 * (self.seq_len + pad_len) // self.pred_len
        self.enc_in = configs.enc_in
        self.number_of_targets = configs.number_of_targets if configs.number_of_targets else configs.enc_in
        n_series = self.number_of_targets
        self.n_series = n_series
        self.d_input = int(np.sqrt(n_series)) * 32 if not configs.d_model else configs.d_model

        self.tokenizer = MultiSegmentTokenizer(s=self.pred_len, n_channels=self.enc_in, d_model=self.d_input, number_of_targets=self.number_of_targets)
        self.dropout = nn.Dropout(configs.dropout)
        self.pre_norm = NORM(1*self.d_input)
        self.final_norm = NORM(1*self.d_input)
        transformer_config = SimpleNamespace(n_layer=configs.e_layers, n_embd=1*self.d_input, n_head=configs.n_heads, dropout=configs.dropout, 
                                             bias=True, max_len=self.max_len, predictor=configs.predictor)
        self.transformer = Transformer(transformer_config)
        self.pos_emb = nn.Embedding(self.max_len, 1*self.d_input)
        self.channel_emb = nn.Parameter(0.02*torch.randn(1, self.number_of_targets, self.max_len, 1*self.d_input))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * configs.e_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, *args, **kwargs):
        # x: [Batch, Input length, Channel]
        
        x = x.permute(0, 2, 1)  # B C L

        # RevIN: preprocessing
        mean = x[:, :, -self.pred_len:].mean(dim=2, keepdim=True)  # B C 1
        std = x.std(dim=2, keepdim=True) + 1e-8                    # B C 1
        x = (x - mean) / std

        x_segmented = self.tokenizer(x)   # B C N s
        x_segmented_target = x_segmented[:, -self.number_of_targets:]

        # Input Tokenization
        x = self.tokenizer.tokenize(x_segmented)   # B Ct N d
        
        B, C, N, d = x.shape
        pos = torch.arange(0, N, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos) + self.channel_emb[:, :, 0:N, :]
        x = self.dropout(x) + pos_emb  # B C N d
        x = self.pre_norm(x)

        # AR/ARMA Transformer
        x = x.view(B*C, N, d)
        x = self.transformer(x)

        # Output Projection
        x = self.final_norm(x)     # B*Ct N d
        #x = x.view(B, C, N, d)     # B Ct N d
        x = x.view(B, C, N//2, 2, d)[:, :, :, 1, :]     # B Ct N d
        x = self.tokenizer.inverse_tokenize(x)  # B Ct N s

        # Next-step prediction loss
        if self.training:
            loss = F.mse_loss(x[:, :, 0:-1, :], x_segmented_target[:, :, 1:, :])

        # Next-step output
        x = x[:, :, -1, :]         # B Ct s

        # RevIN: inverse processing
        x = (x*std[:, -C:] + mean[:, -C:]).permute(0, 2, 1)                  # B s Ct

        if self.training:
            return x, loss
        else:
            return x

    def get_attn_input(self, x, x_mark=None):
        with torch.no_grad():
            x = x.permute(0, 2, 1)  # B C L

            # RevIN: preprocessing
            mean = x[:, :, -self.pred_len:].mean(dim=2, keepdim=True)  # B C 1
            std = x.std(dim=2, keepdim=True) + 1e-8                    # B C 1
            x = (x - mean) / std
    
            x_segmented = self.tokenizer(x)   # B C N s
            x_segmented_target = x_segmented[:, -self.number_of_targets:]
    
            # Input Tokenization
            x = self.tokenizer.tokenize(x_segmented)   # B Ct N d
            
            B, C, N, d = x.shape
            pos = torch.arange(0, N, dtype=torch.long, device=x.device)
            pos_emb = self.pos_emb(pos) + self.channel_emb[:, :, 0:N, :]
            x = self.dropout(x) + pos_emb  # B C N d
            x = self.pre_norm(x)
    
            # AR/ARMA Transformer
            x = x.view(B*C, N, d)
            
            attn_input, x_shortcut = self.transformer.get_attn_input(x)
        return attn_input, x_shortcut