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

class RMSNormMultiHead(nn.Module):
    def __init__(self, n_head: int, dim: int, eps: float = 1e-8):
        super().__init__()
        self.n_head = n_head
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_head, dim))

    @staticmethod
    def rms_norm(x, eps=1e-5):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., n_head, dim)
        """
        normed = self.rms_norm(x.float(), self.eps).type_as(x)
        return normed * self.weight

NORM = RMSNorm
    
class FixedVAR(nn.Module):
    def __init__(self, config, max_len=128):
        super(FixedVAR, self).__init__()
        self.D = config.n_embd
        self.n_head = config.n_head

        assert self.D % self.n_head == 0
        self.d = self.D // self.n_head
        self.C = config.enc_in
        self.fixed_q = nn.Parameter(0.02*torch.randn(1, self.C, max_len, self.n_head, self.d))
        self.fixed_v = nn.Parameter(0.02*torch.randn(1, self.C, max_len, self.n_head, self.d))
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, X, *args, **kwargs):
        B, L, D = X.shape
        
        K = X.reshape(B, L, self.n_head, -1)
        
        b = B // self.C
        
        Q = self.fixed_q[:, :, 0:L].expand(b, -1, -1, -1, -1).reshape(B, L, self.n_head, self.d)
        
        V = self.fixed_v[:, :, 0:L].expand(b, -1, -1, -1, -1).reshape(B, L, self.n_head, self.d)
        
        W = torch.einsum('blhd,blhe->blhde', K, V)         # B L H d d
        W = torch.cumsum(W, dim=1)
        X = torch.einsum('blhd,blhde->blhe', Q, W)
        X = X.reshape(B, L, D)
        X = self.dropout(X)
        return X

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
    
    def collect_qkv(self, X, qv_input=None):
        B, L, D = X.shape
        K = X.reshape(B, L, self.n_head, -1)
        Q = self.q_proj(X if qv_input is None else qv_input).reshape(B, L, self.n_head, -1)
        Q = self.q_norm(Q)
        V = self.v_proj(X if qv_input is None else qv_input).reshape(B, L, self.n_head, -1) 
        V = self.v_norm(V)
        return Q, K, V

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
    
class BatchedInvertibleMatrix(nn.Module):
    def __init__(self, h, d, scale=0.02):
        super(BatchedInvertibleMatrix, self).__init__()
        self.h = h
        self.d = d
        # Initialize parameters for h matrices without imposing structure
        scaling_factor = np.sqrt(scale / np.sqrt(d))
        self.LU = nn.Parameter(scaling_factor * torch.randn(h, d, d))

    def forward(self, get_inverse=True, get_LU=False):
        # Enforce lower triangular structure with ones on the diagonal for each L
        L = torch.tril(self.LU, -1) + torch.eye(self.d, device=self.LU.device).unsqueeze(0).expand(self.h, -1, -1)

        # Enforce upper triangular structure for each U and apply Softplus to the diagonal
        U = torch.triu(self.LU, 1)  # Keep the strictly upper triangular part
        diag_elements = F.softplus(torch.diagonal(self.LU, dim1=-2, dim2=-1))
        U = U + torch.diag_embed(diag_elements)

        if get_LU:
            return L, U

        if not get_inverse:
            return torch.matmul(L, U)

        # Create identity matrix for solving, repeated for each matrix in the batch
        identity = torch.eye(self.d, device=self.LU.device).unsqueeze(0).expand(self.h, -1, -1)

        # Compute L_inv using forward substitution for each matrix in the batch
        L_inv = torch.linalg.solve_triangular(L, identity, upper=False)

        # Compute U_inv using back substitution for each matrix in the batch
        U_inv = torch.linalg.solve_triangular(U, identity, upper=True)

        # Compute the product and its inverse
        LU = torch.matmul(L, U)
        LU_inv = torch.matmul(U_inv, L_inv)

        return LU, LU_inv

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
    
    def collect_qkv(self, x):
        q_list, k_list, v_list = [], [], []
        B, L, D = x.shape
        for ln, mlp in zip(self.lns, self.mlps):
            x = x + mlp(ln(x))

        BC, N, d = x.shape
        x = self.VAR_input_norm(x)
        x_orig = x.clone()
        attn_res = [x_orig.view(BC, N//2, 2, d)[:, :, 1, :]]
        index = len(self.attn)

        S = self.generate_S(get_inverse=False)
        for attn, pn in zip(self.attn, self.pns):
            index -= 1
            q, k, v = attn.collect_qkv(x, qv_input=x_orig)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)
            x = attn(x, qv_input=x_orig) #x + attn(x)
            current_res = torch.einsum('blhd,hde->blhe', x.reshape(BC, N, self.n_head, self.d), S).reshape(BC, N, d)
            attn_res.append(current_res.view(BC, N//2, 2, d)[:, :, 1, :])
        
        return torch.stack(q_list), torch.stack(k_list), torch.stack(v_list)

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
        self.max_len = (self.seq_len + pad_len) // self.pred_len
        configs.max_len = self.max_len
        self.time_emb_dim = configs.time_emb_dim
        self.enc_in = configs.enc_in + configs.time_emb_dim
        self.number_of_targets = configs.number_of_targets if configs.number_of_targets else configs.enc_in
        n_series = self.number_of_targets
        self.n_series = n_series
        self.d_input = int(np.sqrt(self.enc_in)) * 32 if not configs.d_model else configs.d_model

        # Use ARX Tokenization
        self.tokenizer = MultiSegmentTokenizer(s=self.pred_len, n_channels=self.enc_in, d_model=self.d_input, number_of_targets=self.number_of_targets, univariate=configs.univariate)
        self.dropout = nn.Dropout(configs.dropout)
        self.n_head = self.d_input // 16 if not configs.n_heads else configs.n_heads
        self.d = self.d_input // self.n_head
        self.pre_norm = NORM(1*self.d_input)
        self.final_norm = NORM(1*self.d_input)
        self.local_attention = getattr(configs, "local_attention", False)   # Disabled
        self.local_attention_window = getattr(configs, "local_attention_window", 8)
        transformer_config = SimpleNamespace(n_layer=configs.e_layers, n_embd=1*self.d_input, n_head=self.n_head, dropout=configs.dropout, 
                                             bias=True, max_len=self.max_len, predictor=configs.predictor, decay=True, enc_in=self.number_of_targets,
                                             local_attention=self.local_attention, local_attention_window=self.local_attention_window)
        self.transformer = Transformer(transformer_config)
        self.pos_emb = nn.Embedding(self.max_len*2, 1*self.d_input)
        self.pos_emb_2 = nn.Parameter(0.02*torch.randn(1, 1, 2, self.d_input))
        self.channel_emb = nn.Parameter(0.02*torch.randn(1, self.number_of_targets, self.max_len*2, 1*self.d_input))
        self.channel_emb_pos = nn.Parameter(0.02*torch.randn(1, self.number_of_targets, 2, 1*self.d_input))

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

    def forward(self, x, x_mark=None, y=None, *args, **kwargs):
        # x: [Batch, Input length, Channel]
        if x_mark is not None and self.time_emb_dim:
            x = torch.cat([x_mark, x], dim=-1)
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
        pos = torch.arange(0, N//2, dtype=torch.long, device=x.device).repeat_interleave(2)
        pos_emb = self.pos_emb(pos)
        pos_emb = pos_emb + self.channel_emb[:, :, 0:N, :] + self.pos_emb_2.tile((1, 1, N//2, 1)) + self.channel_emb_pos.tile((1, 1, N//2, 1))
        x = self.dropout(x) + pos_emb  # B C N d
        x = self.pre_norm(x)

        # AR/ARMA Transformer
        x = x.view(B*C, N, d)
        x = self.transformer(x) # B C N//2 d

        # Output Projection
        x = self.final_norm(x)     # B*Ct N d
        x = x.view(B, C, N//2, d)
        x = self.tokenizer.inverse_tokenize(x)  # B Ct N s

        # Next-step prediction loss
        if self.training:
            if self.local_attention:
                x = x[:, :, self.local_attention_window:, :]
                x_segmented_target = x_segmented_target[:, :, self.local_attention_window:, :]
                loss = F.mse_loss(x[:, :, 0:-1, :], x_segmented_target[:, :, 1:, :])
            else:
                loss = F.mse_loss(x[:, :, 0:-1, :], x_segmented_target[:, :, 1:, :])

        # Next-step output
        x = x[:, :, -1, :]         # B Ct s

        # RevIN: inverse processing
        x = (x*std[:, -C:] + mean[:, -C:]).permute(0, 2, 1)                  # B s Ct

        if self.training:
            return x, loss
        else:
            return x
        
    def collect_qkv(self, x, x_mark=None):
        with torch.no_grad():
            if x_mark is not None and self.time_emb_dim:
                x = torch.cat([x_mark, x], dim=-1)
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
            pos = torch.arange(0, N//2, dtype=torch.long, device=x.device).repeat_interleave(2)
            pos_emb = self.pos_emb(pos)
            pos_emb = pos_emb + self.channel_emb[:, :, 0:N, :] + self.pos_emb_2.tile((1, 1, N//2, 1)) + self.channel_emb_pos.tile((1, 1, N//2, 1))
            x = self.dropout(x) + pos_emb  # B C N d
            x = self.pre_norm(x)

            # AR/ARMA Transformer
            x = x.view(B*C, N, d)
            q, k, v = self.transformer.collect_qkv(x)
            return q, k, v
