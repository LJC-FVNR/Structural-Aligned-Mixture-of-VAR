from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm.auto import tqdm
from joblib import Parallel, delayed
    
    
def generate_var_sample(
    index,
    C,
    L,
    N,
    var_coef_range,
    noise_std,
    do_normalize,
    clip_value=15,
    p_down=0.0,
    p_up=0.0
):
    np.random.seed(index)

    flip = np.random.rand()
    if flip < p_down:
        L_gen = np.random.randint(L + 1, int(1.5 * L) + 1)
    elif flip < p_down + p_up:
        min_len = max(N, 3)
        if min_len >= L:
            L_gen = L
        else:
            L_gen = np.random.randint(min_len, L)
    else:
        L_gen = L

    var_data = np.zeros((C, L_gen), dtype=np.float32)
    for t in range(N):
        var_data[:, t] = np.random.normal(0, 1.0, size=(C,))

    var_mats = [
        np.random.uniform(var_coef_range[0], var_coef_range[1], size=(C, C)).astype(np.float32)
        for _ in range(N)
    ]

    for t in range(N, L_gen):
        noise = np.random.normal(0, noise_std, size=(C,))
        val_t = np.zeros(C, dtype=np.float32)

        for j in range(N):
            val_t += var_mats[j].dot(var_data[:, t - 1 - j])
        val_t += noise
        var_data[:, t] = val_t

    if L_gen != L:
        x_old = np.linspace(0, 1, L_gen)
        x_new = np.linspace(0, 1, L)
        new_data = np.zeros((C, L), dtype=np.float32)
        for i in range(C):
            new_data[i, :] = np.interp(x_new, x_old, var_data[i, :])
        var_data = new_data  # (C, L)

    if do_normalize:
        for i in range(C):
            mean_ = var_data[i].mean()
            std_ = var_data[i].std() + 1e-8
            var_data[i] = (var_data[i] - mean_) / std_

    np.clip(var_data, -clip_value, clip_value, out=var_data)
    var_data[~np.isfinite(var_data)] = 0.0

    var_data = var_data.T

    return var_data, var_mats

class VARNDataset(Dataset):
    def __init__(
        self,
        C,
        L_I,
        L_P,
        noise_std=0.05,
        ds_type='train', 
        do_normalize=False,
        n_jobs=-1,
        clip_value=15
    ):
        super().__init__()
        if ds_type == "train":
            N = (1, 3)
            var_coef_range = (-0.5, 0.5)
            dataset_size = 20000
        elif ds_type == "val":
            N = (3, 5)
            var_coef_range = (-0.25, 0.25)
            dataset_size = 2000
        self.C = C
        self.L = L_I + L_P
        self.L_I = L_I
        self.L_P = L_P
        self.N = N
        self.var_coef_range = var_coef_range
        self.noise_std = noise_std
        self.dataset_size = dataset_size
        self.do_normalize = do_normalize
        self.n_jobs = n_jobs
        self.clip_value = clip_value
        self.get_index = False

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(generate_var_sample)(
                i,
                self.C,
                self.L,
                np.random.randint(self.N[0], self.N[1]),#self.N,
                self.var_coef_range,
                self.noise_std,
                self.do_normalize,
                self.clip_value
            ) for i in tqdm(range(self.dataset_size))
        )

        all_data_list = []
        all_mats_list = []
        for (var_data, var_mats) in results:
            all_data_list.append(var_data)
            all_mats_list.append(var_mats)

        # (dataset_size, L, C)
        self.all_data = torch.from_numpy(np.stack(all_data_list, axis=0))
        self.all_mats = all_mats_list

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        data = self.all_data[index]  # shape = (L, C)
        X = data[:self.L_I]          # (L_I, C)
        Y = data[-self.L_P:]         # (L_P, C)
        if not self.get_index:
            return X, Y
        else:
            return X, Y, index
            
    def get_var_weights(self, index):
        return self.all_mats[index]