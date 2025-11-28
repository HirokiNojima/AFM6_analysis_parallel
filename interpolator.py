# ファイル名: interpolator.py

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Union

class FastRBFInterpolator2D:
    """
    Approximate RBF interpolation from irregular (X, Y, Z) data
    onto a regular grid using local neighbor-based interpolation.
    
    大規模データセット（例: 1000万点）でも効率的に動作します。
    """

    def __init__(self, grid_size: Tuple[int, int] = (1024, 1024), neighbors: int = 64, epsilon: Union[float, str] = 'auto', device: str = None):
        """
        Parameters
        ----------
        grid_size : tuple
            (nx, ny) number of grid points in x and y directions.
        neighbors : int
            Number of nearest neighbors for local RBF interpolation.
        epsilon : float or 'auto'
            RBF kernel width parameter. 
            If 'auto', it's calculated based on data density (recommended).
        device : str or None
            'mps' (Apple GPU), 'cuda', or 'cpu'. If None, auto-detects available device.
        """
        self.grid_size = grid_size
        self.neighbors = neighbors
        self.epsilon_mode = epsilon # 🌟 'auto' または float値を保持
        # MPS/CUDAが利用可能かチェックし、デバイスを決定
        if device is None:
             if torch.backends.mps.is_available():
                 self.device = torch.device('mps')
             elif torch.cuda.is_available():
                 self.device = torch.device('cuda')
             else:
                 self.device = torch.device('cpu')
        else:
             self.device = torch.device(device)
        print(f"RBF Interpolator running on device: {self.device}")


def fit_transform(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Interpolates scattered points (X, Y, Z) to a regular 2D grid with NaN masking.
        """
        
        # 1. Build regular grid (出力画像グリッド)
        nx, ny = self.grid_size
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        Xg, Yg = np.meshgrid(xi, yi)
        grid_points = np.column_stack((Xg.ravel(), Yg.ravel()))

        # 2. Find K nearest neighbors for each grid point (CPU)
        nbrs = NearestNeighbors(n_neighbors=self.neighbors, algorithm='kd_tree', n_jobs=-1).fit(
            np.column_stack((X, Y))
        )
        print(f"Finding {self.neighbors} nearest neighbors for {len(grid_points)} grid points...")
        dists, idxs = nbrs.kneighbors(grid_points)

        # 3. Move data to Torch device (GPU/CPU) for fast computation
        dists_t = torch.tensor(dists, dtype=torch.float32, device=self.device)
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=self.device)
        values_t = torch.tensor(Z, dtype=torch.float32, device=self.device)
        
        print("Starting RBF weighted interpolation on device...")

        # 🌟 Mod: 距離の基準値（中央値）を先に計算（EpsilonとNaN判定の両方で使うため）
        # dists_t[:, 0] は各グリッド点から「最も近いデータ点」までの距離
        dist_to_nearest = dists_t[:, 0]
        median_dist_to_nearest = torch.median(dist_to_nearest)

        # 4. Epsilon の設定
        if self.epsilon_mode == 'auto':
            # 'auto' の場合: 最近傍距離の中央値の 3.0 倍
            eps = median_dist_to_nearest * 3.0
            print(f"Auto-epsilon set: {eps:.4f} (3.0 * median)")
        else:
            eps = float(self.epsilon_mode)

        # 5. Gaussian RBF weights: exp(-(d^2 / eps^2))
        weights = torch.exp(-(dists_t / eps) ** 2)

        # 6. Gather neighbor values
        local_vals = values_t[idxs_t]

        # 7. Weighted interpolation: sum(w*z) / sum(w)
        # ゼロ除算回避（念のため）
        denom = weights.sum(dim=1)
        denom[denom == 0] = 1e-9 
        Z_interp = (weights * local_vals).sum(dim=1) / denom

        # 8. 距離によるマスキング (NaN化)
        # self.max_dist が設定されている場合、しきい値を超えたらNaNにする
        
        # クラスの __init__ に self.max_dist = 'auto' または 数値 があると仮定
        # 未定義の場合は None として扱う
        max_dist_setting = getattr(self, 'max_dist', None) 

        nan_threshold = None
        if max_dist_setting == 'auto':
            # 自動設定: Epsilonより少し広め（例: 平均距離の4-5倍）を閾値にする
            nan_threshold = median_dist_to_nearest * 5.0
            print(f"Auto-masking threshold set: {nan_threshold:.4f} (5.0 * median)")
        elif max_dist_setting is not None:
            # 数値指定の場合
            nan_threshold = float(max_dist_setting)

        if nan_threshold is not None:
            # しきい値より遠いピクセルを NaN に上書き
            # (PyTorch上で処理するため高速)
            mask = dist_to_nearest > nan_threshold
            Z_interp[mask] = float('nan')
            print(f"Masked {mask.sum().item()} grid points as NaN.")

        # 9. Reshape back to 2D grid (numpy)
        Z_grid = Z_interp.cpu().numpy().reshape(self.grid_size).astype(np.float32)
        
        print("Interpolation complete.")
        return Z_grid

if __name__ == "__main__":
    # テストコード
    import matplotlib.pyplot as plt

    # サンプルデータ生成
    np.random.seed(0)
    N_samples = 30000
    X_sample = np.random.uniform(0, 10, N_samples)
    Y_sample = np.random.uniform(0, 10, N_samples)
    Z_sample = np.sin(X_sample) * np.cos(Y_sample) + 0.1 * np.random.randn(N_samples)

    # インターポレーター初期化
    rbf_interpolator = FastRBFInterpolator2D(grid_size=(512, 512), neighbors=64, epsilon=0.3)

    # 補間実行
    Z_grid = rbf_interpolator.fit_transform(X_sample, Y_sample, Z_sample)

    # 結果表示
    plt.imshow(Z_grid, extent=(0, 10, 0, 10), origin='lower')
    plt.scatter(X_sample, Y_sample, c='r', s=1, label='Sample Points')
    plt.colorbar(label='Interpolated Values')
    plt.title('RBF Interpolation Result')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()