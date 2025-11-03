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
        Interpolates scattered points (X, Y, Z) to a regular 2D grid.

        Parameters
        ----------
        X, Y, Z : np.ndarray
            1D arrays of same length representing irregular sample positions and values.

        Returns
        -------
        Z_grid : np.ndarray
            2D numpy array of shape grid_size containing interpolated values.
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

        # 🌟 4. Epsilon の自動計算ロジック
        if self.epsilon_mode == 'auto':
            # 'auto' の場合:
            # グリッドポイントから見て、最も近い生データ点までの距離の中央値を計算
            median_dist_to_nearest = torch.median(dists_t[:, 0])
            # その 3.0 倍を epsilon として使用
            eps = median_dist_to_nearest * 3.0
            print(f"Auto-epsilon set: 3.0 * median_dist_to_nearest (3.0 * {median_dist_to_nearest:.4f}) = {eps:.4f}")
        else:
            # 'auto' でない場合 (数値が指定された場合)
            eps = float(self.epsilon_mode)

        # 5. Gaussian RBF weights: exp(-(d^2 / eps^2))
        # (以前は 'eps = self.epsilon' だったのを 'eps' 変数を使うように変更)
        weights = torch.exp(-(dists_t / eps) ** 2)

        # 6. Gather neighbor values
        local_vals = values_t[idxs_t]

        # 7. Weighted interpolation: sum(w*z) / sum(w)
        Z_interp = (weights * local_vals).sum(dim=1) / weights.sum(dim=1)

        # 7. Reshape back to 2D grid (numpy)
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