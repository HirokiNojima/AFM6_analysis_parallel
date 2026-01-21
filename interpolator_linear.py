import numpy as np
from scipy.interpolate import NearestNDInterpolator

def afm_to_grid_linear(x_sensor, y_sensor, values, pixel_shape=(256, 256)):
    """
    AFMデータ用 最近傍法補間関数。
    
    1. 入力データのNaN位置を記録して保持する
    2. 各グリッド点を、最も近いデータ点の値で埋める
    3. 入力に含まれていたNaNは補間後も保たれる
    
    【特徴】
    1. メイン処理: 最近傍法による補間（計算高速、不連続特性を保存）。
    2. 入力データ品質を保証: 元々のNaNデータは一貫性を保つ。

    Parameters:
    ----------
    x_sensor, y_sensor : array-like
        センサー座標データ
    values : array-like
        測定値 (高さデータなど)。NaN値を含む場合がある。
    pixel_shape : tuple
        出力画像サイズ (Height, Width)
        
    Returns:
    -------
    grid_z : ndarray
        補間後の2D格子データ (float32)
    """
    # 1. 入力データのサニタイズ（1次元化）
    x = np.asarray(x_sensor).ravel()
    y = np.asarray(y_sensor).ravel()
    z = np.asarray(values).ravel()

    # 【改良】入力データのNaN位置を記録
    input_nan_mask = np.isnan(z)
    n_input_nans = input_nan_mask.sum()
    
    # 補間用には、NaNを含まない有効なデータのみを使用
    valid_mask = ~input_nan_mask
    if valid_mask.sum() == 0:
        raise ValueError("有効なデータ点がありません（すべてNaN）。")
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = z[valid_mask]

    # (N, 2) の座標配列を作成（有効なデータのみ）
    points = np.column_stack((x_valid, y_valid))

    # 2. ターゲットグリッド座標の作成
    x_min, x_max = np.min(x_valid), np.max(x_valid)
    y_min, y_max = np.min(y_valid), np.max(y_valid)
    
    grid_x = np.linspace(x_min, x_max, pixel_shape[1])
    grid_y = np.linspace(y_min, y_max, pixel_shape[0])
    
    # メッシュグリッド作成
    xx, yy = np.meshgrid(grid_x, grid_y)

    # 3. 最近傍法補間の実行
    interp_nearest = NearestNDInterpolator(points, z_valid)
    grid_z = interp_nearest(xx, yy)

    # 【改良】入力データのNaN位置の再現
    # 元データのNaN位置を、グリッド上で推定してNaN化する
    if n_input_nans > 0:
        # 入力データのNaN位置の座標を取得
        x_input_nans = x[input_nan_mask]
        y_input_nans = y[input_nan_mask]
        
        # グリッド座標に変換（インデックス）
        x_indices = np.digitize(x_input_nans, grid_x) - 1
        y_indices = np.digitize(y_input_nans, grid_y) - 1
        
        # グリッド範囲内の点だけをマスク処理
        valid_grid_mask = (
            (x_indices >= 0) & (x_indices < pixel_shape[1]) &
            (y_indices >= 0) & (y_indices < pixel_shape[0])
        )
        
        x_indices = x_indices[valid_grid_mask]
        y_indices = y_indices[valid_grid_mask]
        
        # グリッド上でそれらの位置をNaNに設定
        grid_z[y_indices, x_indices] = np.nan

    return grid_z.astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # --- テスト: 急峻な段差（エッジ）の再現性確認 ---
    np.random.seed(42)
    N_samples = 5000
    
    # ランダムな座標
    X_sample = np.random.uniform(0, 10, N_samples)
    Y_sample = np.random.uniform(0, 10, N_samples)
    
    # ステップ関数を作る（Xが5より大きいと高さが1.0、それ以外は0.0）
    # RBFやIDWだと、この「崖」がなだらかな坂になってしまうが、線形補間なら「崖」として描画される。
    Z_sample = np.where(X_sample > 5.0, 1.0, 0.0) 
    
    # さらに微細な突起を追加 (スパイクノイズのようなもの)
    mask_spike = (X_sample - 2.5)**2 + (Y_sample - 2.5)**2 < 0.05
    Z_sample[mask_spike] = 2.0

    # 補間実行
    img = afm_to_grid_linear(X_sample, Y_sample, Z_sample, pixel_shape=(300, 300))

    # --- 結果表示 ---
    plt.figure(figsize=(10, 8))
    
    # NaN（データがない外側）を目立たせるために背景色を設定
    current_cmap = plt.cm.viridis
    current_cmap.set_bad(color='black') # データ外は黒にする

    plt.imshow(img, extent=(0, 10, 0, 10), origin='lower', cmap=current_cmap, interpolation='nearest')
    plt.colorbar(label='Height')
    plt.title('Nearest Neighbor Interpolation Result\n(NaN preservation)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.show()

# アプリ用にクラスを記述。本来はアプリ側を変更するべき。
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
