import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def afm_to_grid_linear(x_sensor, y_sensor, values, pixel_shape=(256, 256)):
    """
    AFMデータ用 ハイブリッド補間関数。
    (Linear Interpolation + Nearest Neighbor Fill)
    
    【特徴】
    1. メイン処理: Delaunay三角分割による線形補間（微細構造を保存）。
    2. 外挿処理: 外側のNaN領域を最近傍法（Nearest Neighbor）で埋める。
       -> これにより、画像四隅の欠損を防ぎ、かつ急激な値を生成しません。

    Parameters:
    ----------
    x_sensor, y_sensor : array-like
        センサー座標データ
    values : array-like
        測定値 (高さデータなど)
    pixel_shape : tuple
        出力画像サイズ (Height, Width)
    """
    # 1. 入力データのサニタイズ（1次元化）
    x = np.asarray(x_sensor).ravel()
    y = np.asarray(y_sensor).ravel()
    z = np.asarray(values).ravel()

    # (N, 2) の座標配列を作成
    points = np.column_stack((x, y))

    # 2. ターゲットグリッド座標の作成
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    grid_x = np.linspace(x_min, x_max, pixel_shape[1])
    grid_y = np.linspace(y_min, y_max, pixel_shape[0])
    
    # メッシュグリッド作成
    xx, yy = np.meshgrid(grid_x, grid_y)

    # 3. 線形補間 (Delaunay三角分割) の実行
    # fill_value=np.nan にして、外挿領域を明確に区別します
    interp_linear = LinearNDInterpolator(points, z, fill_value=np.nan)
    grid_z = interp_linear(xx, yy)

    # 4. NaN領域（外挿部分）の穴埋め処理
    if np.isnan(grid_z).any():
        
        # NearestNDInterpolator は全領域で値を返せる（NaNにならない）
        interp_nearest = NearestNDInterpolator(points, z)
        grid_z_nearest = interp_nearest(xx, yy)
        
        # 線形補間が NaN だった場所だけ、Nearestの結果で上書きする
        nan_mask = np.isnan(grid_z)
        grid_z[nan_mask] = grid_z_nearest[nan_mask]

    return grid_z

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
    plt.title('Linear Interpolation Result\n(Sharp edges are preserved)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.show()