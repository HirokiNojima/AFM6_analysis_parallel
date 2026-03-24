#%%
import numpy as np
import matplotlib.pyplot as plt
import os

from rich.pretty import data

path_folder = r"C:\nojima\AFM6measurement\260324\1038_100"
data_original= np.load(os.path.join(path_folder, "AFM_Analysis_Results", "topography_map.npz"), allow_pickle=True)['map_data'] * 1e9  # 単位をnmに変換

# %% 1次曲面フィッティングで全体の傾斜およびボウイングを補正する関数
def flatten_surface_1st_order(Z):
    """
    1次曲面フィッティングで全体の傾斜およびボウイングを補正する関数
    Z: 2D numpy array of shape (M, N) representing the height map
    return: 2D numpy array of shape (M, N) representing the flattened height map
    """
    M, N = Z.shape
    X, Y = np.meshgrid(np.arange(N), np.arange(M))
    
    # データを1次元配列に変換
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    
    # 1次曲面のモデル: Z = a*X + b*Y + c
    A = np.c_[X_flat, Y_flat, np.ones(X_flat.shape[0])]
    
    # 最小二乗法で係数を求める
    coeffs, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)
    
    # フィッティングした平面を計算
    Z_fit = (coeffs[0] * X + coeffs[1] * Y + coeffs[2])
    
    # 元のZからフィッティングした平面を引いて補正する
    Z_flattened = Z - Z_fit
    
    return Z_flattened.reshape(M, N)

data_1dflattened = flatten_surface_1st_order(data_original)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Topography")
plt.imshow(data_original, cmap='viridis')
plt.colorbar(label='Height (nm)')
plt.subplot(1, 2, 2)
plt.title("Flattened Topography (1st Order)")
plt.imshow(data_1dflattened, cmap='viridis')
plt.colorbar(label='Height (nm)')
plt.tight_layout()
plt.show()
# %%
data_analysis = np.load(r"C:\nojima\AFM6measurement\260324\1038_100\AFM_Analysis_Results\analysis_data.npz")
print(data_analysis.files)

# %% 2d_highpass filter
from scipy import signal
def highpass_filter_2d(Z, cutoff_frequency):
    """
    高速フーリエ変換を用いて高周波成分を抽出する関数
    Z: 2D numpy array of shape (M, N) representing the height map
    cutoff_frequency: カットオフ周波数 (0 < cutoff_frequency < 0.5)
    return: 2D numpy array of shape (M, N) representing the high-pass filtered height map
    """
    M, N = Z.shape
    
    # 2次元FFTを計算
    Z_fft = np.fft.fft2(Z)
    
    # 周波数のグリッドを作成
    u = np.fft.fftfreq(N)
    v = np.fft.fftfreq(M)
    U, V = np.meshgrid(u, v)
    
    # 周波数の大きさを計算
    D = np.sqrt(U**2 + V**2)
    
    # ハイパスフィルタを作成
    H = D > cutoff_frequency
    
    # フィルタを適用
    Z_fft_filtered = Z_fft * H
    
    # 逆FFTで空間領域に戻す
    Z_filtered = np.fft.ifft2(Z_fft_filtered).real
    
    return Z_filtered
data_highpass = highpass_filter_2d(data_1dflattened, cutoff_frequency=0.03)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Flattened Topography (1st Order)")
plt.imshow(data_1dflattened, cmap='viridis')
plt.colorbar(label='Height (nm)')
plt.subplot(1, 2, 2)
plt.title("High-pass Filtered Topography")
plt.imshow(data_highpass, cmap='viridis')
plt.colorbar(label='Height (nm)')
plt.tight_layout()
plt.show()
# %% 行方向の周波数のみのハイパスフィルタ
def highpass_filter_2d_directional(Z, cutoff_frequency = 0.005, direction='x'):
    """
    高速フーリエ変換を用いて特定の方向の高周波成分を抽出する関数
    Z: 2D numpy array of shape (M, N) representing the height map
    cutoff_frequency: カットオフ周波数 (0 < cutoff_frequency < 0.5)
    direction: 'x' または 'y' でフィルタリングする方向を指定
    return: 2D numpy array of shape (M, N) representing the directional high-pass filtered height map
    """
    M, N = Z.shape
    
    # 2次元FFTを計算
    Z_fft = np.fft.fft2(Z)
    
    # 周波数のグリッドを作成
    u = np.fft.fftfreq(N)
    v = np.fft.fftfreq(M)
    U, V = np.meshgrid(u, v)
    
    # 周波数の大きさを計算
    if direction == 'x':
        D = np.abs(U)
    elif direction == 'y':
        D = np.abs(V)
    else:
        raise ValueError("Direction must be 'x' or 'y'")
    
    # ハイパスフィルタを作成
    H = D > cutoff_frequency
    
    # フィルタを適用
    Z_fft_filtered = Z_fft * H
    
    # 逆FFTで空間領域に戻す
    Z_filtered = np.fft.ifft2(Z_fft_filtered).real
    
    return Z_filtered

cutoff_frequency_x = 0.005
cmap = 'afmhot_r'
data_highpass_x = highpass_filter_2d_directional(data_1dflattened, cutoff_frequency=cutoff_frequency_x, direction='x')
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.title("Flattened Topography (1st Order)")
plt.imshow(data_1dflattened, cmap=cmap)
plt.colorbar(label='Height (nm)')
plt.subplot(1, 2, 2)
plt.title(f"High-pass Filtered Topography (X)_cutoff: {cutoff_frequency_x}")
plt.imshow(data_highpass_x, cmap=cmap)
plt.colorbar(label='Height (nm)')
plt.tight_layout()
plt.show()
# %%
