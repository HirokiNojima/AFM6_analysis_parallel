#%%
import numpy as np
from scipy.optimize import curve_fit

def gaussian_decay(x, sigma):
    """フィッティング用のガウス関数型減衰モデル"""
    return np.exp(-(x / sigma)**2)

def calculate_row_half_widths_fitting(data_2d, chunk_size=1):
    """
    2次元配列を任意の行数(chunk_size)ごとに区切り、縦方向に平均化したプロファイルに対して
    自己相関が0.5に減衰するまでのラグ(半値幅)をモデルフィッティングを用いて計算する
    
    Parameters:
        data_2d (ndarray): 2次元の入力データ
        chunk_size (int): 何行ごとに区切って計算するか（デフォルトは1行ごと）
    """
    n_rows, n_cols = data_2d.shape
    half_widths = []
    
    # 可視化用の保存変数
    example_corr = None
    example_lag05 = None
    example_fit_curve = None

    # chunk_size ごとにループを回す
    for i in range(0, n_rows, chunk_size):
        # 指定した行数のブロックを抽出（配列の終端は自動的に余りの行数になります）
        chunk = data_2d[i : i + chunk_size]
        
        # 縦方向（axis=0）に平均化し、1次元の代表プロファイルを作成
        row_profile = np.mean(chunk, axis=0)
        
        # DC成分の除去
        row_centered = row_profile - np.mean(row_profile)
        
        # 自己相関の計算と右半分の取得
        corr_full = np.correlate(row_centered, row_centered, mode='full')
        mid = len(corr_full) // 2
        corr_right = corr_full[mid:]
        
        # エッジケース: データが完全に平坦な場合
        if corr_right[0] == 0:
            half_widths.append(0.0)
            if i == 0:
                example_corr = np.zeros(n_cols)
                example_corr[0] = 1.0
                example_lag05 = 0.0
                example_fit_curve = np.zeros(n_cols)
            continue
            
        # 最大値を1.0に正規化
        corr_norm = corr_right / corr_right[0]
        
        # --- フィッティング区間（メインローブ）の抽出 ---
        zero_crossings = np.where(corr_norm < 0)[0]
        if len(zero_crossings) > 0:
            fit_end_idx = zero_crossings[0]
        else:
            fit_end_idx = len(corr_norm)
            
        fit_end_idx = min(max(3, fit_end_idx), len(corr_norm))
        
        x_data = np.arange(fit_end_idx)
        y_data = corr_norm[:fit_end_idx]
        
        sigma_val = None # ループを跨いだ値の引継ぎを防ぐための初期化
        
        # --- モデルフィッティングの実行 ---
        try:
            popt, _ = curve_fit(gaussian_decay, x_data, y_data, p0=[10.0], bounds=(0, np.inf))
            sigma_val = popt[0]
            
            # 理論曲線が0.5になるxを解析的に計算
            lag_at_05 = float(sigma_val * np.sqrt(-np.log(0.5)))
            
        except RuntimeError:
            # フィッティングが収束しなかった場合のフォールバック（線形補間）
            idx_below_05 = np.where(corr_norm < 0.5)[0]
            if len(idx_below_05) > 0:
                idx = idx_below_05[0]
                if idx == 0:
                    lag_at_05 = 0.0
                else:
                    y0 = corr_norm[idx - 1]
                    y1 = corr_norm[idx]
                    fraction = (0.5 - y0) / (y1 - y0)
                    lag_at_05 = float((idx - 1) + fraction)
            else:
                lag_at_05 = float(n_cols)
                
        half_widths.append(lag_at_05)
        
        # 初回ループ時（最初のチャンク）に可視化用データを保存
        if i == 0:
            example_corr = corr_norm
            example_lag05 = lag_at_05
            if sigma_val is not None:
                example_fit_curve = gaussian_decay(np.arange(len(corr_norm)), sigma_val)
            else:
                example_fit_curve = None
            
    return np.array(half_widths), example_corr, example_lag05, example_fit_curve

# 各行の標準偏差を計算する関数
def calculate_row_std(data_2d):
    return np.std(data_2d, axis=1)


#%%
import matplotlib.pyplot as plt
filename = r"C:\nojima\AFM6measurement\260324\1305_スイープ計測\AFM_Analysis_Results\topography_map.npz"
data = np.load(filename)['map_data']

# --- 計算実行 ---
half_widths, ex_corr, ex_lag, ex_fit = calculate_row_half_widths_fitting(data, chunk_size=10)
std_devs = calculate_row_std(data)

# --- プロット ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：概念説明（1行目の自己相関と0.5の交点）
lags = np.arange(len(ex_corr))
axes[0].plot(lags, ex_corr, label='Normalized Autocorr (Row 0)')
axes[0].axhline(0.5, color='red', linestyle='--', label='Half Value (0.5)')
axes[0].axvline(ex_lag, color='green', linestyle=':', label=f'Half-Width: {ex_lag:.2f}')
axes[0].set_title("How Half-Width is calculated (Row 0)")
axes[0].set_xlabel("Lag (samples)")
axes[0].set_ylabel("Correlation Value")
axes[0].set_xlim(0, 50) # 見やすくするために表示範囲を制限
axes[0].legend()
axes[0].grid(True)

# 右：全行の半値幅の結果
axes[1].plot(half_widths, 'o-', color='purple')
axes[1].set_title("Half-Width for each Row")
axes[1].set_xlabel("Row Index")
axes[1].set_ylabel("Lag to reach 0.5 (samples)")
axes[1].grid(True)

plt.tight_layout()
plt.show()
# %%
