# %%
import numpy as np
import matplotlib.pyplot as plt

def calc_touch_point(force: np.ndarray, z_distance: np.ndarray) -> int:
    """
    ベースライン補正後のフォースカーブから、
    最も静かな領域をベースラインと定義し、
    そこから逸脱する点をコンタクトポイントとして検出する。

    Parameters
    ----------
    force : np.ndarray
        ベースライン補正済みのフォースデータ（アプローチカーブ）。
    z_distance : np.ndarray
        Z距離データ（アプローチカーブ）。

    Returns
    -------
    int
        検出されたコンタクトポイントのインデックス。
    """
    n_points = len(force)
    if n_points < 50: # データ点が少なすぎる
        return np.argmax(force) # フォールバック: 最大点をCPとする (苦肉の策)

    # --- 1. 真のベースライン領域を特定 ---
    # ノイズ評価ウィンドウサイズ (カーブ長の30% or 50点)
    window_size = max(50, int(n_points * 0.3))
    if window_size >= n_points:
        window_size = n_points // 2

    # 最も静かな領域の「開始インデックス」を取得
    baseline_start_idx = _find_quietest_region_idx(force, window_size)
    baseline_end_idx = min(baseline_start_idx + window_size, n_points - 1)

    # --- 2. ベースライン領域のノイズレベルを計算 ---
    baseline_data = force[baseline_start_idx:baseline_end_idx]
    if len(baseline_data) < 10:
        return np.argmax(force) # フォールバック

    noise_std = np.std(baseline_data)
    noise_mean = np.mean(baseline_data)
    
    # 閾値を設定 (平均 + 5σ)
    # 5シグマと高めに設定し、ノイズ領域での誤検出を防ぐ
    threshold = noise_mean + 5.0 * noise_std 
    
    # 平滑化 (任意だが推奨)
    try:
        force_smooth = uniform_filter1d(force.astype(np.float64), size=5, mode='reflect')
    except:
        force_smooth = force

    # --- 3. 閾値を超えた最初の点を探す ---
    
    # 検索開始位置: ベースライン領域の「終了位置」
    search_start_idx = baseline_end_idx

    # ベースライン終了位置以降で、閾値を「上回った」点のインデックスを探す
    candidates = np.where(force_smooth[search_start_idx:] > threshold)[0]
    
    if len(candidates) > 0:
        # 候補が見つかった場合
        # candidates[0] は search_start_idx からの相対インデックス
        touch_index = candidates[0] + search_start_idx
    else:
        # 閾値を超える点が見つからなかった場合
        # (例: 接触しなかった or ベースラインがカーブの最後だった)
        # フォールバック: 最大押し込み点（＝カーブの最後）をCPとする
        touch_index = n_points - 1
    return touch_index

if __name__ == "__main__":
    # テスト用のダミーデータ
    # z_distance = np.linspace(0, 100, 500)
    # force = np.piecewise(z_distance, [z_distance < 30, (z_distance >= 30) & (z_distance < 70), z_distance >= 70],
    #                      [lambda z: 0.1 * z + np.random.normal(0, 0.5, z.shape),
    #                       lambda z: 3 + np.random.normal(0, 0.5, z.shape),
    #                       lambda z: 0.2 * z - 11 + np.random.normal(0, 0.5, z.shape)])
    data = np.loadtxt(r"C:\Users\icell\Desktop\nojima_python\AFM6analysis_20251024\testdata\approach.txt")
    force = data[:,0]
    z_distance = data[:,1]

    touch_index = calc_touch_point(force, z_distance)
    plt.plot(z_distance, force, label="Force Curve")
    plt.plot(z_distance[touch_index], force[touch_index], 'ro', label="Touch Point")
    plt.show()
    # 2025年10月29日：作成完了。

# %%
import numpy as np
def correct_hyst(ztip , hyst_curve):
    """
    マスターカーブによりヒステリシス補正を行う。
    2分割した後、それぞれの分岐で最も近いZ位置を探し補正を行う。
    Parameters
    ----------
    ztip : numpy.ndarray
        ztip入力電圧。[V]
    hyst_curve : numpy.ndarray
        ヒステリシス補正用マスターカーブ。[Z位置, ztip電圧]

    Returns
    -------
    numpy.ndarray
        補正後のフォースカーブデータ。[m]
    """
    
    # === FC_data側の分岐 ===
    fc_maxZidx = np.argmax(ztip)  # FC_data用
    app_data = ztip[:fc_maxZidx + 1]
    ret_data = ztip[fc_maxZidx:]
    # === hyst_curve側の分岐 ===
    hyst_maxZidx = np.argmax(hyst_curve[:, 1])  # 別名に！
    app_hyst_curve = hyst_curve[:hyst_maxZidx + 1]
    ret_hyst_curve = hyst_curve[hyst_maxZidx:]

    # === アプローチ側のZ補正 ===
    closest_Z = app_hyst_curve[:, 0][
        np.abs(app_hyst_curve[:, 1] - app_data[:, np.newaxis]).argmin(axis=1)
    ]
    app_hyst_data = closest_Z

    # === リトラクト側のZ補正 ===
    closest_Z = ret_hyst_curve[:, 0][
        np.abs(ret_hyst_curve[:, 1] - ret_data[:, np.newaxis]).argmin(axis=1)
    ]
    ret_hyst_data = closest_Z

    # === 補正データの統合 ===
    corrected_data = np.empty_like(ztip)
    corrected_data[:fc_maxZidx + 1] = app_hyst_data
    corrected_data[fc_maxZidx:] = ret_hyst_data

    # === 補正データの単位変換 ===
    corrected_data = corrected_data * 1e-9  # nm -> m

    return corrected_data

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # テスト
    ztip = np.loadtxt(r"C:\Users\icell\Desktop\nojima_python\AFM6analysis_20251024\testdata\FCdata.txt")
    hyst_curve = np.loadtxt(r"C:\Users\icell\Desktop\nojima_python\git\AFM6_analysis_program\module\補正用データ\3kHz\mean_FCdata.txt")
    corrected_data = correct_hyst(ztip[:, 1], hyst_curve)
    plt.plot(corrected_data, ztip[:, 0], label="Corrected Data")
    plt.show()
    # 2025年10月30日：作成完了。

# %%
def devide_app_ret(force, ztip):
    """
    フォースカーブデータをアプローチとリトラクトに分割する。aa

    Parameters
    ----------
    force : numpy.ndarray
        フォースカーブデータ。
    ztip : numpy.ndarray
        Z位置データ。

    Returns
    -------
    tuple
        アプローチデータとリトラクトデータのタプル。
    """
    fc_maxZidx = np.argmax(ztip)  # Z位置データの最大値インデックス
    app_f = force[:fc_maxZidx]
    ret_f = force[fc_maxZidx:]
    app_z_dis = ztip[:fc_maxZidx]
    ret_z_dis = ztip[fc_maxZidx:]

    return [app_f, app_z_dis], [ret_f, ret_z_dis]
    # 2025年10月30日：作成完了

# %%
import numpy as np
from scipy.ndimage import uniform_filter1d # 移動平均・標準偏差に高速
def _find_quietest_region_idx(force: np.ndarray, window_size: int) -> int:
    """
    移動標準偏差を使い、最もノイズが小さい領域の開始インデックスを返す。
    
    Parameters
    ----------
    force : np.ndarray
        力データ
    window_size : int
        ノイズレベルを評価するためのウィンドウサイズ

    Returns
    -------
    int
        最も標準偏差が小さかったウィンドウの開始インデックス
    """
    if len(force) < window_size:
        return 0 # データがウィンドウより短い場合は最初を返す

    # 高速な移動平均と移動分散の計算
    # (force**2)の移動平均 - (forceの移動平均)**2 = 移動分散
    force_sq_mean = uniform_filter1d(force.astype(np.float64)**2, size=window_size, mode='reflect')
    force_mean = uniform_filter1d(force.astype(np.float64), size=window_size, mode='reflect')
    moving_var = force_sq_mean - force_mean**2
    
    # 負の値は数値誤差なので0にする
    moving_var[moving_var < 0] = 0
    
    # 移動標準偏差
    moving_std = np.sqrt(moving_var)

    # 最小の標準偏差を持つ領域のインデックスを返す
    # 端点を避けるため、少し内側で探す
    search_margin = window_size // 2
    if len(moving_std) < search_margin * 2 + 1:
         return np.argmin(moving_std)

    min_std_idx_center = np.argmin(moving_std[search_margin:-search_margin])
    min_std_idx_start = min_std_idx_center + search_margin - (window_size // 2)

    return max(0, min_std_idx_start)

# %%
def baseline_correction(force, z_distance, basefit_range_ratio=0.5):
    """
    フォースカーブのベースライン補正を行う。
    最も静かな（ノイズが小さい）領域を自動検出し、そこをベースラインとしてフィッティングする。
    """
    app, ret = devide_app_ret(force, z_distance)
    
    corrected_parts = []

    # --- アプローチ側の補正 (修正) ---
    if len(app[0]) > 20: # 十分なデータ点があるか
        # フィットに使う点数を決定 (カーブ長の30% or 50点, 多い方)
        window_size = max(50, int(len(app[0]) * 0.3))
        if window_size >= len(app[0]):
             window_size = len(app[0]) // 2

        # 最も静かな領域を探す
        fit_start_idx = _find_quietest_region_idx(app[0], window_size)
        fit_end_idx = min(fit_start_idx + window_size, len(app[0]) - 1)

        # 領域が狭すぎる場合のフォールバック
        if fit_end_idx - fit_start_idx < 10:
            fit_start_idx = int(0.1 * len(app[0]))
            fit_end_idx = int(0.5 * len(app[0]))

        x_fit = app[1][fit_start_idx:fit_end_idx]
        y_fit = app[0][fit_start_idx:fit_end_idx]
        
        base_coef = np.polyfit(x_fit, y_fit, 1)
        corrected = app[0].astype(float) - np.polyval(base_coef, app[1])
        corrected_parts.append(corrected)
    else:
        corrected_parts.append(app[0]) # 短すぎるカーブは補正しない

    # --- リトラクト側の補正 (同様に修正) ---
    if len(ret[0]) > 20:
        window_size = max(50, int(len(ret[0]) * 0.3))
        if window_size >= len(ret[0]):
            window_size = len(ret[0]) // 2

        fit_start_idx = _find_quietest_region_idx(ret[0], window_size)
        fit_end_idx = min(fit_start_idx + window_size, len(ret[0]) - 1)

        if fit_end_idx - fit_start_idx < 10:
            fit_start_idx = int(0.5 * len(ret[0]))
            fit_end_idx = int(0.9 * len(ret[0]))

        x_fit = ret[1][fit_start_idx:fit_end_idx]
        y_fit = ret[0][fit_start_idx:fit_end_idx]

        base_coef = np.polyfit(x_fit, y_fit, 1)
        corrected = ret[0].astype(float) - np.polyval(base_coef, ret[1])
        corrected_parts.append(corrected)
    else:
        corrected_parts.append(ret[0])

    force = np.concatenate(corrected_parts, axis=0)
    return force

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    # テスト用データ
    data = np.loadtxt(r"C:\Users\icell\Desktop\nojima_python\AFM6analysis_20251024\testdata\FCdata.txt")
    z_distance = data[:, 1]
    force = data[:, 0]
    force += 100 * z_distance  # ベースラインオフセットを追加

    force_corrected = baseline_correction(force.copy(), z_distance)

    plt.scatter(z_distance, force, label="Original Force")
    plt.scatter(z_distance, force_corrected, label="Corrected Force")
    plt.legend()
    plt.show()
    # 2025年10月30日　:確認完了
# %%
import math
def calc_young_snnedon(app_force, delta):
    """
    Sneddonモデルを用いてヤング率を計算する。
    Parameters
    ----------
    app_force : numpy.ndarray
        フォースカーブの一方向データ。コンタクトポイントから頂点までのデータ。[N]
    delta : numpy.ndarray
        押し込み量データ。コンタクトポイントから頂点までのデータ。[m]
    Returns
    -------
    float
        ヤング率。[Pa]
    """
    poisson_ratio = 0.5  # ポアソン比
    tip_half_angle = 17.5  # カンチレバー半角[°]
    A = (math.pi * (1 - poisson_ratio ** 2)) / (2 * math.tan(math.radians(tip_half_angle)))
    force_fit = abs(app_force) ** (1 / 2) # 直線化
    try:
        coe = np.polyfit(delta, force_fit, 1)
    except:
        coe = [1e-6, 0]
    youngs_modulus = A * coe[0] ** 2
    return youngs_modulus

# %%
def calc_peakforce(force):
    """
    ピークフォースを計算する。
    Parameters
    ----------
    force : numpy.ndarray
        フォースカーブ全データ[N]
    Returns
    -------
    float
        ピークフォース[N]
    """
    peak_force = np.max(force)
    return peak_force

# %%
def calc_hyst_area(force, z_distance, cont_idx, ret_idx):
    """
    ヒステリシスループ面積を計算する。ヒステリシスループ自体はカンチレバーのたわみを引いても引かなくても理論上は同じなので、ここでは引かない。
    Parameters
    ----------
    force : numpy.ndarray
        フォースカーブ全データ[N]
    z_distance : numpy.ndarray
        Z距離データ[m]
    cont_idx : int
        コンタクトポイントのインデックス
    ret_idx : int
        リトラクトポイントのインデックス
    Returns
    -------
    float
        ヒステリシスループ面積[J]
    """
    app_force = force[cont_idx:np.argmax(z_distance)]
    app_z = z_distance[cont_idx:np.argmax(z_distance)]
    ret_force = force[np.argmax(z_distance)+1:ret_idx]
    ret_z = z_distance[np.argmax(z_distance)+1:ret_idx]

    # ヒステリシスループ面積計算。np.trapzで減少関数を計算すると負の値になるため、足し算している。
    hyst_area = np.trapz(app_force, app_z) + np.trapz(ret_force, ret_z)
    return abs(hyst_area)
