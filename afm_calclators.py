# %%
import numpy as np
import matplotlib.pyplot as plt
import warnings

def _make_dummy_fc(n_points=800, z_max=1e-6, noise=2e-9, seed=0):
    """
    ダミーのフォースカーブを作成する。テスト用。
    Parameters
    ----------
    n_points : int
        フォースカーブのデータ点数。
    z_max : float   
        Z距離の最大値。[m]
    noise : float
        フォースデータに加えるガウスノイズの標準偏差。[N]
    seed : int
        乱数シード。
    Returns
    -------
    tuple
        フォースデータとZ距離データのタプル。
    """
    rng = np.random.default_rng(seed)
    # Z軸: アプローチ→リトラクト
    z_app = np.linspace(0, z_max, n_points // 2, endpoint=True)
    z_ret = np.linspace(z_max, 0, n_points - len(z_app), endpoint=True)
    z = np.concatenate([z_app, z_ret])

    # 理想力: 接触前は0、接触後は線形＋少し曲げ
    contact_z = 0.3 * z_max
    k = 0.05  # 勾配 [N/m]
    force_app = np.where(z_app > contact_z, k * (z_app - contact_z), 0.0)
    force_ret = np.where(z_ret > contact_z, k * (z_ret - contact_z), 0.0)
    # 軽いヒステリシスを加える
    force_ret = force_ret * 0.9

    force = np.concatenate([force_app, force_ret])
    # ノイズ付加
    force += rng.normal(0, noise, size=force.shape)
    return force, z

def calc_touch_point(force: np.ndarray, z_distance: np.ndarray) -> int:
    """
    ベースライン補正後のフォースカーブを用いて、
    フォースカーブ開始地点と値最大点から作成した直線からの距離が最も遠い点を検出する。
    ノイズ対策のため、最初にローパスフィルタをかけて高周波成分を除去。

    Parameters
    ----------
    force : np.ndarray
        ベースライン補正済みのフォースデータ（片道）。
    z_distance : np.ndarray
        Z距離データ（片道）。

    Returns
    -------
    int
        検出されたコンタクトポイントのインデックス。
    """
    n_points = len(force)
    if n_points < 50: # データ点が少なすぎる
        return np.argmax(force) # フォールバック: 最大点をCPとする (苦肉の策)
    # ローパスフィルタ (移動平均)
    window_size = max(5, n_points // 50)  # データ点数の2%または5点、どちらか大きい方
    force = np.convolve(force, np.ones(window_size)/window_size, mode='same')
    
    # 直線の2点
    x1, y1 = z_distance[0], force[0]
    x2, y2 = z_distance[np.argmax(force)], force[np.argmax(force)]

    # 直線の方程式: Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # 分母（直線の長さ）の計算
    denominator = np.sqrt(A**2 + B**2)
    
    # ゼロ除算を防ぐ（2点が同じ位置にある場合のフォールバック）
    if denominator < 1e-10:
        # 2点が同じ位置 → 距離計算できないので最大値を返す
        touch_index = np.argmax(force)
    else:
        # 各点から直線までの距離を計算
        distances = np.abs(A * z_distance + B * force + C) / denominator
        touch_index = np.argmax(distances)

    return touch_index


if __name__ == "__main__":
    # テスト用のダミーデータ
    force, z_distance = _make_dummy_fc()
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
    force, ztip = _make_dummy_fc()
    plt.plot(ztip, force, label="Corrected Data")
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
    with warnings.catch_warnings(): # 多項式フィッティングエラーの警告を無視。
        warnings.simplefilter("ignore", np.RankWarning)
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
