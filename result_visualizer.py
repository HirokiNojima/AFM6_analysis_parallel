import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Tuple

from sympy import Trace
from afm_data import AFMData
from skimage.filters import threshold_otsu 
from scipy.ndimage import median_filter, gaussian_filter, shift
import interpolator_linear
from scipy.signal import correlate

class AFM_Result_Visualizer:
    """
    フォースマップ解析結果の可視化とエクスポートを担うクラス。
    カスタムカラーマップ、センサー座標の使用、1ラインスキャン検出による高解像度マップ処理統合を持つ。
    """

    def __init__(self):
        # 単位系に関する定数 (SI単位への変換)
        self.UNIT_CONVERSION = {
            'topography': 1e6,      # m -> µm
            'cp_z_position': 1e9,   # m -> nm
            'delta': 1e9,           # m -> nm
            'youngs_modulus': 1.0,  # Pa -> Pa (log(Pa)表示のため)
            'peak_force': 1e9,      # N -> nN
            'hysteresis_area': 1e15  # J (N·m) -> fJ
        }
        
        # カスタムプロット設定
        self.PLOT_CONFIG = {
            'youngs_modulus': {
                'cmap': 'afmhot',
                'label': "Young's Modulus (log(Pa))",
                'title': "Young's Modulus",
                'fname': "corrected_Young.png",
                'log_transform': True
            },
            'topography': {
                'cmap': 'afmhot',
                'label': "Height (µm)",
                'title': "Topography",
                'fname': "corrected_Topography.png",
                'log_transform': False
            },
            'peak_force': {
                'cmap': 'viridis',
                'label': "Peak Forces (nN)", 
                'title': "Peak Forces",
                'fname': "corrected_Peakforces.png",
                'log_transform': False
            },
            'delta': {
                'cmap': 'cividis',
                'label': "Delta (nm)", 
                'title': "Delta",
                'fname': "corrected_Delta.png",
                'log_transform': False
            },
            'cp_z_position': { 
                'cmap': 'viridis',
                'label': "CP Z Position (nm)",
                'title': "Contact Point Z",
                'fname': "corrected_cp_z_position.png",
                'log_transform': False
            },
            'hysteresis_area': {
                'cmap': 'seismic',
                'label': "Hysteresis Area (fJ)",
                'title': "Hysteresis Area",
                'fname': "corrected_hist_area.png",
                'log_transform': False
            }
        }
        
        self.DEFAULT_CONFIG = {
            'cmap': 'viridis', 
            'label': "Value (A.U.)",
            'title': "AFM Map",
            'fname': "map.png",
            'log_transform': False
        }

        self.best_lag = 0
        
    def _get_plot_config(self, property_key: str) -> Dict[str, Any]:
        """指定されたプロパティのプロット設定を取得する。"""
        return self.PLOT_CONFIG.get(property_key, self.DEFAULT_CONFIG)

    def _get_map_dimensions(self, data_list: List[AFMData]) -> tuple:
        """メタデータからマップのXStepとYStepを取得する。"""
        if not data_list:
            raise ValueError("データリストが空です。")
            
        # 破棄された metadata 参照ではなく、
        # オブジェクトにコピーされた XStep 属性を見る
        first_obj = data_list[0]
        nx = getattr(first_obj, 'XStep', 1) # 'XStep' 属性がなければ 1
        ny = getattr(first_obj, 'YStep', 1) # 'YStep' 属性がなければ 1

        if nx * ny != len(data_list):
            if np.sqrt(len(data_list)).is_integer():
                nx = ny = int(np.sqrt(len(data_list)))
            else:
                nx = len(data_list)
                ny = 1
        return nx, ny

    def _extract_physical_coords(self, data_list: List[AFMData]) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        AFMDataのxsensor, ysensorから物理座標を抽出する。
        Returns: X_coords_um, Y_coords_um, x_range_um, y_range_um
        """
        if not data_list:
            return np.array([]), np.array([]), 0.0, 0.0

        # AFMDataのxsensor, ysensor属性から直接座標を取得
        X_coords = np.array([data_obj.xsensor for data_obj in data_list])
        Y_coords = np.array([data_obj.ysensor for data_obj in data_list])

        # すべてをマイクロメートル [um] 単位に変換
        X_coords_um = X_coords * 1e6
        Y_coords_um = Y_coords * 1e6

        # スキャン範囲を計算 (最大値と最小値の差)
        x_range_um = X_coords_um.max() - X_coords_um.min()
        y_range_um = Y_coords_um.max() - Y_coords_um.min()
        
        # ゼロ範囲チェック
        if x_range_um < 1e-6: x_range_um = 1e-6
        if y_range_um < 1e-6: y_range_um = 1e-6

        return X_coords_um, Y_coords_um, x_range_um, y_range_um

    def _is_line_scan_by_range(self, x_range_um: float, y_range_um: float, threshold: float = 30.0) -> bool:
        """
        X-Y範囲の比率に基づいて、1ラインスキャンであるかを判定する。
        """
        if x_range_um < 1e-6 and y_range_um < 1e-6:
            return False

        min_range = min(x_range_um, y_range_um)
        max_range = max(x_range_um, y_range_um)
        
        if min_range < 1e-6:
            # どちらかの範囲が極端に小さい場合、最小分解能 (1 nm = 0.001 um) と仮定
            ratio = max_range / 0.001 
        else:
            ratio = max_range / min_range

        return ratio > threshold
    
    def _flatten_surface_2nd_order(self, data_2d):
        """
        2次元配列に対して2次曲面フィッティングを行い、全体の傾斜および
        ピエゾ特有のボウイング（湾曲）を補正する。
        
        Parameters:
            data_2d (np.ndarray): 補正前の2次元高さデータ (Height Map)
        
        Returns:
            np.ndarray: 補正後のデータ
        """
        # 入力データのコピーを作成
        corrected_data = data_2d.copy()
        
        # 画像のサイズを取得
        rows, cols = corrected_data.shape
        
        # X軸, Y軸の座標配列を作成
        x = np.arange(cols)
        y = np.arange(rows)
        X_mesh, Y_mesh = np.meshgrid(x, y)

        Z = corrected_data.astype(np.float64)
        Z_flat = Z.ravel()
        X_flat = X_mesh.ravel()
        Y_flat = Y_mesh.ravel()

        # NaNやInfを除外するマスクを作成
        mask = np.isfinite(Z_flat)
        
        # 2次曲面のフィッティングには最低6点の有効なデータが必要
        if mask.sum() < 6:
            return corrected_data

        # 計画行列の作成 (X^2, Y^2, XY, X, Y, 1)
        A = np.column_stack((
            X_flat[mask]**2, 
            Y_flat[mask]**2, 
            X_flat[mask] * Y_flat[mask], 
            X_flat[mask], 
            Y_flat[mask], 
            np.ones(mask.sum())
        ))
        
        # 最小二乗法で係数 (a, b, c, d, e, f) を算出
        coeffs, *_ = np.linalg.lstsq(A, Z_flat[mask], rcond=None)
        a, b, c, d, e, f = coeffs

        # 求められた係数から2次曲面モデルを構築
        surface = (a * X_mesh**2 + b * Y_mesh**2 + c * X_mesh * Y_mesh + 
                   d * X_mesh + e * Y_mesh + f)
        
        # 元のデータから2次曲面のうねりを差し引く
        corrected_data = Z - surface
            
        return corrected_data
    

    def _remove_scan_line_noise_zero(self, image_data):
        """
            各行の中央値（メディアン）を計算し、一律に差し引く（0次ラインレベリング）。
            複雑なうねり、帯状のノイズ、突発的なZドリフトを完全に除去します。
        """
        corrected = image_data.copy()

        # 1. 各行の中央値を計算 (構造の凸凹に引っ張られないよう、平均ではなくメディアンを使用)
        row_medians = np.median(corrected, axis=1)
        
        # NaNやInf対策（もし含まれている場合）
        row_medians = np.nan_to_num(row_medians, nan=0.0)
        
        # 2. 各行からその行の中央値を差し引く
        # reshape(-1, 1)で縦ベクトルにし、ブロードキャストで全列から引く
        corrected = corrected - row_medians.reshape(-1, 1)
        
        return corrected
    
    def highpass_filter_2d_directional(self, Z, cutoff_frequency = 0.005, direction='x'):
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
    
    def _remove_scan_line_noise(self, image_data, method='median', window_ratio=0.2):
        """
        縦方向の地形変化（うねり・段差）を保護しつつ、横縞ノイズを除去する関数。
        
        Parameters:
        -----------
        image_data : 2D array
        method : 'median' or 'mean'
            各行の代表値の計算方法。通常は 'median' が外れ値に強く推奨。
        window_ratio : float (0.0 < r < 1.0)
            縦方向のトレンドを計算する際の窓サイズの割合。
            画像の高さ(h)の何割を「滑らかさの基準」とするか。
            - 小さい(0.01): 細かい変化も「地形」とみなして残す（ノイズが消えにくい）
            - 大きい(0.5): 大きなうねりのみを「地形」とみなす（ノイズは消えるが、緩やかな坂も消えるかも）
        """
        h, w = image_data.shape
        corrected = image_data.copy()
        
        # 1. 各行の代表値（オフセット）を計算
        if method == 'median':
            offsets = np.median(corrected, axis=1)
        else:
            offsets = np.mean(corrected, axis=1)

        # 2. 窓サイズの決定 (画像の高さに対する割合で決める)
        # sigma はウィンドウサイズの約 1/6 程度に設定すると自然な平滑化になります
        window_size = int(h * window_ratio)
        sigma = window_size / 6.0 
        
        if window_size < 3:
            window_size = 3 # 最小サイズ
            sigma = 1.0

        # 3. トレンド（本来の縦方向の変化）を抽出
        # median_filterだとトレンドが階段状になるため、
        #  滑らかな gaussian_filter を使用します。
        #  これにより「オフセットの急激な変化」だけが浮き彫りになります。
        smooth_trend = gaussian_filter(offsets, sigma=sigma)
        
        # 4. ノイズ成分の抽出
        # Raw(ギザギザ) - Smooth(うねり) = Noise(激しい横縞)
        stripe_noise = offsets - smooth_trend
        
        # 5. 補正実行
        corrected = corrected - stripe_noise.reshape(-1, 1)
        
        return corrected
    
    def _create_2Darray_map(
        self,
        property_key: str,
        X_coords_um: np.ndarray,
        Y_coords_um: np.ndarray,
        Z_values: np.ndarray,
        grid_size: Tuple[int, int] = (512, 512),
        ):
        # 線形補間オブジェクトの作成
        Z_grid = interpolator_linear.afm_to_grid_linear(X_coords_um, Y_coords_um, Z_values, pixel_shape=grid_size)
        # topographyの場合、2次曲面フィッティングで全体の傾斜およびボウイングを補正する。また、高さも反転させ、実際のトポグラフィーに合わせる。
        if property_key == 'topography':
            Z_grid = self._flatten_surface_2nd_order(Z_grid)
            Z_grid = np.max(Z_grid) - Z_grid  # 高さを反転
            Z_grid -= np.min(Z_grid)  # 最小値を0にシフト
            Z_grid = self.highpass_filter_2d_directional(Z_grid, cutoff_frequency=0.005, direction='x')  # X方向の高周波成分を抽出

            # ラインレベリングを実施
            #Z_grid = self._remove_scan_line_noise_zero(Z_grid)
        return Z_grid
    
    def _calc_correlate(self,trace,retrace):
        # 1. 相互相関によるX方向のズレ（ラグ）検出
        # 行ごとにズレを計算するのはノイズに弱いため、画像全体の平均プロファイルで計算します
        trace_profile = np.nanmean(trace, axis=0)
        retrace_profile = np.nanmean(retrace, axis=0)
        
        # 欠損がある場合は0埋め等で仮定して相関をとる
        trace_prof_filled = np.nan_to_num(trace_profile)
        retrace_prof_filled = np.nan_to_num(retrace_profile)
        
        # 相関計算
        correlation = correlate(trace_prof_filled, retrace_prof_filled, mode='same')
        lags = np.arange(-len(trace_profile)//2 + 1, len(trace_profile)//2 + 1)
        
        # 相関がサイズ不一致でズレる場合の微調整
        if len(lags) != len(correlation):
            lags = np.arange(len(correlation)) - (len(correlation) // 2)
            
        best_lag = lags[np.argmax(correlation)]
        return best_lag

    def _merge_afm_data(self, trace, retrace, best_lag=None, outlier_threshold=None):
        """
        往路(trace)と復路(retrace)の位置ズレを補正し、
        欠損(NaN)や外れ値を考慮して統合した画像を返します。
        Young率などは、往復で大きく異なることがあるため、Heightの時のみ差を計算する。
        それ以外のマップの際は、計算した値を使用する。
        """

        # 2. 復路データの位置補正
        # X方向（axis=1）にのみシフトさせる
        retrace_shifted = shift(retrace, [0, best_lag], cval=np.nan, order=1)

        # 3. データの統合（マージ）
        # 空振り対策: 異常値フィルタリング（オプション）
        # 例: 平均から大きく外れている、あるいは物理的にありえない値をNaNにする
        if outlier_threshold is not None:
            # 簡易的な例: 値が極端に小さい/大きいものを除外
            trace[trace < outlier_threshold] = np.nan
            retrace_shifted[retrace_shifted < outlier_threshold] = np.nan

        # 統合ロジック
        # - 両方データがある場所 -> 平均
        # - 片方しかデータがない場所（空振り/シフトによる空白） -> ある方を採用
        # - 両方ない場所 -> NaN
        merged_data = np.nanmean(np.stack([trace, retrace_shifted]), axis=0)
        return merged_data
    
    # --- 保存メソッド ---

    def create_and_save_high_resolution_map(
        self, 
        data_list: List[AFMData], 
        property_key: str,
        output_dir: str, 
        grid_size: Tuple[int, int] = (512, 512),
        range_threshold: float = 30.0,
        only_trace: bool = True
    ):
        """
        RBF補間を用いて高解像度マップを生成し、PNG画像とNPZ 2D配列として保存する。
        1ラインスキャン時もRBF補間を行い、X軸センサー値、Y軸データインデックスとしてプロットする。
        """
        if not data_list:
            print("警告: データリストが空です。")
            return

        os.makedirs(output_dir, exist_ok=True)
        config = self._get_plot_config(property_key)

        # 1. 座標データ (センサー値) とZ値（解析結果）を抽出
        X_coords_um, Y_coords_um, x_range_um, y_range_um = self._extract_physical_coords(data_list)
        N_total = len(data_list)
        nx, ny = self._get_map_dimensions(data_list)
        try:
            Z_values = np.array([getattr(data_obj, property_key) for data_obj in data_list])
        except AttributeError:
            print(f"エラー: AFMDataオブジェクトに属性 '{property_key}' が見つかりません。")
            return
        

        # 往路方向（X座標が増加）と復路方向（X座標が減少）のデータを分別
        trace_indices = []
        retrace_indices = []
        
        for idx in range(len(X_coords_um) - 1):
            if X_coords_um[idx + 1] >= X_coords_um[idx]:
                # X座標が増加 → 往路
                trace_indices.append(idx)
            else:
                # X座標が減少 → 復路
                retrace_indices.append(idx)
        
        # 最後の点の処理
        if len(X_coords_um) > 0:
            last_idx = len(X_coords_um) - 1
            # 最後の点は最後の方向に追加
            if len(trace_indices) > 0 and trace_indices[-1] == last_idx - 1:
                trace_indices.append(last_idx)
            elif len(retrace_indices) > 0 and retrace_indices[-1] == last_idx - 1:
                retrace_indices.append(last_idx)
            else:
                # デフォルトは往路に追加
                trace_indices.append(last_idx)


        # 2. 1ラインスキャン判定とRBF補間用のY座標設定
        is_line_scan_by_range = self._is_line_scan_by_range(x_range_um, y_range_um, range_threshold)
        
        if is_line_scan_by_range:
            print("--- ⚠️ 範囲比率から1ラインスキャンを検出。Y軸をデータインデックスに置換し、RBF補間を継続。 ---")
            
            # 💡 修正: 1ラインスキャン時はY座標をデータインデックス(パス番号)に置き換え
            N_points_per_line = nx
            N_passes = N_total // N_points_per_line if N_points_per_line > 0 else N_total
            
            # Y座標を0からN_passesまで均等に分散させるダミー座標に置き換え
            # (RBF補間の入力Y座標として使用)
            Y_coords_um = np.repeat(np.arange(N_passes), N_points_per_line)
            y_range_plot = N_passes # Y軸のプロット範囲はパス数
            
            # X_coords_umはそのまま使用
            x_range_plot = x_range_um
            
        else:
            y_range_plot = y_range_um
            x_range_plot = x_range_um
        
        # 往路、復路でそれぞれマップを生成
        Z_grid_trace = self._create_2Darray_map(
            property_key, X_coords_um[trace_indices], Y_coords_um[trace_indices], Z_values[trace_indices], grid_size=grid_size
        )
        Z_grid_retrace = self._create_2Darray_map(
            property_key, X_coords_um[retrace_indices], Y_coords_um[retrace_indices], Z_values[retrace_indices], grid_size=grid_size
        )

        # 往路、復路データをマージ
        if property_key == 'topography': # トポ像はぶれが少ないので、位置合わせにはトポ像を使用
            self.best_lag = self._calc_correlate(Z_grid_trace, Z_grid_retrace)
        Z_grid = self._merge_afm_data(Z_grid_trace, Z_grid_retrace, best_lag=self.best_lag)

        # 2Dマップ配列 (.npz) の保存
        map_npz_path = os.path.join(output_dir, f'{property_key}_map.npz')
        np.savez_compressed(
            map_npz_path, 
            map_data=Z_grid, 
            x_min=X_coords_um.min(), x_max=X_coords_um.max(), 
            y_min=Y_coords_um.min(), y_max=Y_coords_um.max(), # 1ラインスキャン時はダミーのパス番号のmin/max
            x_range_um=x_range_plot, y_range_um=y_range_plot
        )
        #画像 (.png) の保存
        conversion_factor = self.UNIT_CONVERSION.get(property_key, 1.0)
        plot_data = Z_grid * conversion_factor
        if config['log_transform']:
            plot_data = np.log10(np.maximum(plot_data, 1e-12))
        
        plt.figure(figsize=(8, 8))
        # extent=[X_min, X_max, Y_min, Y_max]
        median = np.median(plot_data)
        q75, q25 = np.percentile(plot_data, [75, 25])
        iqr = q75 - q25
        vmin = median - 1.5 * iqr
        vmax = median + 1.5 * iqr
        im = plt.imshow(
            plot_data, 
            cmap=config['cmap'], 
            origin='upper', 
            # 1ラインスキャン時はY軸がパス番号の範囲 (0 to N_passes)
            extent=[X_coords_um.min(), X_coords_um.max(), Y_coords_um.min(), Y_coords_um.max()],
            vmin = vmin,
            vmax = vmax
        )
        cbar = plt.colorbar(im)
        cbar.set_label(config['label'])
        
        plt.xlabel(r'X Position ($\mu$m)')
        
        if is_line_scan_by_range:
            plt.ylabel('Scan Pass Number') # 💡 Y軸ラベルを修正
            plt.title(f"{config['title']} (Line Scan Map)")
            image_path = os.path.join(output_dir, f"{property_key}_linescan_{config['fname']}")
        else:
            plt.ylabel(r'Y Position ($\mu$m)')
            plt.title(config['title'])
            image_path = os.path.join(output_dir, f"{property_key}_{config['fname']}")
            
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

    def export_analysis_data_npz(self, data_list: List[AFMData], output_dir: str):
        """
        解析値の1D配列を統合し、NPZファイルとして保存する。
        """
        if not data_list:
            print("警告: エクスポートするデータがありません。")
            return

        os.makedirs(output_dir, exist_ok=True)
        keys = ['topography', 'youngs_modulus', 'delta', 'peak_force', 'hysteresis_area', 'cp_z_position']
        data_to_save = {}
        N = len(data_list)
        
        for key in keys:
             # float32を使用 (メモリ効率のため)
            data_array = np.empty(N, dtype=np.float32) 
            for i, data_obj in enumerate(data_list):
                data_array[i] = getattr(data_obj, key) 
            data_to_save[key] = data_array
            
        npz_path = os.path.join(output_dir, f'analysis_data.npz')
        # np.savez_compressed を使用
        np.savez_compressed(npz_path, **data_to_save)


