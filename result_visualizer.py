import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Tuple
from afm_data import AFMData
from skimage.filters import threshold_otsu 
from scipy.ndimage import median_filter, gaussian_filter
import interpolator_linear

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
        
    def _get_plot_config(self, property_key: str) -> Dict[str, Any]:
        """指定されたプロパティのプロット設定を取得する。"""
        return self.PLOT_CONFIG.get(property_key, self.DEFAULT_CONFIG)

    # --- 補助メソッド ---

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

        print('データリスト長さ確認', data_list[1].xsensor)

        # AFMDataのxsensor, ysensor属性から直接座標を取得
        X_coords = np.array([data_obj.xsensor for data_obj in data_list])
        Y_coords = np.array([data_obj.ysensor for data_obj in data_list])
        print('抽出座標形状確認', X_coords.shape, Y_coords.shape)

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
    
    def _flatten_plane(self, data_2d):
        """
        2次元配列に対して、面でのフィッティングを行い、全体の傾斜を補正する。
        
        Parameters:
            data_2d (np.ndarray): 補正前の2次元高さデータ (Height Map)
        
        Returns:
            np.ndarray: 補正後のデータ
        """
        # 入力データのコピーを作成（元データを破壊しないため）
        corrected_data = data_2d.copy()
        
        # 画像のサイズを取得 (高さ: rows, 幅: cols)
        rows, cols = corrected_data.shape
        
        # X軸の座標配列を作成 (0, 1, 2, ... cols-1)
        x = np.arange(cols)


        # Fit and subtract a first-order plane z = a*x + b*y + c from the entire 2D map
        # X: columns (0..cols-1), Y: rows (0..rows-1)
        y = np.arange(rows)
        X_mesh, Y_mesh = np.meshgrid(x, y)  # shape (rows, cols)

        Z = corrected_data.astype(np.float64)
        Z_flat = Z.ravel()
        X_flat = X_mesh.ravel()
        Y_flat = Y_mesh.ravel()

        mask = np.isfinite(Z_flat)
        if mask.sum() < 3:
            # Not enough valid points to fit a plane; return original copy
            return corrected_data

        A = np.column_stack((X_flat[mask], Y_flat[mask], np.ones(mask.sum())))
        coeffs, *_ = np.linalg.lstsq(A, Z_flat[mask], rcond=None)
        a, b, c = coeffs

        plane = (a * X_mesh + b * Y_mesh + c)
        corrected_data = Z - plane
            
        return corrected_data
    
    import numpy as np

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
        # 【変更点】median_filterだとトレンドが階段状になるため、
        #  滑らかな gaussian_filter を使用します。
        #  これにより「オフセットの急激な変化」だけが浮き彫りになります。
        smooth_trend = gaussian_filter(offsets, sigma=sigma)
        
        # 4. ノイズ成分の抽出
        # Raw(ギザギザ) - Smooth(うねり) = Noise(激しい横縞)
        stripe_noise = offsets - smooth_trend
        
        # 5. 補正実行
        corrected = corrected - stripe_noise.reshape(-1, 1)
        
        return corrected
    
    # --- 保存メソッド ---

    def create_and_save_high_resolution_map(
        self, 
        data_list: List[AFMData], 
        property_key: str,
        output_dir: str, 
        grid_size: Tuple[int, int] = (512, 512),
        range_threshold: float = 30.0
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

        # ヤング率解析不良データを削除
        if property_key == 'youngs_modulus':
            # 大津の方法を用いて、空振り判定の押し込み量を算出
            Delta_values = np.array([getattr(data_obj, 'delta', np.nan) for data_obj in data_list])
            thres_delta = threshold_otsu(Delta_values[~np.isnan(Delta_values)]) * 0.5 # 係数に関しては実験的に調整
            data_list = [data for data in data_list if getattr(data, 'delta', np.nan) >= thres_delta]# 押し込み過少なデータを削除

            filtered_length = len(data_list)
            if filtered_length == 0:
                print("❌ 有効なヤング率データがありません。処理を中止します。")
                return
            
        # 線形補間オブジェクトの作成
        Z_grid = interpolator_linear.afm_to_grid_linear(X_coords_um, Y_coords_um, Z_values, pixel_shape=grid_size)
        # topographyの場合、一次元平面でフィッティングして全体の傾斜を補正する。また、高さも反転させ、実際のトポグラフィーに合わせる。
        if property_key == 'topography':
            Z_grid = self._flatten_plane(Z_grid)
            Z_grid = np.max(Z_grid) - Z_grid  # 高さを反転
            Z_grid -= np.min(Z_grid)  # 最小値を0にシフト

        # ラインレベリングを実施
        Z_grid = self._remove_scan_line_noise(Z_grid, method='median')
            
        # 4. 2Dマップ配列 (.npz) の保存
        map_npz_path = os.path.join(output_dir, f'{property_key}_map.npz')
        # Z_gridは補間後の高解像度データ
        np.savez_compressed(
            map_npz_path, 
            map_data=Z_grid, 
            x_min=X_coords_um.min(), x_max=X_coords_um.max(), 
            y_min=Y_coords_um.min(), y_max=Y_coords_um.max(), # 1ラインスキャン時はダミーのパス番号のmin/max
            x_range_um=x_range_plot, y_range_um=y_range_plot
        )
        print(f"✅ 2Dマップ配列 (.npz) を保存: {map_npz_path}")

        # 5. 画像 (.png) の保存
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
        print(f"✅ 高解像度画像 (.png) を保存: {image_path}")

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
        print(f"✅ 解析データNPZ (1D配列) を保存: {npz_path}")


    