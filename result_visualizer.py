import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Tuple
from afm_data import AFMData 
from interpolator import FastRBFInterpolator2D 

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
                'cmap': 'afmhot_r',
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
    
    def _line_flatten_1st_order(self, data_2d):
        """
        2次元配列に対して、行ごとに1次補正（傾きと切片の除去）を行う関数
        
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
        
        # --- 行ごとのループ処理 ---
        for i in range(rows):
            # 1行分のデータを取得
            y_data = corrected_data[i, :]
            
            # 1次多項式 (y = ax + b) の係数を計算 (最小二乗法)
            # polyfit(x, y, 1) は [傾きa, 切片b] を返します
            slope, intercept = np.polyfit(x, y_data, 1)
            
            # フィッティングした直線を作成
            fitted_line = slope * x + intercept
            
            # 元データからフィッティング直線を引く
            corrected_data[i, :] = y_data - fitted_line
            
        return corrected_data
    
    # --- 保存メソッド ---

    def create_and_save_high_resolution_map(
        self, 
        data_list: List[AFMData], 
        property_key: str, 
        base_filename: str, 
        output_dir: str, 
        grid_size: Tuple[int, int] = (512, 512),
        interpolator_kwargs: Dict[str, Any] = None,
        range_threshold: float = 30.0
    ):
        """
        RBF補間を用いて高解像度マップを生成し、PNG画像とNPZ 2D配列として保存する。
        1ラインスキャン時もRBF補間を行い、X軸センサー値、Y軸データインデックスとしてプロットする。
        """
        print(f"--- 🖼️ 高解像度 {property_key} マップ生成・保存 ---")
        if not data_list:
            print("警告: データリストが空です。")
            return

        os.makedirs(output_dir, exist_ok=True)
        config = self._get_plot_config(property_key)
        print('設定取得完了')
        
        # 1. 座標データ (センサー値) とZ値（解析結果）を抽出
        X_coords_um, Y_coords_um, x_range_um, y_range_um = self._extract_physical_coords(data_list)
        print('座標抽出完了')
        N_total = len(data_list)
        nx, ny = self._get_map_dimensions(data_list)
        print('マップ寸法取得完了')

        try:
            Z_values = np.array([getattr(data_obj, property_key) for data_obj in data_list])
        except AttributeError:
            print(f"エラー: AFMDataオブジェクトに属性 '{property_key}' が見つかりません。")
            return
        print('Z値抽出完了')
        
        # 2. 1ラインスキャン判定とRBF補間用のY座標設定
        is_line_scan_by_range = self._is_line_scan_by_range(x_range_um, y_range_um, range_threshold)
        print(f'1ラインスキャン判定完了: {is_line_scan_by_range}')
        
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
            
        # 3. RBF補間の実行 (1ライン/2D両対応)
        kwargs = interpolator_kwargs if interpolator_kwargs is not None else {}
        interpolator = FastRBFInterpolator2D(grid_size=grid_size, **kwargs)

        Z_grid = interpolator.fit_transform(X_coords_um, Y_coords_um, Z_values)

        # topographyの場合、一次元平面でフィッティングして全体の傾斜を補正する。
        if property_key == 'topography':
            print('トップグラフィー傾斜補正中...')
            Z_grid = self._line_flatten_1st_order(Z_grid)
            print('傾斜補正完了。')
            
        # 4. 2Dマップ配列 (.npz) の保存
        map_npz_path = os.path.join(output_dir, f'{base_filename}_{property_key}_map.npz')
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
            image_path = os.path.join(output_dir, f"{base_filename}_{property_key}_linescan_{config['fname']}")
        else:
            plt.ylabel(r'Y Position ($\mu$m)')
            plt.title(config['title'])
            image_path = os.path.join(output_dir, f"{base_filename}_{property_key}_{config['fname']}")
            
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 高解像度画像 (.png) を保存: {image_path}")

    def export_analysis_data_npz(self, data_list: List[AFMData], base_filename: str, output_dir: str):
        """
        解析値の1D配列を統合し、NPZファイルとして保存する。
        """
        if not data_list:
            print("警告: エクスポートするデータがありません。")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        # 🌟 変更点 1: X/Y座標をキーリストから除外
        keys = ['topography', 'youngs_modulus', 'delta', 'peak_force', 'hysteresis_area', 'cp_z_position']
        
        # 🌟 変更点 2: 座標データ dict の作成を削除
        data_to_save = {}
        
        # 🌟 変更点 3: データ集約の高速化
        N = len(data_list)
        
        for key in keys:
             # float32を使用 (メモリ効率のため)
            data_array = np.empty(N, dtype=np.float32) 
            for i, data_obj in enumerate(data_list):
                data_array[i] = getattr(data_obj, key) 
            data_to_save[key] = data_array
            
        npz_path = os.path.join(output_dir, f'{base_filename}_analysis_data.npz')
        # np.savez_compressed を使用
        np.savez_compressed(npz_path, **data_to_save)
        print(f"✅ 解析データNPZ (1D配列) を保存: {npz_path}")


    