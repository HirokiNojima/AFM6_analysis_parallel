import numpy as np
from typing import Dict, Any

class AFMData:
    """
    生データとメタデータを保持するクラス。
    共通メタデータは参照として保持することでメモリ効率を高めます。
    """
    # 引数名を metadata_ref に変更し、参照を保持
    def __init__(
            self,
            raw_deflection: np.ndarray,
            raw_ztip: np.ndarray,
            raw_zsensor: np.ndarray,
            metadata_ref: Dict[str, Any],
            folder_path: str,
            hyst_curve: np.ndarray,
            xsensor: float = 0.0,
            ysensor: float = 0.0
            ):
        # --- 生データを保持する属性 ---
        self.raw_deflection = raw_deflection
        self.raw_ztip = raw_ztip
        self.raw_zsensor = raw_zsensor
        self.metadata = metadata_ref  # 参照を保持
        self.folder_path = folder_path  # 参照として保持する
        self.hyst_curve = hyst_curve  # ヒステリス曲線データ。参照として保持する。
        self.xsensor = xsensor  # センサのX位置
        self.ysensor = ysensor  # センサのY位置

        # ★★★ 以下の2行を追加 ★★★
        # 必要なメタデータをスカラ値としてコピー
        self.XStep = metadata_ref.get('XStep', 1)
        self.YStep = metadata_ref.get('YStep', 1)
        
        # --- DataProcessorで変化させる属性 ---
        self.force: np.ndarray = np.array([])             # 力 [N]
        self.z_distance: np.ndarray = np.array([])        # 探針-サンプル間距離 [m]
        self.force_corrected: np.ndarray = np.array([])   # ベースライン補正後の力 [N]
        self.contact_point_index: int = -1              # 接触点のインデックス
        self.retract_point_index: int = -1              # リトラクトポイントのインデックス
        
        # --- 解析結果を保持する属性 ---
        self.topography: float = np.nan                   # トポグラフィー高さ [m]
        self.youngs_modulus: float = np.nan               # ヤング率 [Pa]
        self.delta: float = np.nan                        # 押し込み量 [m]
        self.peak_force: float = np.nan                   # ピーク力 [N]
        self.hysteresis_area: float = np.nan              # ヒステリシス面積 [N·m]
        self.cp_z_position: float = np.nan                # 接触点のZ位置 [m]
    
    def clear_raw_data(self):
        """
        解析完了後、メモリ削減のために生データと中間データ配列を破棄する。
        解析結果のスカラ値と座標は保持される。
        """
        self.raw_deflection = None
        self.raw_ztip = None
        self.raw_zsensor = None
        self.force = None
        self.z_distance = None
        self.force_corrected = None
        self.hyst_curve = None # 参照データも不要なら破棄
        self.metadata = None
    