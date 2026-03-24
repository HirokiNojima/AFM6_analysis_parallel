import numpy as np
from typing import List, Dict, Any
from afm_data import AFMData
from processing import AFM_Curve_analyzer
from joblib import Parallel, delayed, cpu_count
from data_input import DataReader

# 並列実行のために、モジュールレベルでラッパー関数を定義
def _analyze_single_curve_wrapper_joblib(
    index_chunk: List[int], 
    folder_path: str,
    metadata_ref: Dict[str, Any]
) -> List[AFMData]:
    """
    チャンク単位でファイルを一度だけ開き、読み込んでから解析する。
    """
    analyzer = AFM_Curve_analyzer()
    reader = DataReader() 
    results_list = []
    
    # 1. バッチ読み込み (ここでTDMSアクセスは1回のみ発生)
    # ----------------------------------------------------
    try:
        # data_objects は AFMData または None のリスト
        raw_data_objects = reader.read_batch_force_curves(folder_path, index_chunk, metadata_ref)
    except Exception as e:
        print(f"❌ Batch read failed: {e}")
        # 全滅した場合のダミー処理が必要ならここに記述
        return []

    # 2. メモリ上にあるデータを順次解析
    # ----------------------------------------------------
    for i, data_obj in enumerate(raw_data_objects):
        index = index_chunk[i]
        
        if data_obj is None:
            # 読み込み失敗時のダミー生成処理 (元のコードと同様)
            try:
                xsensor = metadata_ref.get('xsensor', [0.0] * (index + 1))[index]
                ysensor = metadata_ref.get('ysensor', [0.0] * (index + 1))[index]
                data_obj = AFMData(
                    np.array([np.nan]), np.array([np.nan]), np.array([np.nan]),
                    metadata_ref, folder_path, np.array([[np.nan, np.nan]]), 
                    xsensor, ysensor
                )
            except:
                continue # 完全な失敗

        try:
            # 解析実行
            analyzer.analyze_single_curve(data_obj)
        except Exception as e:
            print(f"⚠️ Index {index} analysis failed: {e}")
            # 解析失敗時もオブジェクト自体は残す（値はNaN）
        
        # 生データを破棄してメモリ開放
        data_obj.clear_raw_data()
        results_list.append(data_obj)
        
    return results_list

class AFM_Map_Analyzer_Joblib:
    """
    フォースマップデータ（AFMDataのリスト）をjoblibで並列処理し解析するクラス。
    """
    # __init__でフォルダパスとメタデータを保持
    def __init__(self, n_jobs: int = -1, folder_path: str = "", metadata_ref: Dict[str, Any] = None):
        self.n_jobs = n_jobs
        self.folder_path = folder_path
        self.metadata_ref = metadata_ref
        if not folder_path or metadata_ref is None:
            raise ValueError("folder_pathとmetadata_refは必須です。")

    def analyze_map_parallel(self, N_curves: int) -> List[AFMData]:
        """
        フォースマップの全フォースカーブをjoblibで並列処理し解析する。

        Parameters
        ----------
        N_curves : int
            フォースマップ全体のカーブ総数。

        Returns
        -------
        List[AFMData]
            解析結果が格納されたAFMDataオブジェクトのリスト。
        """
        
        actual_jobs = cpu_count() if self.n_jobs == -1 else self.n_jobs

        # 0 から N_curves-1 までのインデックスリストを生成
        all_indices = list(range(N_curves))
        
        # データリストをチャンクに分割
        chunk_size = 200 # チャンクサイズは実験的に調整
        print("chunk_size:")
        data_chunks = [
            all_indices[i:i + chunk_size] 
            for i in range(0, N_curves, chunk_size)
        ]

        # Parallelとdelayedを使用して、インデックスチャンクをワーカーに配布
        results_list = Parallel(
            n_jobs=self.n_jobs, 
            verbose=1,
            backend='loky'
        )(
            # ラッパー関数に追加の引数を渡す
            delayed(_analyze_single_curve_wrapper_joblib)(
                chunk, 
                self.folder_path, 
                self.metadata_ref
            ) 
            for chunk in data_chunks
        )
        
        # チャンクごとの結果をフラットなリストに結合 (順序は保持される)
        results_list = [item for sublist in results_list for item in sublist]
        
        print(f"--- ✅ フォースマップ解析完了 ---")
        return results_list
    
if __name__ == "__main__":
    # 動作確認用コード
    from afm_data import AFMData
    data = np.loadtxt(r"C:\Users\icell\Desktop\nojima_python\AFM6analysis_20251024\testdata\FCdata.txt")
    z_distance = data[:, 1]
    force = data[:, 0]
    z_sensor = data[:, 2] / 3e+5  # Zセンサデータを電圧に戻す。
    afm_data = AFMData(
        raw_deflection=force,
        raw_ztip=z_distance,
        raw_zsensor=z_sensor,
        metadata_ref={
            'SPRING_CONSTANT': 0.1,  # N/m
            'InvOLS': 1e-9,          # m/V
            'DISTANCE_PER_VOLT': 30e-6 # m/V
        },
        folder_path="C:/test/path",
        hyst_curve = np.loadtxt(r"C:\Users\icell\Desktop\nojima_python\AFM6analysis_20251024\補正用データ\3kHz\mean_FCdata.txt")  # 仮のヒステリス曲線データ
    )
    analyzer = AFM_Map_Analyzer_Joblib(n_jobs=-1)
    results = analyzer.analyze_map_parallel([afm_data for _ in range(100000)])  # 同じデータを100個解析。同じデータの時はシリアライズにより高速化される。
    print (results[0].topography)  # 最初のデータのトポグラフィー高さを表示