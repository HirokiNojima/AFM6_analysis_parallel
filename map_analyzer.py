import numpy as np
from typing import List, Dict, Any
from afm_data import AFMData
from processing import AFM_Curve_analyzer
from joblib import Parallel, delayed, cpu_count
from data_input import DataReader

# 並列実行のために、モジュールレベルでラッパー関数を定義
def _analyze_single_curve_wrapper_joblib(
    index_chunk: List[int], # カーブインデックスのチャンク
    folder_path: str,
    metadata_ref: Dict[str, Any]
) -> List[AFMData]:
    """
    カーブインデックスのリストを受け取り、各インデックスでTDMSを読み込み、解析を行う。
    失敗した場合は、nanで埋められたダミーオブジェクトを返す。
    """
    analyzer = AFM_Curve_analyzer()
    reader = DataReader() 
    results_list = []
    
    for index in index_chunk:
        data_obj = None # ★ 初期化
        try:
            # 1. 読み込み
            data_obj = reader.read_single_force_curve(
                folder_path, 
                index, 
                metadata_ref
            )
            # 2. 解析
            analyzer.analyze_single_curve(data_obj)
            
        except Exception as e:
            print(f"⚠️ インデックス {index} の解析/読み込みに失敗 (エラー: {e})。NaNとして処理します。")
            
            # 読み込み自体が失敗した場合 (data_objがNoneのまま)
            # プレースホルダーのダミーオブジェクトを作成する
            if data_obj is None:
                try:
                    # メタデータから最低限の座標を取得
                    xsensor = metadata_ref.get('xsensor', [0.0] * (index + 1))[index]
                    ysensor = metadata_ref.get('ysensor', [0.0] * (index + 1))[index]
                    
                    # ダミーのAFMDataを生成 (中身はすべて nan で初期化済)
                    data_obj = AFMData(
                        raw_deflection=np.array([np.nan]), # 空配列の代わりにnan
                        raw_ztip=np.array([np.nan]),
                        raw_zsensor=np.array([np.nan]),
                        metadata_ref=metadata_ref,
                        folder_path=folder_path,
                        hyst_curve=np.array([[np.nan, np.nan]]), # 空でない
                        xsensor=xsensor,
                        ysensor=ysensor
                    )
                except Exception as e_dummy:
                    # メタデータも壊れている場合の究極のフォールバック
                    print(f"❌ {index} のダミー作成失敗: {e_dummy}")
                    data_obj = AFMData(np.array([np.nan]), np.array([np.nan]), np.array([np.nan]),
                                       metadata_ref, folder_path, np.array([[np.nan, np.nan]]), 0.0, 0.0)

            # (もし読み込みは成功し、解析だけ失敗した場合、
            #  data_objはNoneではなく、各プロパティは初期値の np.nan のまま)
        
        # ★★★ 施策 ★★★
        # 解析が完了したら (成功・失敗問わず) メモリを破棄
        if data_obj:
            data_obj.clear_raw_data()
            
            # ★ リストに追加 (失敗しても追加する)
            results_list.append(data_obj)
        
    return results_list

class AFM_Map_Analyzer_Joblib:
    """
    フォースマップデータ（AFMDataのリスト）をjoblibで並列処理し解析するクラス。
    """
    # 🌟 変更点 4: __init__でフォルダパスとメタデータを保持
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
        print(f"--- 🚀 フォースマップ解析開始 (joblib並列処理, n_jobs={actual_jobs}) ---")

        # 🌟 変更点 6: 0 から N_curves-1 までのインデックスリストを生成
        all_indices = list(range(N_curves))
        
        # データリストをチャンクに分割
        chunk_size = 50 # 🌟 チャンクサイズは実験的に調整
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
            # 🌟 変更点 7: ラッパー関数に追加の引数を渡す
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