
import os
import time
from pathlib import Path
from data_input import DataReader
from map_analyzer import AFM_Map_Analyzer_Joblib
from result_visualizer import AFM_Result_Visualizer
from typing import List, Dict, Any
from afm_data import AFMData

def main_analysis_workflow(
    folder_path: str, 
    invols: float, 
    output_dir_name: str = "Analysis_Results",
    grid_size: List[int] = [512, 512],
    n_jobs: int = -1
):
    """
    AFMフォースマップデータの読み込み、解析、可視化を行うメイン関数。

    Parameters
    ----------
    folder_path : str
        データ（config.txt, ForceCurve.tdmsなど）が格納されているフォルダのパス。
    invols : float
        光てこ感度 (InvOLS) の値 [nm/V]。
    output_dir_name : str
        解析結果を保存するフォルダ名。
    grid_size : List[int]
        高解像度マップ生成時のグリッドサイズ [nx, ny]。
    n_jobs : int
        並列処理に使用するCPUコア数 (-1は全コア使用)。
    """
    
    # 1. 初期設定とディレクトリの準備
    start_time = time.time()
    
    # InvOLSの単位をnm/Vから、DataReaderが要求するfloat値 (e.g., 100.0) として扱う
    # DataReader._parse_config内で [m/V] に変換される
    base_dir = Path(folder_path)
    output_path = base_dir / output_dir_name
    os.makedirs(output_path, exist_ok=True)

    print(f"==================================================")
    print(f"🚀 AFM Force Map Analysis Started")
    print(f"📁 Data Folder: {folder_path}")
    print(f"📈 InvOLS: {invols:.2f} nm/V")
    print(f"💾 Output Folder: {output_path.resolve()}")
    print(f"==================================================")

    # 2. データ読み込み (DataReader: マスターの役割)
    print(f"--- 📥 メタデータとカーブ総数の読み込み中 ---")
    try:
        data_reader = DataReader()
        # configを読み込み、カーブ総数とメタデータを取得
        # AFMDataオブジェクトのリストは生成しない
        metadata = data_reader.read_config_only(folder_path, invols)
        if not metadata:
            print("❌ 致命的なエラー: 設定ファイルからメタデータが読み込まれませんでした。")
            return
            
        N_curves = metadata.get('XStep', 1) * metadata.get('YStep', 1)
        
    except Exception as e:
        print(f"❌ データ読み込み中にエラーが発生しました: {e}")
        return
    print(f"--- ✅ メタデータ読み込み完了: 全 {N_curves} カーブ ---")

    # 3. フォースカーブの並列解析 (AFM_Map_Analyzer_Joblib: ワーカーの役割)
    print(f"--- 🔬 フォースマップ解析中 (並列処理: n_jobs={n_jobs}) ---")
    try:
        map_analyzer = AFM_Map_Analyzer_Joblib(
            n_jobs=n_jobs,
            folder_path=folder_path, # フォルダパスをアナライザに渡す
            metadata_ref=metadata    # メタデータをアナライザに渡す
        )
        # 実際のデータリストではなく、カーブの総数を渡す
        analyzed_data_list = map_analyzer.analyze_map_parallel(N_curves)

    except Exception as e:
        print(f"❌ 並列解析中にエラーが発生しました: {e}")
        return
    

    # 4. 結果の可視化と保存 (AFM_Result_Visualizer)
    result_visualizer = AFM_Result_Visualizer()

    # 解析結果 (1D配列) のNPZ保存
    print(f"--- 💾 解析データのNPZ保存中 ---")
    result_visualizer.export_analysis_data_npz(
        analyzed_data_list,
        str(output_path)
    )

    # 高解像度マップの生成と保存
    print(f"--- 🗺️ 高解像度マップの生成と保存中 ---")
    analysis_properties = [
        'topography',
        'youngs_modulus', 
        'peak_force', 
        'delta', 
        'cp_z_position', 
        'hysteresis_area'
    ]
    for prop in analysis_properties:
        try:
            result_visualizer.create_and_save_high_resolution_map(
                analyzed_data_list, 
                property_key=prop, 
                output_dir=str(output_path),
                grid_size=tuple(grid_size)
            )
        except Exception as e:
            print(f"⚠️ {prop} マップの生成・保存中にエラーが発生しました: {e}")

    # 5. 終了処理
    end_time = time.time()
    total_time = end_time - start_time
    print(f"==================================================")
    print(f"✅ 全解析プロセスが完了しました。")
    print(f"⏱️ 実行時間: {total_time:.2f} 秒")
    print(f"==================================================")

# --- 実行ブロック ---
if __name__ == '__main__':
    # データフォルダパス
    folder_path = input("input data folder path: ")
    
    # 光てこ感度 [nm/V]
    invols_nm_per_volt = float(input("input InvOLS (nm/V): "))
    
    # 高解像度マップのグリッドサイズ
    map_grid_size = [300, 300] 
    
    # 並列処理のコア数 (-1: 全コア)
    num_jobs = -1 

    main_analysis_workflow(
        folder_path=folder_path, 
        invols=invols_nm_per_volt,
        output_dir_name="AFM_Analysis_Results",
        grid_size=map_grid_size,
        n_jobs=num_jobs
    )


# # %% 親フォルダからすべて解析する場合
# mother_folder = r"C:\nojima\AFM6measurement\260129"
# import os 
# invOLS = 100
# map_grid_size = [300, 300]
# num_jobs = -1
# for folder_name in os.listdir(mother_folder):
#     print(f"Processing folder: {os.path.join(mother_folder, folder_name)}")
#     main_analysis_workflow(
#         folder_path=os.path.join(mother_folder, folder_name), 
#         invols=invOLS,
#         output_dir_name="AFM_Analysis_Results",
#         grid_size=map_grid_size,
#         n_jobs=num_jobs
#     )

# %%
