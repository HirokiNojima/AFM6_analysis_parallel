import os
import numpy as np
import matplotlib.pyplot as plt
from data_input import DataReader
from map_analyzer import AFM_Map_Analyzer_Joblib
from result_visualizer import AFM_Result_Visualizer
from typing import List, Dict, Any
from afm_data import AFMData

def main():
    #InvOLS算出
    PATH = input('測定データフォルダのパスを入力')
    use_Zsen = input('横軸にZSensor値を使用？ (1/0): ') == '1'  # Zセンサ値を使用するかどうか
    if not use_Zsen:
        spring_constant = float(input('ばね定数(N/m)を入力: '))  # 例: 42.0
        in_liquid = input('液中計測？ (1/0): ') == '1'  # 液体中の測定かどうか
    savepath = os.path.join(PATH,'InvOLS')
    os.makedirs(savepath, exist_ok=True) #Forcecurveフォルダがない場合は作成

    #設定
    data_reader = DataReader()
    metadata = data_reader.read_config_only(PATH, invols=100.0)  # 仮のInvOLS値を渡す（実際にはmain内で計算する）
    if not metadata:
        print("❌ 致命的なエラー: 設定ファイルからメタデータが読み込まれませんでした。")
        return
    N_curves = metadata.get('XStep', 1) * metadata.get('YStep', 1)

    #フォースカーブデータの読み込み
    all_curve_data = loader.load_all_FCdata_from_tdms(
        tqdm_filepath, 
        int(config['XStep']*config['YStep']), 
        int(config['FCあたりのデータ取得点数']), 
        )
    if not use_Zsen:
        hyst_curve = loader.get_hyst_data_relative_to_script(config,spring_constant=spring_constant, in_liquid=in_liquid)

    #InvOLS算出
    InvOLS_list = []
    for curve in all_curve_data[:100]:  # 最初の100カーブで計算
        if not use_Zsen:
            curve = analyze.correct_hyst_by_curve(curve, hyst_curve)
        else:
            curve[:,1] = curve[:,2]  # Zセンサ値を用いたInvOLS算出
        curve = analyze.baseline_correction(curve, 0.3)
        CPandRP = analyze.find_CPandRP_threshold(curve, threshold=0.1 * np.max(curve[:, 0]))
        InvOLS = analyze.calc_InvOLS(curve, CPandRP['CP_idx'])
        InvOLS_list.append(InvOLS)
    
    #結果保存
    plt.scatter(curve[:, 1], curve[:, 0],s = 0.1)
    plt.scatter(curve[CPandRP["CP_idx"], 1], curve[CPandRP["CP_idx"], 0], color='red')
    plt.xlabel('ZTip_in(V)')
    plt.ylabel('Deflection(V)')
    plt.title('InvOLS(nm/V):' + str(InvOLS))
    plt.savefig(os.path.join(savepath, 'InvOLS_plot.png'))
    InvOLS = np.median(InvOLS_list)
    InvOLS_list.remove(np.max(InvOLS_list))
    print('InvOLS(nm/V):', InvOLS, '\n', 'SD:', np.std(InvOLS_list))

    with open(os.path.join(savepath, 'InvOLS_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f'InvOLS median (nm/V): {InvOLS}\n')
        f.write(f'InvOLS SD: {np.std(InvOLS_list)}\n')
        f.write(f'InvOLS list (nm/V): {InvOLS_list}\n')
    plt.close()

if __name__ == "__main__":
    print('InvOLS算出を開始します。')
    main()
    print('InvOLS算出を終了します。')