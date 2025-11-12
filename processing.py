import afm_calclators as afm
from afm_data import AFMData
import numpy as np

class AFM_Curve_analyzer:
    def analyze_single_curve(self, data_obj: AFMData):
        self._convert_def_to_force(data_obj)
        self._calc_z_distance(data_obj)
        self._force_correct(data_obj)
        self._calc_CPandRP(data_obj)
        self._calc_topography(data_obj)
        self._calc_young_and_delta(data_obj)
        self._calc_peak_force(data_obj)
        self._calc_hyst_area(data_obj)
        self._calc_cp_z_position(data_obj)

    def _convert_def_to_force(self, data_obj: AFMData):
        """
        フォースカーブのDeflectionデータを力に変換する。
        力 = ばね定数 * Deflection
        """
        k = data_obj.metadata['SPRING_CONSTANT']  # ばね定数 [N/m]
        invols = data_obj.metadata['InvOLS']  # InvOLS [m/V]
        data_obj.force = data_obj.raw_deflection * k * invols  # 力 [N]

    def _calc_z_distance(self, data_obj: AFMData):
        """
        フォースカーブ横軸部分を作成する。
        """
        ztip = data_obj.raw_ztip # ZTip入力電圧
        hyst_curve = data_obj.hyst_curve # ヒステリス補正用マスターカーブ
        # ヒステリス補正
        corrected_data = afm.correct_hyst(ztip, hyst_curve)
        data_obj.z_distance = corrected_data  # 探針-サンプル間距離 [m]
    
    def _force_correct(self, data_obj: AFMData):
        """
        フォースカーブのベースライン補正を行う。
        """
        data_obj.force_corrected = afm.baseline_correction(data_obj.force, data_obj.z_distance)

    def _calc_CPandRP(self, data_obj: AFMData):
        """
        フォースカーブの接触点とリトラクトポイントを検出する。
        """
        app, ret = afm.devide_app_ret(data_obj.force_corrected, data_obj.z_distance)
        cp_idx = afm.calc_touch_point(app[0], app[1])
        rp_idx = afm.calc_touch_point(ret[0], ret[1]) + len(app[0])  # リトラクトポイントのインデックスを全体インデックスに変換
        data_obj.contact_point_index = cp_idx
        data_obj.retract_point_index = rp_idx
    
    def _calc_topography(self, data_obj: AFMData):
        """
        フォースカーブからトポグラフィー高さを計算する。
        """
        cp_idx = data_obj.contact_point_index
        data_obj.topography = data_obj.z_distance[cp_idx] + data_obj.raw_zsensor[cp_idx] * data_obj.metadata['DISTANCE_PER_VOLT']  # トポグラフィー高さ [m]

    def _calc_young_and_delta(self, data_obj: AFMData):
        """
        フォースカーブからヤング率と押し込み量を計算する。
        """
        force_carea = data_obj.force_corrected[data_obj.contact_point_index:np.argmax(data_obj.z_distance)]
        z_distance_carea = data_obj.z_distance[data_obj.contact_point_index:np.argmax(data_obj.z_distance)]
        delta = z_distance_carea - force_carea / data_obj.metadata['SPRING_CONSTANT'] - z_distance_carea[0]  # 押し込み量 [m]
        youngs_modulus = afm.calc_young_snnedon(force_carea, delta)
        data_obj.youngs_modulus = youngs_modulus  # ヤング率 [Pa]
        data_obj.delta = np.max(delta)  # 押し込み量 [m]

    def _calc_peak_force(self, data_obj: AFMData):
        """
        フォースカーブからピーク力を計算する。
        """
        data_obj.peak_force = np.max(data_obj.force_corrected)  # ピーク力 [N]
    
    def _calc_hyst_area(self, data_obj: AFMData):
        """
        フォースカーブからヒステリシスループ面積を計算する。
        """
        data_obj.hysteresis_area = afm.calc_hyst_area(
            data_obj.force_corrected,
            data_obj.z_distance,
            data_obj.contact_point_index,
            data_obj.retract_point_index
        )  # ヒステリシス面積 [N·m]
    
    def _calc_cp_z_position(self, data_obj: AFMData):
        """
        フォースカーブから接触点のZ位置を計算する。
        """
        data_obj.cp_z_position = data_obj.z_distance[data_obj.contact_point_index]  # 接触点のZ位置 [m]
        

if __name__ == "__main__":
    # 動作確認用コード
    import time


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
    start_time = time.time()
    
    analyzer = AFM_Curve_analyzer()
    analyzer.analyze_single_curve(afm_data)
    print("Topography (m):", afm_data.topography)
    print("Young's Modulus (Pa):", afm_data.youngs_modulus)
    print("Delta (m):", afm_data.delta)
    print("Peak Force (N):", afm_data.peak_force)
    print("Hysteresis Area (N·m):", afm_data.hysteresis_area)
    print("Contact Point Z Position (m):", afm_data.cp_z_position)
    import matplotlib.pyplot as plt
    plt.plot(afm_data.z_distance, afm_data.force_corrected, label="Corrected Force Curve")
    plt.scatter(afm_data.z_distance[afm_data.contact_point_index], afm_data.force_corrected[afm_data.contact_point_index], color='red', label="Contact Point")
    plt.scatter(afm_data.z_distance[afm_data.retract_point_index], afm_data.force_corrected[afm_data.retract_point_index], color='green', label="Retract Point")
    plt.xlabel("Z Distance (m)")    
    plt.ylabel("Force (N)")
    plt.legend()
    plt.show()



    # 2025年10月30日：作成完了。