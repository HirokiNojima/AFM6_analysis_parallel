# ファイル名: napari_app.py
import napari
import numpy as np
import os
from pathlib import Path
from sklearn.neighbors import KDTree
from nptdms import TdmsFile
import sys

# --- GUI / Plotting ---
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox
)
from matplotlib.figure import Figure
# napari-matplotlib プラグイン
from napari_matplotlib.base import NapariMPLWidget

# --- 既存のAFM解析モジュール ---
from data_input import DataReader
from processing import AFM_Curve_analyzer
from interpolator import FastRBFInterpolator2D
from result_visualizer import AFM_Result_Visualizer
from afm_data import AFMData

# --- app.py から流用するヘルパー関数 ---
# (napariではキャッシュデコレータは不要)

def generate_interpolated_map(xy_coords, z_values, grid_size=(256, 256)):
    """ RBF補間を実行し、マップ像データを生成する """
    print("マップ像を補間中...")
    interpolator = FastRBFInterpolator2D(grid_size=grid_size)
    z_grid = interpolator.fit_transform(xy_coords[:, 0], xy_coords[:, 1], z_values)
    print("補間完了。")
    return z_grid

def build_kdtree(xy_coords):
    """ 座標からKDTreeを構築する """
    print("KDTreeを構築中...")
    return KDTree(xy_coords)


class AFMViewer:
    """
    NapariをベースにしたAFMインタラクティブビューア。
    Streamlitのセッション状態(st.session_state)の代わりに、
    このクラスのインスタンス変数(self.data)にデータを保持する。
    """
    def __init__(self):
        # 1. データ保持用の辞書を初期化
        self.data = {
            "folder_path": None,
            "metadata": None,
            "analysis_results": None,
            "kdtree": None,
            "xy_coords_um": None,
            "map_extent": None,
            "available_maps": [],
            "tdms_file": None,
            "deflection_ch": None,
            "ztip_ch": None,
            "zsensor_ch": None,
            "hyst_curve": None,
        }
        
        # 2. 既存モジュールのインスタンスを生成
        self.reader = DataReader()
        self.analyzer = AFM_Curve_analyzer()
        self.plot_configs = AFM_Result_Visualizer()

        # 3. Napariビューア本体を作成
        self.viewer = napari.Viewer()
        
        # 4. カスタムGUIウィジェット（ボタン、プロット、テキスト）を作成してドッキング
        self._add_widgets()
        
        # 5. (★最重要) マウスのドラッグ/クリックイベントをコールバック関数に接続
        self.viewer.mouse_drag_callbacks.append(self.on_mouse_click)

    def _add_widgets(self):
        """Napariビューアの右側にカスタムウィジェットを追加する"""
        
        # --- 1. 操作パネル (PyQt) ---
        self.control_widget = QWidget()
        control_layout = QVBoxLayout()
        
        # ロードボタン
        btn_load = QPushButton("解析済みフォルダをロード")
        btn_load.clicked.connect(self.load_analysis_data) # ボタンクリック時の動作を接続
        control_layout.addWidget(btn_load)
        
        # マップ選択ドロップダウン
        self.map_selector = QComboBox()
        self.map_selector.addItems(["(データ未ロード)"])
        self.map_selector.currentTextChanged.connect(self.display_map) # 選択変更時の動作を接続
        control_layout.addWidget(QLabel("表示するマップ:"))
        control_layout.addWidget(self.map_selector)
        
        self.control_widget.setLayout(control_layout)
        self.viewer.window.add_dock_widget(self.control_widget, area='right', name='コントロール')

        # --- 2. Matplotlibプロット (napari-matplotlib) ---
        self.plot_widget = NapariMPLWidget(self.viewer) # <--- ★ self.viewer を渡す
        self.plot_fig = self.plot_widget.figure
        self.plot_ax = self.plot_fig.add_subplot(111) # <--- ★ add_subplot(111) に戻す
        self.plot_ax.set_title("マップをクリック")
        self.plot_ax.set_xlabel("Z Distance (nm)")
        self.plot_ax.set_ylabel("Force (nN)")
        self.plot_ax.grid(True)
        self.viewer.window.add_dock_widget(self.plot_widget, area='right', name='フォースカーブ')
        
        # --- 3. 解析結果テキスト (PyQt) ---
        self.text_widget = QWidget()
        text_layout = QVBoxLayout()
        self.text_label = QLabel("マップをクリックすると解析結果が表示されます...")
        self.text_label.setWordWrap(True) # 自動折り返し
        self.text_label.setStyleSheet("font-family: monospace;") # 等幅フォント
        text_layout.addWidget(self.text_label)
        self.text_widget.setLayout(text_layout)
        self.viewer.window.add_dock_widget(self.text_widget, area='right', name='解析結果')

    def load_analysis_data(self):
        """「ロード」ボタンが押されたときの処理。Streamlit版の load_analysis_data とほぼ同じ"""
        
        # PyQtのファイルダイアログでフォルダを選択
        folder_path_str = QFileDialog.getExistingDirectory(None, "解析済みデータフォルダを選択")
        if not folder_path_str:
            return # ユーザーがキャンセルした
            
        print(f"Loading data from: {folder_path_str}")
        try:
            folder_path = Path(folder_path_str)
            
            # (a) NPZ ロード
            base_filename = folder_path.name
            npz_filename = f"{base_filename}_analysis_data.npz"
            npz_path = folder_path / "AFM_Analysis_Results" / npz_filename
            self.data["analysis_results"] = np.load(npz_path)
            self.data["available_maps"] = list(self.data["analysis_results"].keys())
            
            # (b) メタデータ ロード
            self.data["metadata"] = self.reader.read_config_only(str(folder_path), invols=100.0)
            self.data["folder_path"] = str(folder_path)
            
            # (c) KDTree 構築
            x_m = self.data["metadata"]['xsensor']
            y_m = self.data["metadata"]['ysensor']
            xy_um = np.column_stack((x_m * 1e6, y_m * 1e6))
            self.data["xy_coords_um"] = xy_um
            self.data["kdtree"] = build_kdtree(xy_um)
            
            # (d) マップ範囲
            xmin, xmax = xy_um[:, 0].min(), xy_um[:, 0].max()
            ymin, ymax = xy_um[:, 1].min(), xy_um[:, 1].max()
            self.data["map_extent"] = [xmin, xmax, ymin, ymax]

            # (e) TDMSファイルを開き、オブジェクトを保持 (高速化のキモ)
            tdms_path = folder_path / "ForceCurve.tdms"
            if self.data["tdms_file"]: self.data["tdms_file"].close() # 古いファイルが開いていれば閉じる
            self.data["tdms_file"] = TdmsFile.open(str(tdms_path))
            self.data["deflection_ch"] = self.data["tdms_file"]["Forcecurve"]["Deflection"]
            self.data["ztip_ch"] = self.data["tdms_file"]["Forcecurve"]["ZTip_input"]
            self.data["zsensor_ch"] = self.data["tdms_file"]["Forcecurve"]["ZSensor"]
            
            # (f) ヒステリシスカーブもキャッシュ
            self.data["hyst_curve"] = self.reader.read_hysteresis_curve(
                self.data["metadata"], script_file_path=__file__
            )

            print("データロード完了。")
            
            # (g) UIを更新
            self.map_selector.clear()
            self.map_selector.addItems(self.data["available_maps"])
            # (ロードが完了したら自動的に最初のマップを表示)
            
        except Exception as e:
            print(f"データロード中にエラー発生: {e}")
            self.text_label.setText(f"データロード中にエラー発生:\n{e}")

    def display_map(self, map_key: str):
        """ドロップダウンでマップが選択されたときの処理"""
        if not map_key or map_key == "(データ未ロード)" or not self.data["analysis_results"]:
            return
            
        print(f"マップ表示を更新: {map_key}")
        
        # (a) NPZからZ値を取得
        z_values = self.data["analysis_results"][map_key]
        plot_config = self.plot_configs._get_plot_config(map_key)
        
        # (b) マップ補間 (Streamlit版と同じ)
        grid_size = (256, 256) # Napariは高解像度でも高速
        map_grid = generate_interpolated_map(
            self.data["xy_coords_um"], 
            z_values,
            grid_size=grid_size
        )
        
        # (c) データ変換 (Logなど)
        conversion = self.plot_configs.UNIT_CONVERSION.get(map_key, 1.0)
        plot_data = map_grid * conversion
        if plot_config['log_transform']:
            plot_data = np.log10(np.maximum(plot_data, 1e-12))
            
        # (d) カラーマップ名
        cmap_name = plot_config['cmap']
        if cmap_name == 'afmhot': napari_cmap = 'hot'
        elif cmap_name == 'afmhot_r': napari_cmap = 'hot_r'
        else: napari_cmap = cmap_name

        # (e) Napariの座標系に合わせてスケーリング
        xmin, xmax, ymin, ymax = self.data["map_extent"]
        scale = ((ymax - ymin) / grid_size[0], (xmax - xmin) / grid_size[1])
        translate = (ymin, xmin)

        # (f) Napariビューアに画像レイヤーとして追加
        try:
            # 既に "AFM Map" レイヤーがあれば削除
            self.viewer.layers.pop("AFM Map")
        except KeyError:
            pass # 初回ロード
            
        self.viewer.add_image(
            plot_data,
            name="AFM Map",
            colormap=napari_cmap,
            scale=scale,
            translate=translate
        )

    def on_mouse_click(self, viewer, event):
        """
        Napariビューアのマウスイベント(ドラッグ/プレス/リリース)を処理する。
        """
        # (a) 'mouse_press' (クリック開始) イベント以外は無視
        if event.type != 'mouse_press':
            return

        # (b) 左クリック以外は無視
        if event.button != 1:
            return
            
        # (c) データがロードされていなければ無視
        if not self.data["kdtree"]:
            return
            
        # (d) (★バグ修正) event.position は既にワールド座標 (物理座標)
        #    座標変換は一切不要。
        click_world_coords = event.position
        
        # Napariは (y, x) 順で座標を返す
        click_y, click_x = click_world_coords
        
        # (e) 軽量解析を実行
        try:
            # KDTree.query は (x, y) 順で受け取る
            data_obj, idx = self.get_clicked_curve_data_fast(click_x, click_y)
            
            # (f) Matplotlibプロットを更新
            self._update_plot(data_obj, idx)
            
            # (g) テキストを更新
            self._update_text(idx)
            
        except Exception as e:
            # 座標範囲外クリックなどでKDTreeが失敗した場合は無視する
            pass

    def get_clicked_curve_data_fast(self, click_x, click_y):
        """ Streamlit版の get_clicked_curve_data と全く同じ（高速版）"""
        
        # (1) KDTreeで最近傍インデックス (idx) を検索
        dist, nearest_idx = self.data["kdtree"].query([[click_x, click_y]], k=1)
        idx = nearest_idx[0][0]

        # (2) 必要なメタデータを self.data から取得
        metadata = self.data["metadata"]
        N_points = metadata['FCあたりのデータ取得点数']
        
        # (3) 高速スライス処理 (self.data から)
        start_idx = idx * N_points
        end_idx = start_idx + N_points
        
        deflection_data = self.data["deflection_ch"][start_idx:end_idx]
        ZTip_input_data = self.data["ztip_ch"][start_idx:end_idx]
        Zsensor_data = self.data["zsensor_ch"][start_idx:end_idx]
        
        xsensor = metadata['xsensor'][idx]
        ysensor = metadata['ysensor'][idx]
        
        hyst_curve = self.data["hyst_curve"] # キャッシュから取得

        # (4) AFMDataオブジェクトをその場で生成
        data_obj = AFMData(
            raw_deflection=deflection_data,
            raw_ztip=ZTip_input_data,
            raw_zsensor=Zsensor_data,
            metadata_ref=metadata,
            folder_path=self.data["folder_path"],
            hyst_curve=hyst_curve,
            xsensor=xsensor,
            ysensor=ysensor
        )
        
        # (5) UI表示用の軽量解析を実行
        self.analyzer.analyze_for_display(data_obj)
        
        return data_obj, idx

    def _update_plot(self, data_obj, idx):
        """MatplotlibのAxesを更新する"""
        self.plot_ax.clear() # 既存のプロットを消去
        
        z_nm = data_obj.z_distance * 1e9
        force_nN = data_obj.force_corrected * 1e9
        self.plot_ax.plot(z_nm, force_nN, label="Force Curve")
        
        cp_idx = data_obj.contact_point_index
        if cp_idx != -1:
            self.plot_ax.scatter(
                z_nm[cp_idx], force_nN[cp_idx], 
                color='red', s=50, label="Contact Point", zorder=5
            )
            
        self.plot_ax.set_title(f"Force Curve (Index: {idx})")
        self.plot_ax.set_xlabel("Z Distance (nm)")
        self.plot_ax.set_ylabel("Force (nN)")
        
        # 凡例に白い背景色を指定する
        self.plot_ax.legend(facecolor='white') 
        self.plot_ax.grid(True)
        
        # Matplotlibのキャンバスを再描画
        self.plot_fig.canvas.draw_idle()

    def _update_text(self, idx):
        """PyQtのQLabelのテキストを更新する"""
        results = self.data["analysis_results"]
        results_text = f"""
Index: {idx}
X (µm): {self.data['xy_coords_um'][idx, 0]:.2f}
Y (µm): {self.data['xy_coords_um'][idx, 1]:.2f}
---
Young's (log(Pa)): {np.log10(results['youngs_modulus'][idx]):.2f}
Topography (µm): {results['topography'][idx] * 1e6:.2f}
Peak Force (nN): {results['peak_force'][idx] * 1e9:.2f}
Delta (nm): {results['delta'][idx] * 1e9:.2f}
Hysteresis (fJ): {results['hysteresis_area'][idx] * 1e15:.2f}
"""
        self.text_label.setText(results_text)
        
# --- アプリケーションの実行 ---
if __name__ == "__main__":
    
    # napariは内部でPyQtのQApplicationを管理する
    # napari.run() がイベントループを開始する
    print("AFM Napari Viewer を起動します...")
    viewer_instance = AFMViewer()
    napari.run()