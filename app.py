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
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox,
    QSpinBox, QHBoxLayout, QCheckBox
)
from PyQt5.QtCore import QTimer
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
        # 現在表示中のフォースカーブのインデックス
        self.current_display_idx = None
        
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

        # フォースカーブのインデックス指定ウィジェット
        idx_layout = QHBoxLayout()
        idx_layout.addWidget(QLabel("Force Curve Index:"))
        self.idx_spinbox = QSpinBox()
        self.idx_spinbox.setMinimum(0)
        self.idx_spinbox.setMaximum(0)  # ロード時に更新
        self.idx_spinbox.setEnabled(False)
        # 値が変更されたとき、指定インデックスのフォースカーブを表示
        self.idx_spinbox.valueChanged.connect(self.on_idx_changed)
        idx_layout.addWidget(self.idx_spinbox)
        control_layout.addLayout(idx_layout)

        # --- analyze_for_display の各ステップを切り替えるチェックボックス ---
        control_layout.addWidget(QLabel("表示用解析ステップ:"))
        self.chk_convert = QCheckBox("Deflection -> Force (convert)")
        self.chk_convert.setChecked(True)
        control_layout.addWidget(self.chk_convert)
        self.chk_convert.stateChanged.connect(self.on_display_option_changed)

        self.chk_zdist = QCheckBox("Calculate Z distance (z_distance)")
        self.chk_zdist.setChecked(True)
        control_layout.addWidget(self.chk_zdist)
        self.chk_zdist.stateChanged.connect(self.on_display_option_changed)

        self.chk_baseline = QCheckBox("Baseline correction (force_corrected)")
        self.chk_baseline.setChecked(True)
        control_layout.addWidget(self.chk_baseline)
        self.chk_baseline.stateChanged.connect(self.on_display_option_changed)

        self.chk_cp = QCheckBox("Find contact/retract points (CP/RP)")
        self.chk_cp.setChecked(True)
        control_layout.addWidget(self.chk_cp)
        self.chk_cp.stateChanged.connect(self.on_display_option_changed)
        
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
            # インデックス指定ウィジェットを有効化して範囲を設定
            try:
                n_curves = len(self.data["metadata"]["xsensor"])
                if n_curves > 0:
                    self.idx_spinbox.setMaximum(max(0, n_curves - 1))
                    self.idx_spinbox.setValue(0)
                    self.idx_spinbox.setEnabled(True)
            except Exception:
                # メタデータが未整備の場合はスキップ
                pass
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
        # guard against zero extent (which causes singular transform errors)
        dx = max(1e-9, (xmax - xmin))
        dy = max(1e-9, (ymax - ymin))
        scale = (dy / grid_size[0], dx / grid_size[1])
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
        # 保存しておく: マップのグリッドサイズ/スケール/translate を later highlight に使う
        try:
            self.last_map_grid_size = grid_size
            self.last_map_scale = scale
            self.last_map_translate = translate
        except Exception:
            self.last_map_grid_size = None
            self.last_map_scale = None
            self.last_map_translate = None
        # マップを切り替えた場合は、もし現在表示中の idx があればそれをハイライト
        try:
            if getattr(self, 'current_display_idx', None) is not None:
                self.highlight_idx_on_map(self.current_display_idx)
        except Exception:
            pass

    def highlight_idx_on_map(self, idx):
        """マップ上に現在表示中の idx に対応する点をハイライト表示する。

        Napari の座標系は (y, x) の順を使うため、self.data['xy_coords_um'] の
        (x, y) を入れ替えて渡す。
        """
        try:
            coords = self.data.get('xy_coords_um')
            if coords is None or getattr(coords, 'size', 0) == 0:
                return
            if idx < 0 or idx >= len(coords):
                return
            x_um = float(coords[idx, 0])
            y_um = float(coords[idx, 1])
            # Use world coordinates (y, x) in same units as the image translate/scale
            # This is more robust across map changes: provide world coords and let Napari
            # place the points in the shared world coordinate system.
            point = [[y_um, x_um]]
            point_scale = None

            # compute a size that's appropriate for the current map (in screen px)
            size_px = self.compute_point_size()
            # target on-screen pixel size for cursor-like marker (approx mouse cursor)
            target_pixels = 5
            # estimate world-size equivalent using current map scale (if available)
            world_size = None
            try:
                if getattr(self, 'last_map_scale', None) is not None:
                    avg_world_per_pixel = float(np.mean(self.last_map_scale))
                    world_size = float(target_pixels) * avg_world_per_pixel
            except Exception:
                world_size = None

            # If a layer named 'Selected Point' exists, update it; otherwise add new points layer
            if 'Selected Point' in self.viewer.layers:
                layer = self.viewer.layers['Selected Point']
                try:
                    layer.data = point
                    # ensure the layer uses a screen/view-based size mode if available
                    try:
                        # try to apply a screen-based size first
                        if hasattr(layer, 'size'):
                            try:
                                layer.size = target_pixels
                            except Exception:
                                layer.size = size_px
                        # try several possible size_mode values to request screen/view sizing
                        for mode in ('view', 'screen', 'fixed', 'constant', 'data'):
                            if hasattr(layer, 'size_mode'):
                                try:
                                    layer.size_mode = mode
                                    break
                                except Exception:
                                    continue
                        # if size_mode didn't take effect, fall back to world-sized marker
                        if world_size is not None:
                            try:
                                layer.size = world_size
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    # Fall back to removing and re-adding layer
                    try:
                        self.viewer.layers.remove(layer)
                    except Exception:
                        pass
                    new_layer = self.viewer.add_points(point, name='Selected Point', size=size_px, face_color='red')
                    try:
                        # debug: print computed and applied size / mode
                        print(f"[debug] compute_point_size -> {size_px}")
                        if hasattr(new_layer, 'size'):
                            print(f"[debug] new_layer.size (after add) = {getattr(new_layer, 'size', None)}")
                        # try to set screen/view mode and a target pixel size
                        try:
                            if hasattr(new_layer, 'size'):
                                new_layer.size = target_pixels
                        except Exception:
                            pass
                        for mode in ('view', 'screen', 'fixed', 'constant', 'data'):
                            if hasattr(new_layer, 'size_mode'):
                                try:
                                    new_layer.size_mode = mode
                                    print(f"[debug] new_layer.size_mode = {mode}")
                                    break
                                except Exception:
                                    continue
                        # fallback: set world-size if we computed one
                        if world_size is not None:
                            try:
                                new_layer.size = world_size
                                print(f"[debug] new_layer.size (world fallback) = {world_size}")
                            except Exception:
                                pass
                    except Exception:
                        pass
            else:
                new_layer = self.viewer.add_points(point, name='Selected Point', size=size_px, face_color='red')
                try:
                    print(f"[debug] compute_point_size -> {size_px}")
                    if hasattr(new_layer, 'size'):
                        print(f"[debug] new_layer.size (after add) = {getattr(new_layer, 'size', None)}")
                    try:
                        if hasattr(new_layer, 'size'):
                            new_layer.size = target_pixels
                    except Exception:
                        pass
                    for mode in ('view', 'screen', 'fixed', 'constant', 'data'):
                        if hasattr(new_layer, 'size_mode'):
                            try:
                                new_layer.size_mode = mode
                                print(f"[debug] new_layer.size_mode = {mode}")
                                break
                            except Exception:
                                continue
                    if world_size is not None:
                        try:
                            new_layer.size = world_size
                            print(f"[debug] new_layer.size (world fallback) = {world_size}")
                        except Exception:
                            pass
                except Exception:
                    pass
            # ensure the Selected Point layer is on top (deferred)
            try:
                QTimer.singleShot(0, self.ensure_selected_point_on_top)
            except Exception:
                pass
        except Exception as e:
            print(f"highlight_idx_on_map error idx={idx}: {e}")

    def ensure_selected_point_on_top(self):
        """Remove and re-add Selected Point so it becomes the last layer (topmost).

        This is a robust way to ensure the point is drawn above image layers across
        different Napari versions and rendering backends.
        """
        try:
            if 'Selected Point' not in self.viewer.layers:
                return
            layer = self.viewer.layers['Selected Point']
            # capture current data and some display properties
            data = None
            try:
                data = layer.data.copy()
            except Exception:
                data = layer.data
            # compute a marker size based on the current map/grid and point density
            try:
                size = self.compute_point_size()
            except Exception:
                size = 3
            face_color = None
            try:
                face_color = layer.face_color
            except Exception:
                face_color = 'red'

            # move layer to last position (top) and set display properties
            def readd():
                try:
                    if 'Selected Point' in self.viewer.layers:
                        layer2 = self.viewer.layers['Selected Point']
                        # move to last index
                        try:
                            layers_list2 = list(self.viewer.layers)
                            if layer2 in layers_list2:
                                cur2 = layers_list2.index(layer2)
                                dest2 = len(layers_list2) - 1
                                if cur2 != dest2:
                                    self.viewer.layers.move(cur2, dest2)
                        except Exception:
                            pass

                        # apply size/mode settings without re-creating the layer
                        try:
                            print(f"[debug] readd compute_point_size -> {size}")
                            target_pixels = 5
                            world_size = None
                            try:
                                if getattr(self, 'last_map_scale', None) is not None:
                                    avg_world_per_pixel = float(np.mean(self.last_map_scale))
                                    world_size = float(target_pixels) * avg_world_per_pixel
                            except Exception:
                                world_size = None

                            if hasattr(layer2, 'size'):
                                try:
                                    layer2.size = target_pixels
                                    print(f"[debug] moved layer2.size (target pixels) = {getattr(layer2, 'size', None)}")
                                except Exception:
                                    try:
                                        layer2.size = size
                                    except Exception:
                                        pass

                            for mode in ('view', 'screen', 'fixed', 'constant', 'data'):
                                if hasattr(layer2, 'size_mode'):
                                    try:
                                        layer2.size_mode = mode
                                        print(f"[debug] moved layer2.size_mode = {mode}")
                                        break
                                    except Exception:
                                        continue

                            if world_size is not None:
                                try:
                                    layer2.size = world_size
                                    print(f"[debug] moved layer2.size (world fallback) = {world_size}")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    else:
                        # if the layer no longer exists, add it
                        try:
                            new_layer = self.viewer.add_points(data, name='Selected Point', size=size, face_color=face_color)
                            print(f"[debug] re-added Selected Point layer (fallback)")
                        except Exception:
                            pass
                except Exception:
                    pass

            QTimer.singleShot(0, readd)
        except Exception:
            pass

    def compute_point_size(self, min_px: int = 1, max_px: int = 40, fraction: float = 0.02):
        """Compute a sensible point marker size in pixels based on the
        currently displayed map. The heuristic is:

        - Use the last_map_grid_size (pixels) as a baseline: marker ~ fraction * avg(grid)
        - If point density is high (many coords), scale down the marker.
        - Clamp result to [min_px, max_px].

        This yields a stable onscreen size that adapts to map resolution
        and data density.
        """
        try:
            # baseline from grid size (average of height/width)
            if getattr(self, 'last_map_grid_size', None) is None:
                base = 256.0
            else:
                base = float(np.mean(self.last_map_grid_size))

            size = float(base) * float(fraction)

            # reduce size when there are many points
            n_points = 0
            coords = self.data.get('xy_coords_um')
            if coords is not None:
                try:
                    n_points = len(coords)
                except Exception:
                    n_points = 0

            if n_points > 0:
                # scale factor: for very many points make markers smaller
                density_scale = max(0.3, min(1.0, (1000.0 / float(n_points)) ** 0.5))
                size *= density_scale

            # shrink diameter to approximately 1/5 (~20%) as requested
            size *= 0.2

            size_px = int(max(min_px, min(max_px, round(size))))
            return size_px
        except Exception:
            return min_px

    def _bring_layer_to_front(self, layer_name: str):
        """指定したレイヤーをレイヤーリストの最前面（上）に移動する。"""
        try:
            layer = self.viewer.layers[layer_name]
            # find current index and move to last position
            layers_list = list(self.viewer.layers)
            cur_idx = layers_list.index(layer)
            # Move to last index so it's drawn on top (Napari draws later layers above earlier ones)
            dest_idx = len(layers_list) - 1
            if cur_idx != dest_idx:
                # Defer the move to the Qt event loop to avoid mutating the model during paint
                def do_move():
                    try:
                        layers_list2 = list(self.viewer.layers)
                        if layer in layers_list2:
                            cur2 = layers_list2.index(layer)
                            dest2 = len(layers_list2) - 1
                            if cur2 != dest2:
                                self.viewer.layers.move(cur2, dest2)
                    except Exception:
                        pass

                QTimer.singleShot(0, do_move)
        except Exception:
            pass

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
            # スピンボックスはクリックで更新された値を表示するが、
            # valueChangedシグナルを発火させないよう一時的にブロックする
            try:
                self.idx_spinbox.blockSignals(True)
                self.idx_spinbox.setValue(idx)
            except Exception:
                pass
            finally:
                try:
                    self.idx_spinbox.blockSignals(False)
                except Exception:
                    pass

            self._update_plot(data_obj, idx)

            # (g) テキストを更新
            self._update_text(idx)
            # 現在表示中のインデックスを記録
            try:
                self.current_display_idx = idx
            except Exception:
                pass

            # マップ上に選択点をハイライト
            try:
                self.highlight_idx_on_map(idx)
            except Exception:
                pass
            
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
        # 選択されたチェックボックスに従って解析を実行
        self.analyze_for_display(data_obj)
        
        return data_obj, idx

    def analyze_for_display(self, data_obj: AFMData):
        """チェックボックスの設定に従って表示用の解析ステップを実行する。

        依存関係を考慮して必要な前処理が自動的に呼ばれる。
        (convert) -> (z_distance) -> (baseline) -> (CP/RP)
        """
        # convert: raw_deflection -> force
        try:
            if self.chk_convert.isChecked():
                self.analyzer._convert_def_to_force(data_obj)
            else:
                # チェックが外れている場合は force を raw_deflection と同じにする
                data_obj.force = data_obj.raw_deflection.copy()
        except Exception:
            # 保険: 例外は無視して続行
            pass

        # z_distance
        try:
            if self.chk_zdist.isChecked():
                self.analyzer._calc_z_distance(data_obj)
            else:
                # チェックが外れている場合は z_distance を raw_ztip と同じにする
                data_obj.z_distance = data_obj.raw_ztip.copy()
        except Exception:
            pass

        # baseline (force_corrected) - depends on force
        try:
            if self.chk_baseline.isChecked():
                # ensure force exists
                if not hasattr(data_obj, 'force') or data_obj.force is None:
                    self.analyzer._convert_def_to_force(data_obj)
                self.analyzer._force_correct(data_obj)
            else:
                # チェックが外れている場合は force_corrected を force と同じにする
                if hasattr(data_obj, 'force') and data_obj.force is not None:
                    data_obj.force_corrected = data_obj.force.copy()
        except Exception:
            pass

        # CP/RP - depends on force_corrected and z_distance
        try:
            if self.chk_cp.isChecked():
                # ensure prerequisites
                if not hasattr(data_obj, 'force_corrected') or data_obj.force_corrected is None:
                    if not hasattr(data_obj, 'force') or data_obj.force is None:
                        self.analyzer._convert_def_to_force(data_obj)
                    self.analyzer._force_correct(data_obj)
                if not hasattr(data_obj, 'z_distance') or data_obj.z_distance is None:
                    self.analyzer._calc_z_distance(data_obj)
                self.analyzer._calc_CPandRP(data_obj)
        except Exception:
            pass


    def get_curve_data_by_index(self, idx):
        """指定したインデックスのフォースカーブデータを取得して解析オブジェクトを返す"""
        metadata = self.data["metadata"]
        N_points = metadata['FCあたりのデータ取得点数']

        start_idx = idx * N_points
        end_idx = start_idx + N_points

        deflection_data = self.data["deflection_ch"][start_idx:end_idx]
        ZTip_input_data = self.data["ztip_ch"][start_idx:end_idx]
        Zsensor_data = self.data["zsensor_ch"][start_idx:end_idx]

        xsensor = metadata['xsensor'][idx]
        ysensor = metadata['ysensor'][idx]

        hyst_curve = self.data["hyst_curve"]

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

        # 選択されたチェックボックスに従って解析を実行
        self.analyze_for_display(data_obj)
        return data_obj, idx

    def on_idx_changed(self, value):
        """スピンボックスでインデックスが変更されたときのハンドラ"""
        if not self.data.get("kdtree"):
            return
        try:
            data_obj, idx = self.get_curve_data_by_index(int(value))
            self._update_plot(data_obj, idx)
            self._update_text(idx)
            # 現在表示中のインデックスを更新
            self.current_display_idx = idx
            # マップ上に選択点をハイライト
            try:
                self.highlight_idx_on_map(idx)
            except Exception as e:
                print(f"on_idx_changed: highlight failed for idx={idx}: {e}")
        except Exception as e:
            # 範囲外や読み込み失敗はログに出す（無言で失敗しないようにする）
            print(f"on_idx_changed error for idx={value}: {e}")
            self.text_label.setText(f"指定インデックスの読み込みに失敗しました:\n{e}")
            return

    def on_display_option_changed(self, state):
        """チェックボックスの設定が変わったときに、現在表示中のインデックスを再描画する"""
        # 優先: current_display_idx があればそれを使う。なければスピンボックス値（有効時）を使う
        idx = None
        if getattr(self, 'current_display_idx', None) is not None:
            idx = self.current_display_idx
        elif hasattr(self, 'idx_spinbox') and self.idx_spinbox.isEnabled():
            idx = int(self.idx_spinbox.value())

        if idx is None:
            return

        try:
            data_obj, _ = self.get_curve_data_by_index(int(idx))
            self._update_plot(data_obj, int(idx))
            self._update_text(int(idx))
            # 保持
            self.current_display_idx = int(idx)
            # マップ上に選択点をハイライト
            try:
                self.highlight_idx_on_map(int(idx))
            except Exception as e:
                print(f"on_display_option_changed: highlight failed for idx={idx}: {e}")
        except Exception as e:
            print(f"on_display_option_changed: failed to redraw idx={idx}: {e}")
            # GUIにエラーを表示しておく
            try:
                self.text_label.setText(f"表示更新に失敗しました:\n{e}")
            except Exception:
                pass

    def _update_plot(self, data_obj, idx):
        """MatplotlibのAxesを更新する"""
        self.plot_ax.clear() # 既存のプロットを消去
        # X軸: 優先順位 z_distance -> raw_ztip
        x_arr = None
        if self.chk_zdist.isChecked():
            x_arr = data_obj.z_distance * 1e9
            x_label = "Z Distance (nm)"
            x_scale = 1e9
        else:
            x_arr = data_obj.raw_ztip
            x_label = "Z Tip (raw, V)"
            x_scale = 1.0


        # Y軸: 優先順位 force_corrected -> force -> raw_deflection
        y_arr = None
        if self.chk_convert.isChecked():
            y_arr = data_obj.force * 1e9
            y_label = "Force (nN)"
            y_scale = 1e9
        else:
            y_arr = data_obj.raw_deflection
            y_label = "Deflection (raw, V)"
            y_scale = 1.0


        # スケールしてプロット
        try:
            x_plot = x_arr * x_scale
            y_plot = y_arr * y_scale
            self.plot_ax.plot(x_plot, y_plot, label="Force Curve")
        except Exception as e:
            # プロットできない場合はログ出力して終了
            print(f"_update_plot: failed to plot data for idx={idx}: {e}")
            return

        # Contact point があれば表示（存在チェック）
        cp_idx = getattr(data_obj, 'contact_point_index', -1)
        try:
            if cp_idx is not None and cp_idx != -1 and cp_idx < len(x_plot):
                self.plot_ax.scatter(
                    x_plot[cp_idx], y_plot[cp_idx],
                    color='red', s=50, label="Contact Point", zorder=5
                )
        except Exception:
            # 無視して続行
            pass
            
        self.plot_ax.set_title(f"Force Curve (Index: {idx})")
        self.plot_ax.set_xlabel(x_label)
        self.plot_ax.set_ylabel(y_label)

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