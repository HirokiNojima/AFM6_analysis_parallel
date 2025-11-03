# AFMフォースマップ高速解析ツールキット

このプロジェクトは、原子間力顕微鏡（AFM）のフォースマップモードで取得された大規模データセット（特にTDMSファイル形式）を高速に解析・可視化するためのPythonツールキットです。

主な機能は、バッチ処理による全フォースカーブの並列解析と、`napari`を使用したインタラクティブな解析結果のビューアです。

## ✨ 主な特徴

  * **メモリ効率の高い並列処理**:
    `joblib`によるマルチコアCPU処理を活用します。最大の特色は、巨大な`ForceCurve.tdms`ファイル全体をメモリに読み込むのではなく、各ワーカープロセスが必要なカーブデータ（インデックス）のみをディスクから都度読み込む「**分散I/O設計**」を採用している点です。これにより、PCのRAM容量を大幅に超える巨大なデータセット（数GB〜数十GB）でも安定して解析可能です。
  * **GPUアクセラレーション補間**:
    解析後の散布データ（例：各点でのヤング率）を`FastRBFInterpolator2D`（PyTorchバックエンド）を使用して高速に補間します。Apple Silicon (`mps`) および NVIDIA (`cuda`) GPUに自動対応し、高解像度（1024x1024など）の2Dマップを瞬時に生成します。
  * **インタラクティブなデータ探索**:
    `napari`ベースの専用ビューア (`app.py`) が付属しています。解析結果の2Dマップ（ヤング率、トポグラフィーなど）をクリックすると、そのピクセルに対応する**生のフォースカーブ**と**解析結果**（接触点など）が即座にプロットされ、直感的なデータ検証が可能です。
  * **高度な物理モデルと補正**:
      * **ヒステリシス補正**: 実測のマスターカーブ (`mean_FCdata.txt`) に基づき、ピエゾの非線形ヒステリシスを補正します (`afm_calclators.correct_hyst`)。
      * **自動ベースライン補正**: カーブ全体で最もノイズが小さい（標準偏差が低い）領域を自動検出し、高精度な線形ベースライン補正を行います (`afm_calclators.baseline_correction`)。
      * **自動接触点検出**: ベースラインのノイズレベル（標準偏差）に基づいた閾値処理により、ロバストな接触点検出を実現します (`afm_calclators.calc_touch_point`)。
      * **物理量計算**: Sneddonモデル（円錐圧子）によるヤング率計算 (`calc_young_snnedon`)、ピークフォース、ヒステリシス面積などを算出します。

-----

## ⚙️ ワークフローとアーキテクチャ

本ツールキットは、大きく分けて「**1. 解析バッチ処理**」と「**2. インタラクティブ表示**」の2つのワークフローで構成されます。

### 1\. 解析バッチ処理 (main.py)

これは、生データフォルダを処理し、解析結果をファイルに出力するメインのスクリプトです。

1.  **メタデータ読み込み**: `main.py`が起動し、`DataReader.read_config_only`を呼び出します。`config.txt`やセンサ位置ファイルから、マップ寸法 ($N_x, N_y$) やばね定数などのメタデータのみを読み込みます。**この時点では、巨大なフォースカーブデータ（TDMS）は一切メモリにロードされません。**
2.  **タスク分散**: `AFM_Map_Analyzer_Joblib`が、処理すべきカーブの総数（例：1024点）に基づき、[0, 1, 2, ..., 1023] という**インデックスのリスト**を作成します。このリストが`joblib`の並列ワーカーに分割・配布されます。
3.  **並列実行 (ワーカー処理)**:
      * 各ワーカーは、割り当てられたインデックス（例：[128, 129, ...]）を受け取ります。
      * インデックスごとに `DataReader.read_single_force_curve` を呼び出し、TDMSファイルから**その1カーブ分のデータだけ**をディスクから読み込みます。
      * `AFM_Curve_analyzer.analyze_single_curve` が、ヒステリシス補正、ベースライン補正、接触点検出、ヤング率計算など、すべての物理解析を実行します。
      * 解析完了後、`AFMData.clear_raw_data()` が呼び出され、生データ配列（`raw_deflection`など）をメモリから破棄します。最終的な計算結果（ヤング率=10.5 GPaなど、スカラ値のみ）が保持されます。
4.  **結果集約**: メインプロセスは、全ワーカーからスカラ値のみが残った軽量な`AFMData`オブジェクトのリストを収集します。
5.  **エクスポート**: `AFM_Result_Visualizer`がこのリストを受け取り、以下の2種類のファイルを生成します。
      * **NPZ (1D配列)**: 全点の解析結果（`youngs_modulus`, `topography`など）を1次元配列として `_analysis_data.npz` に保存します。
      * **PNG (2Dマップ)**: `FastRBFInterpolator2D`を使用し、各物理量の高解像度2Dマップ画像（.png）を生成・保存します。

### 2\. インタラクティブ表示 (app.py)

これは、解析バッチ処理（`main.py`）によって生成された結果を `napari` で対話的に閲覧するためのGUIアプリケーションです。

1.  **結果ロード**: `app.py`を起動し、GUIの「解析済みフォルダをロード」ボタンから、`main.py`で処理した**元のデータフォルダ**（例: `1506_0`）を選択します。
2.  **データ準備**:
      * `AFM_Analysis_Results` サブフォルダから `_analysis_data.npz` を読み込みます。
      * 元の `ForceCurve.tdms` ファイルへの**ファイルハンドル**を開きます（データはまだ読み込みません）。
      * 高速な空間検索のため、`xsensor`, `ysensor` 座標からKDTreeを構築します。
3.  **マップ表示**: ドロップダウンから 'youngs\_modulus' などを選択すると、NPZの1DデータをRBF補間し、`napari`の画像レイヤーとして表示します。
4.  **インタラクション**:
      * ユーザーがマップ上をクリックします。
      * クリック座標 `(x, y)` から、KDTreeが最も近い**データインデックス**（例：インデックス 512）を瞬時に特定します。
      * `get_clicked_curve_data_fast` 関数が、開いているTDMSファイルのハンドルを使い、**インデックス 512 の生カーブデータのみ**をディスクから読み込みます。
      * `AFM_Curve_analyzer.analyze_for_display`（軽量版アナライザ）が実行され、カーブと接触点がMatplotlibウィジェットにプロットされます。

-----

## 📦 必須ライブラリ (Requirements)

このプロジェクトの実行には、以下のPythonライブラリが必要です。

```bash
# 基本的な数値計算・データI/O
numpy
nptdms
scikit-learn
joblib

# GPU補間 (RBF)
torch

# GUIとプロット
matplotlib
napari[pyqt5]
napari-matplotlib
```

`pip` を使用してインストールできます。

```bash
pip install numpy nptdms scikit-learn joblib torch matplotlib "napari[pyqt5]" napari-matplotlib
```

-----

## 🏃 実行方法

### 1\. バッチ解析の実行 (main.py)

1.  `main.py` ファイルをテキストエディタで開きます。

2.  ファイル末尾の `if __name__ == '__main__':` ブロック内を編集します。

      * `folder_path`: 解析したいデータフォルダ（`config.txt` や `ForceCurve.tdms` がある場所）のパスを指定します。
      * `invols_nm_per_volt`: 使用したカンチレバーの光てこ感度（InvOLS）を `[nm/V]` 単位で指定します。
      * `map_grid_size`: 出力するPNGマップの解像度（例: `[512, 512]`）を指定します。
      * `num_jobs`: 並列処理に使用するCPUコア数（`-1` で全コア使用）を指定します。

3.  ターミナルで `main.py` を実行します。

    ```bash
    python main.py
    ```

4.  解析が完了すると、`folder_path` 内に `AFM_Analysis_Results` というサブフォルダが作成され、その中にNPZファイルとPNGマップ画像が保存されます。

### 2\. インタラクティブビューアの起動 (app.py)

1.  ターミナルで `app.py` を実行します。

    ```bash
    python app.py
    ```

2.  Napariウィンドウが起動します。

3.  右側の「コントロール」パネルにある **「解析済みフォルダをロード」** ボタンをクリックします。

4.  `main.py` で解析した**元のデータフォルダ**（`AFM_Analysis_Results` フォルダが *含まれている* 親フォルダ）を選択します。

5.  「表示するマップ」ドロップダウンから 'youngs\_modulus' などを選択すると、2Dマップが表示されます。

6.  マップ上をクリックすると、右側の「フォースカーブ」と「解析結果」パネルがリアルタイムで更新されます。

-----

## 🗂️ ファイル構成

```
.
├── afm_calclators.py    # 物理計算・補正関数群 (Hertz, Sneddon, ベースライン補正など)
├── afm_data.py          # AFMDataクラス (単一カーブのデータコンテナ)
├── app.py               # 🚀 インタラクティブNapariビューア
├── data_input.py        # DataReaderクラス (config.txt, TDMSファイル I/O担当)
├── interpolator.py      # FastRBFInterpolator2Dクラス (PyTorchによるRBF補間)
├── main.py              # 🚀 バッチ解析実行スクリプト (これを実行する)
├── map_analyzer.py      # AFM_Map_Analyzer_Joblibクラス (並列処理オーケストレータ)
├── processing.py        # AFM_Curve_analyzerクラス (単一カーブの解析パイプライン)
└── result_visualizer.py # AFM_Result_Visualizerクラス (PNG, NPZへの保存・プロット)
```
