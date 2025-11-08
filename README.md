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

## 🚀 Quickstart（最短で動かす手順）

1. リポジトリをクローンまたはフォルダに移動。
2. Python 仮想環境を作成して有効化（推奨）。
   - Windows (PowerShell):
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS / Linux:
     ```
     python -m venv .venv
     source .venv/bin/activate
     ```
3. 必要なライブラリをインストール（以下の「必須ライブラリ」参照）。
4. main.py の末尾にある実行ブロックを編集して、解析対象の folder_path 等を設定。
5. バッチ解析を実行:
   ```
   python main.py
   ```
6. 解析完了後、napari ビューアで結果を確認:
   ```
   python app.py
   ```

-----

## 📦 必須ライブラリ (Requirements)

このプロジェクトの実行には以下が必要です。PyPI 版のインストール例を示します。環境に応じて torch のインストール方法は異なります（以下参照）。

基本パッケージ:
```bash
pip install numpy nptdms scikit-learn joblib matplotlib "napari[pyqt5]" napari-matplotlib
```

torch（PyTorch）: GPU を使う場合は公式サイトのインストールコマンドを使ってください。
- Apple Silicon (MPS):
  ```
  pip install --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
  または公式ドキュメントの MPS 用指示に従ってください。
- NVIDIA (CUDA):
  公式インストールページで CUDA バージョンに合うコマンドを選択してください（例: `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118`）。
- CPU のみ:
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

注: 環境によっては napari の GUI バックエンドや依存関係が別途必要になることがあります（PyQt5 等）。

{ 追記: requirements.txt を使ったインストール方法 }
## 📦 必須ライブラリ (Requirements)

このプロジェクトの実行には以下が必要です。PyPI 版のインストール例を示します。環境に応じて torch のインストール方法は異なります（以下参照）。

基本パッケージ:
```bash
pip install numpy nptdms scikit-learn joblib matplotlib "napari[pyqt5]" napari-matplotlib
```

torch（PyTorch）: GPU を使う場合は公式サイトのインストールコマンドを使ってください。
- Apple Silicon (MPS):
  ```
  pip install --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
  または公式ドキュメントの MPS 用指示に従ってください。
- NVIDIA (CUDA):
  公式インストールページで CUDA バージョンに合うコマンドを選択してください（例: `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118`）。
- CPU のみ:
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

注: 環境によっては napari の GUI バックエンドや依存関係が別途必要になることがあります（PyQt5 等）。

{ requirements.txt を使う旨の説明と例 }
### requirements.txt の利用（オプション）
プロジェクト依存関係をまとめて管理したい場合は、プロジェクトルートに requirements.txt を置き、以下のようにして一括インストールできます。

例: requirements.txt（最小例）
```
numpy
nptdms
scikit-learn
joblib
matplotlib
napari[pyqt5]
napari-matplotlib
# torch は環境 (CPU/MPS/CUDA) によってインストール方法が異なるため、個別に指定を推奨します
```

インストールコマンド:
```bash
pip install -r requirements.txt
```

注: torch（PyTorch）はプラットフォーム依存で最適なホイールを選ぶ必要があるため、requirements.txt に固定する場合は環境に合わせたバージョンを明示してください（例: `torch==2.1.0+cu118` のように追加の index URL が必要になることがあります）。

-----

## 🏃 実行方法

### 1\. バッチ解析の実行 (main.py)

1. main.py ファイルをテキストエディタで開き、`if __name__ == '__main__':` ブロックのパラメータを設定します（例: folder_path, invols_nm_per_volt, map_grid_size, num_jobs）。
2. 例:
   ```python
   # main.py の末尾: 実行例
   if __name__ == '__main__':
       folder_path = r"C:\path\to\your\data\1506_0"
       invols_nm_per_volt = 25.0  # 例: 25 nm/V
       map_grid_size = [512, 512]
       num_jobs = -1  # -1 で全コア使用
       # ...existing code...
   ```
3. 実行:
   ```
   python main.py
   ```
4. 出力: `AFM_Analysis_Results` サブフォルダに `_analysis_data.npz` と PNG マップが生成されます。

### 2\. インタラクティブビューアの起動 (app.py)

1. ターミナルで:
   ```
   python app.py
   ```
2. Napari ウィンドウが開きます。右側のコントロールから「解析済みフォルダをロード」で、`main.py` の処理対象フォルダ（`AFM_Analysis_Results` が含まれる親）を選択します。
3. マップをクリックして個々のフォースカーブを表示します。

-----

## 🔧 トラブルシューティング（よくある問題）

- メモリ不足エラー:
  - 本ツールは個々のワーカーがディスクから1カーブ分のみを読み込む設計ですが、num_jobs を減らすとさらにメモリ負荷を下げられます（例: `num_jobs=2`）。
- torch が MPS / CUDA を認識しない:
  - PyTorch のインストールコマンドが環境に合っているか確認してください。`import torch; torch.backends.mps.is_available()` や `torch.cuda.is_available()` で確認できます。
- napari が起動しない:
  - PyQt5 等の GUI ライブラリが正しくインストールされているか確認してください。仮想環境で実行する場合、同じ環境で napari をインストールしてください。

## Contributing / 開発者向けメモ

- コードベースは以下のモジュールで構成されています:
  - afm_calclators.py, afm_data.py, data_input.py, processing.py, map_analyzer.py, interpolator.py, result_visualizer.py, app.py, main.py
- 新機能やバグ修正は Pull Request を作成してください。簡単な実行手順と再現データ（可能なら）を添えてください。
