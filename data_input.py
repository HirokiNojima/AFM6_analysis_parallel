import os
import re
import numpy as np
from nptdms import TdmsFile
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from afm_data import AFMData
from pathlib import Path

class DataReader:
    """
    特定のファイル形式からデータを読み込み、AFMDataオブジェクトを生成するクラス。
    config.txtとForceCurve.tdmsの読み込みを担います。
    """
    @staticmethod
    def _parse_config(folder_path: str, invols: float) -> Dict[str, Any]:
        """config.txt を Shift_JIS で読み込み、辞書を返すプライベートメソッド。"""

        file_path = os.path.join(folder_path, "config.txt")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'指定されたファイルが見つかりません: {file_path}')

        with open(file_path, 'rb') as f:
            raw = f.read()
        text = raw.decode('shift_jis', errors='replace')

        result = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^(.*?):\s*(.*)$', line)
            if not m:
                continue
            key = m.group(1).strip()
            val = m.group(2).strip()

            if ',' in val:
                items = [v.strip() for v in val.split(',') if v.strip()!='']
                conv = []
                for it in items:
                    try:
                        num = float(it)
                        conv.append(int(num) if num.is_integer() else num)
                    except Exception:
                        conv.append(it)
                result[key] = conv
            else:
                try:
                    num = float(val)
                    result[key] = int(num) if num.is_integer() else num
                except Exception:
                    result[key] = val

        # InvOLSや装置定数を追加
        result['InvOLS'] = invols * 1e-9  # m/V
        result['DISTANCE_PER_VOLT'] = 30e-6  # m/V # センサ変換係数（固定値）
        if result['カンチレバー種類'] == 'AC40':
            result['SPRING_CONSTANT'] = 0.1  # N/m
        elif result['カンチレバー種類'] == 'AC240':
            result['SPRING_CONSTANT'] = 2.0  # N/m
        elif result['カンチレバー種類'] == 'AC160':
            result['SPRING_CONSTANT'] = 42.0  # N/m
        else:
            result['SPRING_CONSTANT'] = input('カンチレバーのばね定数(N/m)を入力してください: ')
        return result
    
    @staticmethod
    def read_hysteresis_curve(config, script_file_path= __file__) -> np.ndarray:
            frequency = config['周波数(Hz)']

            if frequency == 100.0:
                freq_path = "100Hz"
            elif frequency == 1000.0:
                freq_path = "1kHz"
            elif frequency == 2000.0:
                freq_path = "2kHz"
            elif frequency == 3000.0:
                freq_path = "3kHz"
            elif frequency == 4000.0:
                freq_path = "4kHz"
            elif frequency == 5000.0:
                freq_path = "5kHz"
            elif frequency == 6000.0:
                freq_path = "6kHz"
            else:
                raise ValueError("Unsupported frequency: {}".format(frequency))
            
            # スクリプトの絶対パスを取得し、シンボリックリンクを解決
            base_path = Path(script_file_path).resolve()
            # スクリプトが存在するディレクトリ
            script_dir = base_path.parent
            
            # 相対パスのセグメントを結合して最終的なパスを生成
            try:
                full_path = script_dir.joinpath("補正用データ", freq_path, "mean_FCdata.txt")
                hyst_curve = np.loadtxt(full_path)
            except FileNotFoundError:
                raise FileNotFoundError("補正用データが見つかりません: {}".format(full_path))
            return hyst_curve
    
    # 🌟 新規追加: 設定ファイルのみを読み込むメソッド
    @staticmethod
    def read_config_only(folder_path: str, invols: float) -> Dict[str, Any]:
        """config.txt と必要なセンサ位置を読み込み、メタデータ辞書を返す。"""
        cfg = DataReader._parse_config(folder_path, invols)
        sensor_positions = DataReader.read_xysensor(folder_path)
        cfg['xsensor'] = sensor_positions[0] * cfg['DISTANCE_PER_VOLT']
        cfg['ysensor'] = sensor_positions[1] * cfg['DISTANCE_PER_VOLT']
        return cfg

    @staticmethod
    def read_xysensor(folder_path: str) -> List[float]:
        """
        Xsensors.txt / Ysensors.txt からセンサ位置（単一の数値）を読み込んで [xsensor, ysensor] [m]を返す。
        """
        files = [("Xsensors.txt", "X"), ("Ysensors.txt", "Y")]
        values = {}
        for fname, key in files:
            path = os.path.join(folder_path, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(f'指定されたファイルが見つかりません: {path}')
            try:
                val = np.loadtxt(path)
            except Exception as e:
                raise RuntimeError(f'{path} の読み込みに失敗しました: {e}')
            # 単一値なら float に変換（配列の場合はそのまま返す）
            if hasattr(val, 'size') and val.size == 1:
                val = float(val)
            values[key] = val

        xsensor = values.get("X", 0.0)
        ysensor = values.get("Y", 0.0)
        return [xsensor, ysensor]
    
    @staticmethod
    def _read_tipoffsets(folder_path: str) -> np.ndarray:
        """Tipoffsets.txt からチップオフセット値を読み込むプライベートメソッド。"""
        file_path = os.path.join(folder_path, "ZTipoffsets.txt")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'指定されたファイルが見つかりません: {file_path}')
        try:
            tip_offsets = np.loadtxt(file_path) * 0.65e-6  # m単位に変換(カタログ値を用いて変更している)
        except Exception as e:
            raise RuntimeError(f'{file_path} の読み込みに失敗しました: {e}')
        return tip_offsets
    
    def read_batch_force_curves(self, folder_path: str, indices: List[int], metadata: Dict[str, Any]) -> List[AFMData]:
        """
        TDMSファイルを一度だけ開き、指定された複数のインデックスのデータをまとめて読み込む。
        これによりI/Oオーバーヘッドを劇的に削減する。
        """
        N_points = metadata['FCあたりのデータ取得点数']
        file_path = os.path.join(folder_path, "ForceCurve.tdms")
        
        # 共通データの準備（ヒステリシス、チップオフセットなど）
        hyst_curve = self.read_hysteresis_curve(metadata, script_file_path=__file__)
        all_tip_offsets = self._read_tipoffsets(folder_path)
        
        data_objects = []

        try:
            # ★ ファイルオープンは一度だけ
            with TdmsFile.open(file_path) as tdms_file:
                deflection_channel = tdms_file["Forcecurve"]["Deflection"]
                ZTip_input_channel = tdms_file["Forcecurve"]["ZTip_input"]
                Zsensor_channel = tdms_file["Forcecurve"]["ZSensor"]
                
                # ループ内でファイルハンドルを開いたまま読み込み続ける
                for index in indices:
                    try:
                        start_idx = index * N_points
                        end_idx = start_idx + N_points
                        
                        # チャンネルからスライスでデータ取得
                        # (nptdmsは賢いので、必要な部分だけディスクからシークして読みます)
                        deflection_data = deflection_channel[start_idx:end_idx]
                        ZTip_input_data = ZTip_input_channel[start_idx:end_idx]
                        Zsensor_data = Zsensor_channel[start_idx:end_idx]
                        
                        xsensor = metadata['xsensor'][index]
                        ysensor = metadata['ysensor'][index]
                        tip_offset = all_tip_offsets[index]

                        data_obj = AFMData(
                            raw_deflection=deflection_data,
                            raw_ztip=ZTip_input_data,
                            raw_zsensor=Zsensor_data,
                            metadata_ref=metadata,
                            folder_path=folder_path,
                            hyst_curve=hyst_curve,
                            xsensor=xsensor,
                            ysensor=ysensor,
                            tip_offset=tip_offset
                        )
                        data_objects.append(data_obj)
                    except Exception as e:
                        print(f"⚠️ Index {index} read failed inside batch: {e}")
                        data_objects.append(None) # 失敗時はNoneを入れておく
                        
            return data_objects
            
        except Exception as e:
            raise RuntimeError(f"TDMS batch open failed: {e}")
 
# --- 使用例 ---
if __name__ == '__main__':
    # 注意: 実行には 'config.txt' と 'ForceCurve.tdms' が格納されたフォルダパスが必要です
    # 例: folder_path = "./path/to/your/data"
    # 実際にはテスト用のダミーファイルを作成するか、実際のデータパスを指定してください。
    
    folder_path = r"D:\nojima\AFM6measurement\251029\1905_プログラムテスト"
    data_reader = DataReader()
    invols = 100.0  # InvOLSの例
    force_curves = data_reader.read_tdms_force_map(folder_path, invols)

    if force_curves:
        print(f"最初のカーブのデータ点数: {len(force_curves[0].raw_deflection)}")
        print(f"フォルダパス: {force_curves[0].xsensor}")
    pass


