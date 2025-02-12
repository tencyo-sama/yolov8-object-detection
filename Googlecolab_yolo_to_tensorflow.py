!pip install ultralytics

import os
from ultralytics import YOLO
from google.colab import files

# best.pt をアップロード
uploaded = files.upload()
pt_model_path = list(uploaded.keys())[0]


# YOLOv8 の PyTorch モデルをロード
model = YOLO(pt_model_path)

# モデルの入力サイズ
imgsz = (640, 640)  # 必要に応じて変更

# TensorFlow.js 形式にエクスポート
model.export(format="tfjs", imgsz=imgsz)

# tfjs_model フォルダをダウンロード
!zip -r tfjs_model.zip {pt_model_path.replace('.pt', '_web_model')}  # フォルダ名を修正
files.download('tfjs_model.zip')