import yaml
from ultralytics import YOLO

# yolo8n.pt to tensorflow lite
model = YOLO('yolov8s_best.pt')
model.export(format='tflite')

# metadata.yaml to labels.txt
metadata = yaml.load(open('best_web_model\metadata.yaml', 'r'), Loader=yaml.SafeLoader)
labels = metadata['names']
with open('labels.txt', 'w') as f:
    for label in labels:
        f.write(labels[label] + '\n')