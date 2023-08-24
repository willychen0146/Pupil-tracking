from ultralytics import YOLO
import argparse
from pathlib import Path

# Load a model
model = YOLO("yolov8n-seg.yaml")  # build a new model from scratch
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

def training():
    parser = argparse.ArgumentParser(description='training function of pupil tracking')
    parser.add_argument('-y', '--yaml_dir', type=Path, help='input directory of yaml file for YOLO model')
    parser.add_argument('-m', '--model_dir', type=Path, help='output directory of model file for YOLO model')
    args = parser.parse_args()
    
    # Use the model
    # Training
    results = model.train(model=args.model_dir, data=args.yaml_dir, epochs=300, patience=30, save=True, save_period=25, name='pupil_tracking', batch=32)

    # Validation
    results = model.val()  # evaluate model performance on the validation set
    
    # export
    success = YOLO("yolov8n-seg.pt").export(format="onnx")  # export a model to ONNX format
    
if __name__ == '__main__':
    training()