from ultralytics import YOLO
from sketching import sketch_result
import argparse
from pathlib import Path

def testing():
    parser = argparse.ArgumentParser(description='testing function of pupil tracking')
    parser.add_argument('-o', '--output_dir', type=Path, help='output directory of testing')
    parser.add_argument('-m', '--model_dir', type=Path, help="directory of model's pt file")
    parser.add_argument('-t', '--test_dir', type=Path, help='input directory of organized test data for YOLO model')
    args = parser.parse_args()
    
    # directory path
    directory_path = args.output_dir
    model_path = args.model_dir
    folder_path = args.test_dir
    
    # load model
    model = YOLO(model_path)
    
    results = model.predict(source=folder_path, save=True, show_labels=True, show_conf=True, stream=True, save_txt=True, save_conf=True)

    for _ in results:
        pass
    
    # produce result
    sketch_result(model, directory_path, folder_path)
    
if __name__ == "__main__":
    testing()


    