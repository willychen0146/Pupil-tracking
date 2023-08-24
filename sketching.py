import cv2
from ultralytics import YOLO
import numpy as np
import torch

import os

def get_all_file_paths(directory: str) -> list:
    file_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:           
            file_paths.append([os.path.join(root, file), file[:-4]])
    
    return file_paths

    
def sketch_result(model, directory_path: str, folder_path: str):
    # Create the folder 'result' in the desired directory
    result_path = os.path.join(directory_path, 'result')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    file_paths = get_all_file_paths(folder_path)

    for path in file_paths:
        img= cv2.imread(path[0])

        results = model.predict(source=img.copy(), stream=True)

        for result in results:
            # get array results
            try:
                masks = result.masks.data
            
            except AttributeError:
                continue
            
            boxes = result.boxes.data
            # extract classes   
            clss = boxes[:, 5]
            
            # get indices of results where class is 0 (people in COCO)
            pupil_indices = torch.where(clss == 0)
            
            # use these indices to extract the relevant masks
            pupil_masks = masks[pupil_indices]
            
            # scale for visualizing results
            pupil_masks = torch.any(pupil_masks, dim=0).int() * 255

            # save to file
            cv2.imwrite(os.path.join(os.path.join(directory_path, 'result'), path[1] + ".png"), pupil_masks.cpu().numpy())