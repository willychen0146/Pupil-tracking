import alphashape
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import shutil
import cv2

def main(args):
    rawdata_dir = Path(args.rawdata_dir)
    pred_img_dir = Path(args.pred_img_dir)
    pred_txt_dir = Path(args.pred_txt_dir)
    output_dir = Path(args.output_dir)

    sets = ['S5', 'S6', 'S7', 'S8']
    h, w, c = 480, 640, 3

    output_dir.mkdir(parents=True, exist_ok=True)

    for s in sets:
        output_set_dir = output_dir / s
        output_set_dir.mkdir(parents=True, exist_ok=True)

        set_dir = rawdata_dir / s
        raw_subdirs = [subdir for subdir in set_dir.iterdir() if subdir.is_dir()]
        
        for subdir in raw_subdirs:
            output_subdir = output_set_dir / subdir.name
            output_subdir.mkdir(parents=True, exist_ok=True)
            n_img = len(list(subdir.iterdir()))

            with open(output_subdir / 'conf.txt', 'w') as f:

                for index in range(n_img):
                    pred_txt_path = pred_txt_dir / (s + '-' + subdir.name + '-' + str(index) + '.txt')
                    pred_img_path = pred_img_dir / (s + '-' + subdir.name + '-' + str(index) + '.png')
                    tgt_img_path = output_subdir / (str(index) + '.png')

                    if pred_txt_path.is_file():
                        f.write('1.0\n')
                        # with open(pred_path, 'r') as pfile:
                        #     conf = pfile.readline().strip().split()[-1]
                        #     f.write(f'{conf}\n')
                    else:
                        f.write('0.0\n')

                    if pred_img_path.is_file():
                        shutil.copyfile(pred_img_path, tgt_img_path)
                    else:
                        img = np.zeros((h, w, c), np.uint8)
                        cv2.imwrite(str(tgt_img_path), img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the confidence score')
    parser.add_argument('-r', '--rawdata_dir', type=str, help='directory containing raw data (S1, S2, ...)')
    parser.add_argument('-t', '--pred_txt_dir', type=str, help='directory containing the predicted txt files')
    parser.add_argument('-i', '--pred_img_dir', type=str, help='directory containing the predicted image files')
    parser.add_argument('-o', '--output_dir', type=str, default='solution', help='output directory for confidence')

    args = parser.parse_args()
    main(args)