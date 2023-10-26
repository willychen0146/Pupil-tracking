import alphashape
import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def create_label(image_path, label_path):
    img_arr = np.asarray(Image.open(image_path))
    h, w, c = img_arr.shape

    img_R = img_arr[:,:,0] # get the red channel of image
    y, x = np.nonzero(img_R) # take the nonzero coordinates of img_R

    xy = np.stack([x, y], axis=1, dtype='int64')
    geoshape = alphashape.alphashape(xy, alpha=1.0)
    try:
        geoshape_x, geoshape_y = geoshape.exterior.coords.xy
        geoshape_x = np.asarray(geoshape_x, dtype='int64')
        geoshape_y = np.asarray(geoshape_y, dtype='int64')
        label_line = '0 ' + ' '.join([f'{cord[0]/w} {cord[1]/h}' for cord in zip(geoshape_x, geoshape_y)])
    except:
        label_line = ''
        print(image_path) # display the path of images with no labels

    # There may be a better way to do it, but this is what I have found so far
    # cords = list(features.shapes(arr, mask=(arr >0)))[0][0]['coordinates'][0]
    # label_line = '0 ' + ' '.join([f'{int(cord[0])/arr.shape[0]} {int(cord[1])/arr.shape[1]}' for cord in cords])

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open('w') as f:
        f.write(label_line)

def process_images(input_dir, output_dir):
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Recursively process images in input directory
    for img_path in input_dir_path.rglob('*.png'):
        if img_path.is_file():
            output_subdir_path = output_dir_path / img_path.relative_to(input_dir_path).parent
            output_subdir_path.mkdir(parents=True, exist_ok=True)

            label_path = output_subdir_path / f'{img_path.stem}.txt'
            create_label(img_path, label_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate labels for images in subdirectories.')
    parser.add_argument('-i', '--input_dir', type=str, help='input directory containing subdirectories with images')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory for labels')

    args = parser.parse_args()
    process_images(args.input_dir, args.output_dir)