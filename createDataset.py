import os
import glob
import shutil
from pathlib import Path
from random import shuffle
import argparse

tr_sets = ['S1', 'S2', 'S3', 'S4']
tt_sets = ['S5', 'S6', 'S7', 'S8']

def img_labels(imgs):
    lbes = []
    for img in imgs:
        sp = img.split('/')
        sp[-3] = sp[-3] + '_labels'
        sp[-1] = sp[-1].split('.')[0] + '.txt'
        lbe = '/'.join(sp)
        lbes.append(lbe)
    return lbes


def create_dataset(output_dir, split, imgs, lbes=None):
    Path(output_dir / split / 'images').mkdir(parents=True, exist_ok=True)

    for num, img in enumerate(imgs):
        src_img = img
        name = '-'.join(img.split('/')[-3:]).split('.')[0]

        tgt_img = output_dir / split / 'images' / f'{name}.jpg'
        
        shutil.copyfile(src_img, tgt_img)

        if split != 'test':
            Path(output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

            src_lbe = lbes[num]
            tgt_lbe = output_dir / split / 'labels' / f'{name}.txt'
            shutil.copyfile(src_lbe, tgt_lbe)

def process_images(root_dir, output_dir, valid_ratio):
    
    if os.path.exists(output_dir):
        print('The directory has existed!')

    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        tr_va_imgs = []
        tt_imgs = []
        for tr_set in tr_sets:
            if os.path.exists(root_dir / tr_set):
                tr_va_imgs += glob.glob(f'{root_dir/tr_set}/*/*.jpg')

        for tt_set in tt_sets:
            if os.path.exists(root_dir / tt_set):
                tt_imgs += glob.glob(f'{root_dir/tt_set}/*/*.jpg')

        # shuffle the train/valid dataset
        shuffle(tr_va_imgs)

        # get the corresponding labels
        tr_va_lbes = img_labels(tr_va_imgs)

        n_valid = int(len(tr_va_imgs) * valid_ratio)
        tr_imgs = tr_va_imgs[n_valid:]
        va_imgs = tr_va_imgs[:n_valid]
        
        tr_lbes = tr_va_lbes[n_valid:]
        va_lbes = tr_va_lbes[:n_valid]

        ################################
        # ------ Ganzin-Dataset
        #  |
        #  |---- train
        #  |   |- images
        #  |   |- labels
        #  |
        #  |---- valid
        #  |   |- images
        #  |   |- labels
        #  |
        #  |---- test
        #      |- images
        #      |- labels

        # Path(output_dir / 'images').mkdir(parents=True, exist_ok=True)
        # Path(output_dir / 'labels').mkdir(parents=True, exist_ok=True)

        create_dataset(output_dir, 'train', tr_imgs, tr_lbes)
        create_dataset(output_dir, 'valid', va_imgs, va_lbes)
        create_dataset(output_dir, 'test', tt_imgs)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate labels for images in subdirectories.')
    parser.add_argument('-r', '--root_dir', type=Path, help='input directory containing subdirectories with images')
    parser.add_argument('-o', '--output_dir', type=Path, help='output directory for labels')
    parser.add_argument('-v', '--valid_ratio', type=float, help='the ratio of validation data')

    args = parser.parse_args()
    process_images(args.root_dir, args.output_dir, args.valid_ratio)