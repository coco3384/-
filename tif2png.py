import os
import cv2
import numpy as np
import rasterio
import glob
import argparse
from tqdm import tqdm

def min_max(array, lower_bound=0.02, upper_bound=0.98):
    array_sort = np.sort(array)
    min = int(array.size * lower_bound)
    max = int(array.size * upper_bound)
    return array_sort[min], array_sort[max]

def main(file_path, save_path,  flag='dmc', cumulative_count_cut=False):
    if cumulative_count_cut:
        save_path = save_path + '_' + flag + '_cum_cut'
    else:
        save_path = save_path + '_' + flag 
    os.makedirs(save_path, exist_ok=True)
    file_path = glob.glob(os.path.join(file_path, '*.tif'))
    for fp in tqdm(file_path):
        basename = os.path.basename(fp).split('.')[0] + '.png'
        sp = os.path.join(save_path, basename)
        img = rasterio.open(fp)
        png = np.zeros((*img.shape, 0))
        for ch in range(1, 4):
            band = img.read(ch)
            band_flatten = band.flatten()
            if cumulative_count_cut and basename.find('rgb') == -1:
                max = int(np.nanpercentile(band_flatten, 98))
                band[band > max] = max
            result = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            png = np.dstack((png, result))
        png = png.astype('float32')
        cv2.imwrite(sp, cv2.cvtColor(png, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path/to/data directory')
    parser.add_argument('--save_dir', default='output', help='path/to/save directory, default is <output_type> under current directory.')
    parser.add_argument('--cumulative_count_cut', action='store_true', help='do cumulative count cut or not')
    parser.add_argument('--type', default='dmc', help='ortho or dmc, default is dmc.')
    args = parser.parse_args()
    main(file_path=args.dataset, save_path=args.save_dir, flag=args.type, cumulative_count_cut=args.cumulative_count_cut)
