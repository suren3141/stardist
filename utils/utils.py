import os, glob
import numpy as np
from pathlib import Path

def get_file_list(data_dir_list, file_type, img_path="images", ann_path='bin_masks', inst_path=None, filt=None):
    """
    """

    if isinstance(data_dir_list, str): data_dir_list = [data_dir_list]

    file_list = []

    if file_type == '.png':
        for dir_path in data_dir_list:
            ext = '*.png' if filt is None else f'{filt}.png'
            image_files =  sorted(glob.glob(os.path.join(dir_path, img_path, ext)))

            if ann_path is not None:
                ann_files =  sorted(glob.glob(os.path.join(dir_path, ann_path, ext)))
            else:
                ann_files = [None] * len(image_files)

            if inst_path is not None:
                ext = '*.tif' if filt is None else f'{filt}.tif'
                inst_files =  sorted(glob.glob(os.path.join(dir_path, inst_path, ext)))
            else:
                inst_files = [None] * len(image_files)

            files = list(zip(image_files, ann_files, inst_files))

            file_list.extend(files)
            # file_list.extend(image_files)

        file_list = sorted(file_list, key=lambda x:x[0])       
        # file_list.sort()  # to always ensure same input ordering
    
    else:
        raise NotImplementedError()
    
    # Make sure all file names are the same
    for f in file_list:
        x = [Path(i).stem for i in f if i is not None]
        assert len(set(x)) == 1 and x[0] != '', f"Issue with folder containing {f}"

    return file_list




def get_file_label(gt_dirs, gt=True, img_path=None, ann_path=None, inst_path=None, filt=None):
    file_list = []
    file_labels = []


    for k, v in gt_dirs.items():
        if img_path is not None:
            f = get_file_list(v, ".png", img_path=img_path, ann_path=ann_path, inst_path=inst_path, filt=filt)
        elif gt:
            f = get_file_list(v, ".png", inst_path='inst_masks', filt=filt)
        else:
            f = get_file_list(v, ".png", img_path="samples", ann_path="labels", filt=filt)
        file_list.extend(f)
        file_labels.extend([f"{k}"] * len(f))
    
    return file_list, file_labels



from PIL import Image

def read_img(filename, mode='RGB', size=None):
    img = Image.open(filename)
    img_rgb = img.convert(mode)
    if size is not None:
        if isinstance(size, int): size = (size, size)
        assert img_rgb.size == size
        img_rgb = img_rgb.resize(size)
    img_array = np.array(img_rgb)
    return img_array

def link(src, dst, remove_existing=False):
    if not os.path.exists(dst):
        os.symlink(src, dst)
    elif remove_existing:
        os.unlink(dst)
        os.symlink(src, dst)
    else:
        raise ValueError(f"path {dst} already exists")
