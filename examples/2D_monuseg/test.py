from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
from tqdm import tqdm
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

import glob, os, sys
sys.path.append('/workspace/stardist')
sys.path.append('/workspace/hover_net')

from hover_net.compute_stats import get_dice_1

import json


np.random.seed(42)
lbl_cmap = random_label_cmap()

from stardist.matching import matching

from PIL import Image
from train import *


def test_single_model(basedir, out_name, X_test, Y_test):

    model = StarDist2D(None, name=out_name, basedir=basedir)

    Y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_test)]
    
    dice = get_dice_1(Y_test, Y_pred)

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(Y_test, Y_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

    out = {}
    out['dice'] = dice
    for t, x in zip(taus, stats):
        out[t] = x._asdict()

    out_path = os.path.join(basedir, out_name, 'stats.json')

    with open(out_path, 'w+') as f:
        json.dump(out, f)

    print(stats[taus.index(0.5)])


def main_test(models):

    basedir='/mnt/dataset/stardist/models_monuseg'

    use_inst_mask = True

    sem2inst = sem_to_inst_map

    n_channel = 3
    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    img_preprocess = lambda x : normalize(x,1,99.8,axis=axis_norm)

    if use_inst_mask:
        label_preprocess = lambda x : fill_label_holes(x)
    else:
        label_preprocess = lambda x : fill_label_holes(sem2inst(x))

    x_id = 0
    y_id = 2 if use_inst_mask else 1
    mask_dtype = 'I' if use_inst_mask else 'L'

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = gputools_available()
    print(use_gpu)

    allocated_mem = min(1e10, get_total_mem())
    print("Allocated memory:", allocated_mem)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8, total_memory=allocated_mem)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    test_dirs = {
        "test": ["/mnt/dataset/MoNuSeg/patches_valid_inst_256x256_128x128/MoNuSegTestData"],
    }

    test_file_list, _ = get_file_label(test_dirs, gt=True)

    X_test = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), test_file_list))
    Y_test = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), test_file_list))

    for out_name in models:
        assert os.path.exists(os.path.join(basedir, out_name))
        test_single_model(basedir, out_name, X_test, Y_test)

if __name__ == "__main__":

    models = ['stardist_25gt_25syn.x5_inst', 'stardist_25gt_25syn_inst', 'stardist_25gt_25syn.x2_inst', 'stardist_25gt_25syn.x3_inst', 'stardist_25gt_25syn.x4_inst', 'stardist_25gt_25syn.x5_inst']

    main_test(models)

