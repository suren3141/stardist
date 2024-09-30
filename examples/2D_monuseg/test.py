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

from hover_net.metrics.stats_utils import get_dice_1
from metrics import evaluate_instance_f1

import json


np.random.seed(42)
lbl_cmap = random_label_cmap()

from stardist.matching import matching

from PIL import Image
from train import *


def test_single_model(basedir, model_name, X_test, Y_test, out_name = None):

    model = StarDist2D(None, name=model_name, basedir=basedir)

    Y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_test)]
    
    dice = [get_dice_1(y, y_) for y, y_ in zip(Y_test, Y_pred)]

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(Y_test, Y_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

    stats_test = [matching(y_t, y_p, thresh=0.5, criterion='iou', report_matches=False) for y_t,y_p in zip(Y_test, Y_pred)]

    out = {}
    out['dice'] = np.mean(dice)
    for t, x in zip(taus, stats):
        out[t] = x._asdict()

    out['sample'] = dict(
        dice = dice,
        stats = [s._asdict() for s in stats_test]
    )

    # instance_f1 = evaluate_instance_f1(Y_pred, Y_test)
    # out[f'inst_f1'] = instance_f1

    if out_name is None:
        out_name = model_name

    out_path = os.path.join(basedir, out_name)

    with open(os.path.join(out_path, 'stats.json'), 'w+') as f:
        json.dump(out, f)

    if False:
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

        for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
            ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax1.set_xlabel(r'IoU threshold $\tau$')
        ax1.set_ylabel('Metric value')
        ax1.grid()
        ax1.legend()

        for m in ('fp', 'tp', 'fn'):
            ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax2.set_xlabel(r'IoU threshold $\tau$')
        ax2.set_ylabel('Number #')
        ax2.grid()
        ax2.legend();        

        fig.savefig(os.path.join(out_path, 'plot.png'))
        plt.close()

    print(stats[taus.index(0.5)])


def main_test(models):

    syn_model = 'sdm'
    basedir=f'/mnt/dataset/stardist/{syn_model}_monuseg_v1.2a_SynPlusGT_filt'
    # basedir='/mnt/dataset/stardist/models_monuseg_v1.1'

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


    test_dirs = {
        # "test": ["/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/MoNuSegTestData"],
        "test" : ["/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/__ResNet50_umap_n_components_3_random_state_42_hdbscan_min_samples_10_min_cluster_size_50_v1.2/6/MoNuSegTestData/"],
    }

    test_file_list, _ = get_file_label(test_dirs, gt=True)

    X_test = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), tqdm(test_file_list)))
    Y_test = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), tqdm(test_file_list)))

    for ind, model_name in enumerate(models):
        # if ind in [0, 1, 2, 3] : continue
        out_name = os.path.join(model_name, "clus6")
        if os.path.exists(os.path.join(basedir, model_name)):
            test_single_model(basedir, model_name, X_test, Y_test)
        else:
            print(f"dir {os.path.join(basedir, model_name)} does not exist")

if __name__ == "__main__":

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = gputools_available()
    print(use_gpu)

    # Maximum memory to allocated in Mb
    allocated_mem = min(6000, get_total_mem())
    print("Allocated memory:", allocated_mem)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8, total_memory=allocated_mem, allow_growth=False)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    # models = ['stardist_128_128_FT.2']

    models = ['stardist_128_128_05gt_inst', 'stardist_128_128_05gt_05syn_inst', 'stardist_128_128_05gt_05syn.x2_inst', 'stardist_128_128_05gt_05syn.x3_inst', 'stardist_128_128_05gt_05syn.x4_inst', 'stardist_128_128_05gt_05syn.x5_inst']
    # models = ['stardist_25gt_inst', 'stardist_25gt_25syn_inst', 'stardist_25gt_25syn.x2_inst', 'stardist_25gt_25syn.x3_inst', 'stardist_25gt_25syn.x4_inst', 'stardist_25gt_25syn.x5_inst']
    # models_filt = ['stardist_25gt_25syn_inst_filt', 'stardist_25gt_25syn.x2_inst_filt', 'stardist_25gt_25syn.x3_inst_filt', 'stardist_25gt_25syn.x4_inst_filt', 'stardist_25gt_25syn.x5_inst_filt']
    # models_filt2 = ['stardist_05gt_inst', 'stardist_05gt_05syn_inst', 'stardist_05gt_05syn.x2_inst', 'stardist_05gt_05syn.x3_inst', 'stardist_05gt_05syn.x4_inst', 'stardist_05gt_05syn.x5_inst']

    main_test(models)

