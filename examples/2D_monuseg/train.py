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

import cv2

# sys.path.append('/workspace/hover_net')

import glob, os, sys
sys.path.append('/workspace/stardist')

from utils.utils import *



# from dataloader.utils import get_file_list
# from dataloader.train_loader import MoNuSegDataset

# from torch.utils.data import DataLoader

np.random.seed(42)
lbl_cmap = random_label_cmap()

from stardist.matching import matching
import nvidia_smi

def get_total_mem():
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)

    free_mem = info.free

    nvidia_smi.nvmlShutdown()

    return free_mem / 1e6

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y


def sem_to_inst_map(sem_map):
    """
    Convert semantic segmentation map to instance segmentaion map.
    Not accurate, but works ok
    """
    num_labels, labels = cv2.connectedComponents(sem_map)

    inst_map = np.zeros_like(sem_map)

    for label in range(1, num_labels):
        inst_map[labels==label] = label

    return inst_map

def sem_to_inst_stardist(sem_map):
    model_versatile = StarDist2D.from_pretrained('2D_versatile_fluo')
    labels, details = model_versatile.predict_instances(sem_map)
    return labels

def subsample(file_list, perc=None, size=None, prob=None):
    if perc is None and size is None: raise ValueError

    if perc is not None:
        np.random.seed(42)
        ind = np.random.choice(len(file_list), int(len(file_list)*float(perc)), replace=False, p=prob)
    if size is not None:
        np.random.seed(42)
        ind = np.random.choice(len(file_list), int(size), replace=False, p=prob)

    file_list = [file_list[i] for i in ind]

    return file_list

def get_train_val(gt_dirs):

    gt_file_list, label = get_file_label(gt_dirs, gt=True)

    # print(f"gt_file -> {len(gt_file_list)}")

    if "val" not in label:
        rng = np.random.RandomState(42)
        ind = rng.permutation(len(gt_file_list))
        n_val = max(1, int(round(0.15 * len(ind))))
        ind_train, ind_val = ind[:-n_val], ind[-n_val:]
        gt_file_val = [gt_file_list[i] for i in ind_val]
        gt_file_train = [gt_file_list[i] for i in ind_train]
    else:
        ind_train = [l for l in range(len(label)) if label[l]=="train"]
        ind_val = [l for l in range(len(label)) if label[l]=="val"]
        gt_file_train = [gt_file_list[i] for i in ind_train]
        gt_file_val = [gt_file_list[i] for i in ind_val]    

    print(f"\ttrain_file -> {len(gt_file_train)}")
    print(f"\tval_file -> {len(gt_file_val)}")

    return gt_file_train, gt_file_val

def filt_from_train(syn_file_list, gt_file_train, filter_from_train=False):
    if filter_from_train:
        syn_file_list_filtered = []

        for f in syn_file_list:
            valid = any([Path(f[0]).stem.startswith(Path(i[0]).stem) for i in gt_file_train])  # TODO : updated this
            if valid : syn_file_list_filtered.append(f)
    else:
        syn_file_list_filtered = syn_file_list

    return syn_file_list_filtered

def get_prob_from_json(image_list, json_name, clip_percentile=.5, prefix='./monuseg/test/syn'):
    import json
    from sklearn.preprocessing import MinMaxScaler
    with open(json_name, 'r') as f:
        data = json.load(f)

    def get_json_keys(image_list, prefix):   
        # TODO : Better implementation

        image_names = [i[0] for i in image_list]
        image_names = [os.path.split(i) for i in image_names]
        image_dir = [os.path.split(i[0]) for i in image_names]
        image_names = [os.path.join(x[1], y[1]) for (x, y) in zip(image_dir, image_names)]
        image_dir = [os.path.split(i[0]) for i in image_dir]
        image_names = [os.path.join(prefix, x[1], y) for (x, y) in zip(image_dir, image_names)]

        return image_names

    image_keys =  get_json_keys(image_list, prefix)
    score = [data[i] for i in image_keys]
    score = np.array(score)
    if clip_percentile:
        score = np.clip(score, 0, np.quantile(score, clip_percentile))
    scale = MinMaxScaler()
    score_scaled = scale.fit_transform(score.reshape(-1, 1))
    prob = 1 - score_scaled.flatten()
    return prob/np.sum(prob)


def main(out_name, filters=None, mult=None):

    use_inst_mask = True
    filter_from_train = True
    sample_by_score = True
    score_json = f'/mnt/dataset/patchcore/monuseg_results/IM224_SAM_P01_D1024-1024_PS-9_ST-4_AN-1_S0/evaluated_results/scores_v1.2.json'

    sem2inst = sem_to_inst_map
    syn_model = 'sdm'
    basedir=f'/mnt/dataset/stardist/{syn_model}_monuseg_v1.2_SynPlusGT_filt_prob'
    train_epochs = 500

    gt_dirs = {
        # "train" : ["/mnt/cvai_s3/CVAI/genai/Stardist_data/MoNuSegTrainingData/"]
        # "all": ["/mnt/dataset/MoNuSeg/patches_valid_256x256_128x128/MoNuSegTrainingData"],
        # "train": ["/mnt/dataset/MoNuSeg/patches_256x256_128x128/ResNet18_kmeans_10_v1.1/4/MoNuSegTrainingData"],
        # "train": ["/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/05ss/MoNuSegTrainingData/"],
        # "train": ["/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/MoNuSegTrainingData/"],
        "train" : ["/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/__ResNet50_umap_n_components_3_random_state_42_hdbscan_min_samples_10_min_cluster_size_50_v1.2/6/MoNuSegTrainingData/"],
        # "train": ["/mnt/cvai_s3/CVAI/genai/Stardist_data/05ss"],

        # "test": ["/mnt/dataset/MoNuSeg/patches_256x256_128x128/ResNet18_kmeans_10_v1.1/4/MoNuSegTestData"],
        # "val": ["/mnt/dataset/MoNuSeg/patches_valid_inst_256x256_128x128/MoNuSegTestData"],
        # "val": ["/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/MoNuSegTestData/"],
        "val" : ["/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/__ResNet50_umap_n_components_3_random_state_42_hdbscan_min_samples_10_min_cluster_size_50_v1.2/6/MoNuSegTestData/"],
    }

    gt_file_train, gt_file_val = get_train_val(gt_dirs)

    gt_file_train = subsample(gt_file_train, perc=.05)

    print(f"\ttrain_file subsampled-> {len(gt_file_train)}")

    # syn_pardir = "/mnt/dataset/MoNuSeg/out_sdm/monuseg_patches_128.64CH_200st_1e-4lr_8bs_hv_ResNet18_kmeans_10_v1.1_4/ResNet18_kmeans_10_v1.1/*/"
    # syn_pardir = "/mnt/cvai_s3/CVAI/genai/Stardist_data/05ss"
    # syn_pardir = "/mnt/cvai_s3/CVAI/genai/Stardist_data/patches_valid_inst_128x128_128x128/05ss"

    '''
    syn_dirs = sorted(glob.glob(os.path.join(syn_pardir, "*")))

    def get_syn_name(x):
        c, x = os.path.split(x)
        c = os.path.split(c)[-1]
        return f"{c}_{x}"

    syn_dirs = { get_syn_name(x): x for x in syn_dirs}
    '''

    # syn_dirs = {
    #     "syn" : syn_pardir
    # }

    # syn_dirs = {
    #     "style1" : "/mnt/dataset/MoNuSeg/dataset_col_aug/style1_128x128",
    #     "style2" : "/mnt/dataset/MoNuSeg/dataset_col_aug/style2_128x128",
    #     "style3" : "/mnt/dataset/MoNuSeg/dataset_col_aug/style3_128x128",
    # }

    # syn_dirs = {
    #     "s1.2_020000" : "/mnt/dataset/MoNuSeg/out_sdm_128x128/patches_valid_128.32CH_1000st_1e-4lr_8bs_hvb_col_cos_clus6/output_train_shuffle_model_s1.2_020000/",
    #     "s1.5_020000" : "/mnt/dataset/MoNuSeg/out_sdm_128x128/patches_valid_128.32CH_1000st_1e-4lr_8bs_hvb_col_cos_clus6/output_train_shuffle_model_s1.5_020000/",
    #     "s1.2_040000" : "/mnt/dataset/MoNuSeg/out_sdm_128x128/patches_valid_128.32CH_1000st_1e-4lr_8bs_hvb_col_cos_clus6/output_train_shuffle_model_s1.2_040000/",
    #     "s1.5_040000" : "/mnt/dataset/MoNuSeg/out_sdm_128x128/patches_valid_128.32CH_1000st_1e-4lr_8bs_hvb_col_cos_clus6/output_train_shuffle_model_s1.5_040000/",
    # }

    ## SDM files
    syn_pardir = "/mnt/dataset/MoNuSeg/out_sdm_128x128/patches_valid_128.32CH_1000st_1e-4lr_8bs_hvb_col_cos_clus6"
    syn_dirs = sorted(glob.glob(os.path.join(syn_pardir, "v1.2_*")))

    syn_dirs = {os.path.split(k)[-1] : k for k in syn_dirs}


    syn_file_list = []
    if syn_model == "col_aug":
        assert mult is not None, "Both filters and mult can't be None"
        ## subsample syn files
        syn_file_list, _ = get_file_label(syn_dirs, img_path='images', inst_path='inst_masks')
        if filter_from_train:
            syn_file_list_filtered = filt_from_train(syn_file_list, gt_file_train, filter_from_train)
        else:
            syn_file_list_filtered = syn_file_list

        syn_file_list_filtered = subsample(syn_file_list_filtered, size=len(gt_file_train) * int(mult))

    elif syn_model == "sdm" and filters is None:
        assert mult is not None, "Both filters and mult can't be None"
        ## subsample syn files
        syn_file_list, _ = get_file_label(syn_dirs, img_path='samples', inst_path='inst_masks')
        if filter_from_train:
            syn_file_list_filtered = filt_from_train(syn_file_list, gt_file_train, filter_from_train)
        else:
            syn_file_list_filtered = syn_file_list
        print(len(syn_file_list_filtered))

        if sample_by_score:
            prob_score = get_prob_from_json(syn_file_list_filtered, score_json)
        else:
            prob_score = None

        syn_file_list_filtered = subsample(syn_file_list_filtered, size=len(gt_file_train) * int(mult), prob=prob_score)

    else:
        for f in filters:
            s, _ = get_file_label(syn_dirs, img_path='images', inst_path='inst_masks', filt=f)
            # s, _ = get_file_label(syn_dirs, img_path='images_in_silico_inst', inst_path='masks_in_silico_inst', filt=f)
            syn_file_list.extend(s)
        syn_file_list_filtered = filt_from_train(syn_file_list, gt_file_train, filter_from_train)



    print(f"\tsyn_file -> {len(syn_file_list_filtered)}")


    ## sampling based on F1 vals
    if False:
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

        model = StarDist2D(None, name='stardist_128_128_gt_inst', basedir='/mnt/dataset/stardist/models_monuseg')

        X_syn = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), tqdm(syn_file_list_filtered)))
        Y_syn = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), tqdm(syn_file_list_filtered)))
        Y_pred_syn = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
            for x in tqdm(X_syn)]

        stats_syn = [matching(y_t, y_p, thresh=.7, criterion='iou', report_matches=False) for y_t,y_p in zip(Y_syn, Y_pred_syn)]
        metric_syn = [getattr(stat, 'f1') for stat in stats_syn]

        metric_syn = np.power(metric_syn, 2)

        ind = np.random.choice(range(len(syn_file_list_filtered)), size=int(len(syn_file_list_filtered)*.2), replace=False, p=metric_syn/np.sum(metric_syn))
        syn_file_list_filtered = [syn_file_list_filtered[i] for i in ind]

        print(f"syn_file (prob filt) -> {len(syn_file_list_filtered)}")

    train_file_list = gt_file_train + syn_file_list_filtered
    # train_file_list = gt_file_train
    val_file_list = gt_file_val

    n_channel = 3
    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    img_preprocess = lambda x : normalize(x,1,99.8,axis=axis_norm)
    label_preprocess = lambda x : fill_label_holes(x)

    # if use_inst_mask:
    #     label_preprocess = lambda x : fill_label_holes(x)
    # else:
    #     label_preprocess = lambda x : fill_label_holes(sem2inst(x))

    '''
    train_set = MoNuSegDataset(
        train_file_list, file_type=".png", mode="train", with_type=False, preprocess=(img_preprocess, label_preprocess),
        target_gen=(None, None), input_shape=(256,256), mask_shape=(256,256))

    val_set = MoNuSegDataset(
        val_file_list, file_type=".png", mode="test", with_type=False, preprocess=(img_preprocess, label_preprocess),
        target_gen=(None, None), input_shape=(256,256), mask_shape=(256,256))

    train_loader = DataLoader(train_set, num_workers= 8, batch_size= 8, shuffle=True, drop_last=False, )
    val_loader = DataLoader(val_set, num_workers= 8, batch_size= 8, shuffle=False, drop_last=False, )

    '''
    
    '''
    X_name = sorted(glob('/mnt/dataset/MoNuSeg/patches_valid_256x256_128x128/MoNuSegTrainingData/images/*.png'))
    Y_name = sorted(glob('/mnt/dataset/MoNuSeg/patches_valid_256x256_128x128/MoNuSegTrainingData/bin_masks/*.png'))
    print(len(X_name), len(Y_name))
    assert all(Path(x).name==Path(y).name for x,y in zip(X_name,Y_name))


    X = list(map(lambda x: read_img(x, 'RGB'), X_name))
    Y = list(map(lambda x: read_img(x, 'L'), Y_name))
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(sem_to_inst_map(y)) for y in tqdm(Y)]

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))
    '''

    x_id = 0
    y_id = 2 if use_inst_mask else 1
    mask_dtype = 'I' if use_inst_mask else 'L'

    X_trn = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), tqdm(train_file_list)))
    Y_trn = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), tqdm(train_file_list)))

    X_syn = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), tqdm(syn_file_list_filtered)))
    Y_syn = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), tqdm(syn_file_list_filtered)))

    X_val = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), tqdm(val_file_list)))
    Y_val = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), tqdm(val_file_list)))

    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    # 32 is a good default choice (see 1_data.ipynb)
    n_rays = 32

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = (2,2)

    use_gpu = True

    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        use_gpu      = use_gpu,
        n_channel_in = n_channel,
        train_patch_size = Y_trn[0].shape
    )
    print(conf)
    vars(conf)

    model = StarDist2D(conf, name=out_name, basedir=basedir)

    median_size = calculate_extents(list(Y_trn), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    '''
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)        
    '''

    # if len(X_syn) > 0:

    #     print("Training with synthetic...")
    #     model.train(X_syn, Y_syn, validation_data=(X_val,Y_val), augmenter=augmenter, epochs=train_epochs)

    #     model.prepare_for_training()

    # else:
    #     print("Skipping synthetic...")

    print("Training with real...")
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter, epochs=train_epochs)

    # model.optimize_thresholds(X_val, Y_val)

    # test_dirs = {
    #     "test": ["/mnt/dataset/MoNuSeg/patches_valid_inst_256x256_128x128/MoNuSegTestData"],
    # }

    # test_file_list, _ = get_file_label(test_dirs, gt=True)

    # X_test = list(map(lambda x: img_preprocess(read_img(x[x_id], 'RGB')), test_file_list))
    # Y_test = list(map(lambda x: label_preprocess(read_img(x[y_id], mask_dtype)), test_file_list))

    # Y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
    #           for x in tqdm(X_test)]

    # taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # stats = [matching_dataset(Y_test, Y_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

    # print(stats[taus.index(0.5)])


if __name__ == "__main__":

    out_names = []
    filter_arr = []
    sizes = []

    # out_names += ['stardist_128_128_05gt_inst'] 
    # filter_arr += [[f'*_{i}' for i in range(0)]] 
    # sizes += [0]

    out_names += ['stardist_128_128_05gt_05syn_inst', 'stardist_128_128_05gt_05syn.x2_inst', 'stardist_128_128_05gt_05syn.x3_inst', 'stardist_128_128_05gt_05syn.x4_inst', 'stardist_128_128_05gt_05syn.x5_inst']
    # filter_arr += [[f'*_{i}' for i in range(1)], [f'*_{i}' for i in range(2)], [f'*_{i}' for i in range(3)], [f'*_{i}' for i in range(4)], [f'*_{i}' for i in range(5)]]
    filter_arr += [None] * 5
    sizes += list(range(1, 6))

    # out_names = ['stardist_128_128_gt_inst']
    # filter_arr = [[f'*_{i}' for i in range(0)]]

    # out_names = ['stardist_128_128_FT.2']
    # filter_arr = [[f'*_{i}' for i in range(5)]]

    # out_names, filter_arr = ['tmp'], [[]]

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = gputools_available()
    print(use_gpu)

    # Maximum memory to allocated in Mb
    allocated_mem = min(8000, get_total_mem())
    print("Allocated memory:", allocated_mem)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8, total_memory=allocated_mem, allow_growth=False)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    for ind, o in enumerate(out_names):
        print(ind)
        # if ind not in [3] : continue
        # if ind in [0, 1, 2] : continue
        main(o, filters=None, mult=sizes[ind])

