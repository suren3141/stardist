import numpy as np
import os, glob
from utils import get_file_label, link
from pathlib import Path
import shutil


def get_GT_syn_files(gt_dir=None, syn_pardir=None, inst_path='masks_in_silico_inst'):
    use_inst_mask = True
    filters = [f'*_{i}' for i in range(5)]

    gt_dirs = {
        "train": [gt_dir],
    }

    gt_file_list, _ = get_file_label(gt_dirs, gt=True)

    print(f"gt_file -> {len(gt_file_list)}")

    # syn_pardir = "/mnt/dataset/MoNuSeg/out_sdm/monuseg_patches_128.64CH_200st_1e-4lr_8bs_hv_ResNet18_kmeans_10_v1.1_4/ResNet18_kmeans_10_v1.1/*/"
    syn_dirs = sorted(glob.glob(os.path.join(syn_pardir, "*")))

    def get_syn_name(x):
        c, x = os.path.split(x)
        c = os.path.split(c)[-1]
        return f"{c}_{x}"

    syn_dirs = { get_syn_name(x): x for x in syn_dirs}

    syn_dirs = {
        "syn" : syn_pardir
    }

    syn_file_list = []
    for f in filters:
        s, _ = get_file_label(syn_dirs, img_path='images_in_silico_inst', inst_path=inst_path, filt=f)
        syn_file_list.extend(s)

    syn_file_list_filtered = []

    for f in syn_file_list:
        valid = any([Path(f[0]).stem.startswith(Path(i[0]).stem) for i in gt_file_list])
        if valid : syn_file_list_filtered.append(f)

    print(f"syn_file -> {len(syn_file_list_filtered)}")

    return gt_file_list, syn_file_list_filtered

if __name__ == "__main__":
    gt_pardir = "/mnt/cvai_s3/CVAI/genai/Stardist_data/05ss"
    syn_pardir = "/mnt/cvai_s3/CVAI/genai/Stardist_data/25ss"

    gt_file_list, syn_file_list_filtered = get_GT_syn_files(gt_dir=gt_pardir, syn_pardir=syn_pardir)

    out_path = "/mnt/cvai_s3/CVAI/genai/Stardist_data/05ss"
    img_path = os.path.join(out_path, "images_in_silico_inst")
    ann_path = os.path.join(out_path, "masks_in_silico_inst")

    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(ann_path): os.makedirs(ann_path)

    for f in syn_file_list_filtered:
        x, _, y = f
        x_name = os.path.split(x)[-1]
        y_name = os.path.split(y)[-1]

        x_dst = os.path.join(img_path, x_name)
        y_dst = os.path.join(ann_path, y_name)

        shutil.copyfile(x, x_dst)
        shutil.copyfile(y, y_dst)
        
        print(x, x_dst)
