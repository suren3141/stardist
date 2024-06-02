import numpy as np
from scipy.optimize import linear_sum_assignment

#####
def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_iou_matrix(gt_mask, pred_mask):
    gt_mask = remap_label(gt_mask)
    pred_mask = remap_label(pred_mask)

    # Includes background, but IOU is 0
    num_gt = len(np.unique(gt_mask))
    num_pred = len(np.unique(pred_mask))
    iou_matrix = np.zeros((num_pred, num_gt))

    assert num_gt > 2 and num_pred > 2, "Only 0/1 found in label. Make sure it's not binary or remove the assertion"

    for i in range(1, num_gt):
        gt_bin = gt_mask == i
        for j in range(1, num_pred):
            pred_bin = pred_mask == j
            iou_matrix[j, i] = calculate_iou(gt_bin, pred_bin)

    assert (iou_matrix > 0).sum(), "All IOU vals < 0"
    return iou_matrix

def hungarian_matching(iou_matrix):

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_pair = [(r, c) for (r, c) in zip(row_ind, col_ind) if iou_matrix[r, c] > 0]
    return matched_pair

def evaluate_instance_segmentation_single(pred_mask, gt_mask, iou_threshold=0.5):
    iou_matrix = compute_iou_matrix(pred_mask, gt_mask)
    matched_pairs = hungarian_matching(iou_matrix)
    assert len(matched_pairs) > 0, "No matched pairs"

    # ignore zeros
    num_gt = len(np.unique(gt_mask)) -1
    num_pred = len(np.unique(pred_mask)) -1

    tp = len([pair for pair in matched_pairs if iou_matrix[pair[0], pair[1]] >= iou_threshold])
    fp = num_pred - tp
    fn = num_gt - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall)
    
    return {
        'f1' : f1,
        'precision': precision,
        'recall': recall,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }

def evaluate_instance_f1(pred_masks, gt_masks, iou_threshold=0.5):
    f1_vals = []
    for pred, gt in zip(pred_masks, gt_masks):
        instance_segmentation_metric = evaluate_instance_segmentation_single(pred, gt, iou_threshold=iou_threshold)
        f1_vals.append(instance_segmentation_metric['f1'])

    return np.mean(f1_vals)

