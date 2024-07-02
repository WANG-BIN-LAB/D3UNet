
from __future__ import absolute_import, print_function

import pandas as pd
import GeodisTK
import numpy as np
from scipy import ndimage
import os
import nibabel as nib


# pixel accuracy
def binary_pa(s, g):
    """
        calculate the pixel accuracy of two N-d volumes.
        s: the segmentation volume of numpy array
        g: the ground truth volume of numpy array
        """
    pa = ((s == g).sum()) / g.size
    return pa


# Dice evaluation
def binary_dice(s, g):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    assert (len(s.shape) == len(g.shape))
    prod = np.multiply(s, g)
    s0 = prod.sum()
    dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)

    return dice


# RVD evaluation
def binary_rvd(s, g):
    """
    calculate the RVD score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    assert (len(s.shape) == len(g.shape))
    prod = s - g
    prod[prod < 0] = 1
    s0 = prod.sum()
    rvd = (s0 + 1e-10) / (g.sum() + 1e-10)

    return rvd


# VOE evaluation
def binary_voe(s, g):
    """
    calculate the voe score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    assert (len(s.shape) == len(g.shape))
    prod = s - g
    prod[prod < 0] = 1
    s0 = prod.sum()
    voe = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)

    return voe


# IOU evaluation
# Jaccard evaluation
def binary_iou(s, g):
    assert (len(s.shape) == len(g.shape))

    intersecion = np.multiply(s, g)

    union = np.asarray(s + g > 0, np.float32)
    iou = intersecion.sum() / (union.sum() + 1e-10)
    return iou


# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)


# 平均表面距离
def binary_assd(s, g, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


# relative volume error evaluation
def binary_relative_volume_error(s_volume, g_volume):
    s_v = float(s_volume.sum())
    g_v = float(g_volume.sum())
    assert (g_v > 0)
    rve = abs(s_v - g_v) / g_v
    return rve


def compute_class_sens_spec(pred, label):
    """
    Compute sensitivity and specificity for a particular example
    for a given class for binary.
    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (height, width, depth).
        label (np.array): binary array of labels, shape is
                          (height, width, depth).
    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """
    tp = np.sum((pred == 1) & (label == 1))
    tn = np.sum((pred == 0) & (label == 0))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


def get_evaluation_score(s_volume, g_volume, spacing, metric):
    if (len(s_volume.shape) == 4):
        assert (s_volume.shape[0] == 1 and g_volume.shape[0] == 1)
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    if (s_volume.shape[0] == 1):
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    metric_lower = metric.lower()

    if (metric_lower == "dice"):
        score = binary_dice(s_volume, g_volume)

    elif (metric_lower == "iou"):
        score = binary_iou(s_volume, g_volume)

    elif (metric_lower == "rvd"):
        score = binary_rvd(s_volume, g_volume)

    elif (metric_lower == "voe"):
        score = binary_voe(s_volume, g_volume)

    elif (metric_lower == 'assd'):
        score = binary_assd(s_volume, g_volume, spacing)

    elif (metric_lower == "hausdorff95"):
        score = binary_hausdorff95(s_volume, g_volume, spacing)

    elif (metric_lower == "rve"):
        score = binary_relative_volume_error(s_volume, g_volume)

    elif (metric_lower == "volume"):
        voxel_size = 1.0
        for dim in range(len(spacing)):
            voxel_size = voxel_size * spacing[dim]
        score = g_volume.sum() * voxel_size
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))

    return score


if __name__ == '__main__':

    seg_path = 'seg'
    gd_path = 'gt_data'
    save_dir = 'excel_out'
    seg = sorted(os.listdir(seg_path))

    dices = []
    ious = []
    rvds = []
    voes = []
    assds = []
    volumes = []
    hds = []
    rves = []
    case_name = []
    senss = []
    specs = []
    for name in seg:
        if not name.startswith('.') and name.endswith('nii.gz'):

            seg_ = nib.load(os.path.join(seg_path, name))
            seg_arr = seg_.get_fdata().astype('float32')
            gt_name = "gt_2019082606.nii.gz"
            gd_ = nib.load(os.path.join(gd_path,gt_name ))
            gd_arr = gd_.get_fdata().astype('float32')
            case_name.append(name)

           
            seg_class = seg_.get_data()
            seg_class_two = np.where(seg_class > 1, seg_class, 0)
            seg_class_one = seg_class - seg_class_two
            seg_class_two = seg_class_two / 2
            gd_class = gd_.get_fdata()
            gd_class_two = np.where(gd_class > 1, gd_class, 0)
            gd_class_one = gd_class - gd_class_two
            gd_class_two = gd_class_two / 2

            seg_arr_class_two = np.where(seg_arr > 1, seg_arr, 0)
            seg_arr_class_one = seg_arr - seg_arr_class_two
            seg_arr_class_two = seg_arr_class_two / 2

            gd_arr_class_two = np.where(gd_arr > 1, gd_arr, 0)
            gd_arr_class_one = gd_arr - gd_arr_class_two
            gd_arr_class_two = gd_arr_class_two / 2

            seg_input = seg_class_two
            gd_input = gd_class_two

            seg_arr_input = seg_arr_class_two
            gd_arr_input = gd_arr_class_two

            # 求VOE
            voe = get_evaluation_score(seg_input, gd_input, spacing=None, metric='voe')
            voes.append(voe)

            # 求RVD
            rvd = get_evaluation_score(seg_input, gd_input, spacing=None, metric='rvd')
            rvds.append(rvd)

            # 求hausdorff95距离
            hd_score = get_evaluation_score(seg_arr_input, gd_arr_input, spacing=None, metric='hausdorff95')
            hds.append(hd_score)

            # 求体积相关误差
            rve = get_evaluation_score(seg_arr_input, gd_arr_input, spacing=None, metric='rve')
            rves.append(rve)

            # 求dice
            dice = get_evaluation_score(seg_input, gd_input, spacing=None, metric='dice')
            dices.append(dice)

            # 求iou
            iou = get_evaluation_score(seg_input, gd_input, spacing=None, metric='iou')
            ious.append(iou)

            # 求assd
            assd = get_evaluation_score(seg_arr_input, gd_arr_input, spacing=None, metric='assd')
            assds.append(assd)

            
            sens, spec = compute_class_sens_spec(seg_input, gd_input)
            senss.append(sens)
            specs.append(spec)
    # 存入pandas

    data = {'dice': dices, 'VOE': voes,'RVD': rvds, 'IOU': ious, 'ASSD': assds, 'RVE': rves, 'Sens': senss, 'Spec': specs, 'HD95': hds}
    df = pd.DataFrame(data=data, columns=['dice', 'VOE', 'RVD', 'IOU', 'ASSD', 'RVE', 'Sens', 'Spec', 'HD95'], index=case_name)
    df.to_csv(os.path.join(save_dir, '24_944_873_root_metrics.csv'))
