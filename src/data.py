import random
import copy
import cv2
import torch
import torch.utils.data
import numpy as np
import os.path as op
import utils

# utils for data preprocessing


def swap_lr_labels_segm_target_channels(segm_target):
    """
    Flip left and right label (not the width) of a single segmentation image.
    """
    assert isinstance(segm_target, torch.Tensor)
    assert len(segm_target.shape) == 3
    assert segm_target.min() >= 0
    assert segm_target.max() <= 32
    img_segm = segm_target.clone()
    right_idx = ((1 <= img_segm)*(img_segm <= 16)).nonzero(as_tuple=True)
    left_idx = ((17 <= img_segm)*(img_segm <= 32)).nonzero(as_tuple=True)
    img_segm[right_idx[0], right_idx[1], right_idx[2]] += 16
    img_segm[left_idx[0], left_idx[1], left_idx[2]] -= 16
    img_segm_swapped = img_segm.clone()
    img_segm_swapped[1], img_segm_swapped[2] = img_segm_swapped[2].clone(), img_segm_swapped[1].clone()
    return img_segm_swapped


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def rand_crop_segm(img_segm, bbox, input_img_shape):
    # randomly crop segmentation masks based on translation, scale and rotation augmentation
    trans, scale, rot, _, _ = get_aug_config()

    # crop image based on bbox
    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[3] * trans[1]

    box_segm = copy.deepcopy(bbox)
    img_segm, _, _ = generate_patch_image(
            img_segm, box_segm, False, scale, rot,
            input_img_shape, cv2.INTER_NEAREST)
    img_segm = img_segm.astype(np.uint8)
    return img_segm


def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape, interpl_strategy):
    # crop images based on augmentation
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=interpl_strategy)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans


def load_segm(path, txn):
    # load segmentation given LMDB handle.
    img = utils.read_lmdb_image(txn, path)
    if img is None:
        return img

    # all channels should be identical
    sum_image = img.reshape(-1, 3).sum(axis=0)
    assert sum_image[0] == sum_image[1]
    assert sum_image[1] == sum_image[2]
    return img[:, :, 0]


def get_aug_config():
    # get augmentation objects from config
    trans_factor = 0.01
    scale_factor = 0.01
    rot_factor = 5
    color_factor = 0.01

    # affine transformation that will be applied on the input image
    trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    return trans, scale, rot, do_flip, color_scale


def augmentation(
        img, img_segm, bbox, hand_type, mode,
        input_img_shape):
    # data augmentation

    img = img.copy()
    hand_type = hand_type.copy()

    if mode == 'train':
        trans, scale, rot, do_flip, color_scale = get_aug_config()
    else:
        trans, scale, rot, do_flip, color_scale = [0, 0], 1.0, 0.0, False, np.array([1, 1, 1])

    # crop image based on bbox
    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[3] * trans[1]
    img, trans, inv_trans = generate_patch_image(
            img, bbox, do_flip, scale, rot,
            input_img_shape, cv2.INTER_LINEAR)

    # crop segm target same way as the image
    if img_segm is not None:
        box_segm = copy.deepcopy(bbox)
        img_segm, _, _ = generate_patch_image(
                img_segm, box_segm, do_flip, scale, rot,
                input_img_shape, cv2.INTER_NEAREST)
        img_segm = img_segm.astype(np.uint8)

    img = np.clip(img * color_scale[None, None, :], 0, 255)

    if do_flip:
        hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
    return img, img_segm, hand_type, do_flip


def process_bbox(bbox, original_img_shape, input_img_shape):
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = input_img_shape[1]/input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox


def downsample(raw_data, split):
    # downsample datalist based on split
    # if split is minitrain or minival, return only 1000 samples for quick sanity check
    # Note: Only use this for a quick sanity check to make sure the code does not break.
    # Do not use this for your submission.

    assert isinstance(raw_data, list)

    if 'mini' not in split:
        return raw_data
    import random
    random.seed(1)
    assert random.randint(0, 100) == 17, \
        "Same seed but different results; Subsampling might be different."
    if split == 'minival':
        num_samples = 1000
    elif split == 'minitrain':
        num_samples = 1000
    else:
        assert False, "Unknown split {}".format(split)
    return random.sample(raw_data, num_samples)


def process_anno(
        img, bbox_rootnet,
        input_img_shape, img_path, mode):
    # packaging up the annotation

    ann = img['anno']
    capture_id = img['capture']
    seq_name = img['seq_name']
    cam = img['camera']
    frame_idx = img['frame_idx']
    img_path = op.join(img_path, mode, img['file_name'])

    hand_type = ann['hand_type']
    hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

    # the absolute root depth of the left/right hands
    # are either predicted by root net or we use the gt.
    img_width, img_height = img['width'], img['height']

    # 2d bbox, not 3d
    bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
    bbox = process_bbox(bbox, (img_height, img_width), input_img_shape)
    data = {
        'img_path': img_path, 'seq_name': seq_name,
        'hand_type': hand_type, 'hand_type_valid': hand_type_valid,
        'file_name': img['file_name'], 'bbox': bbox,
        'capture': capture_id, 'cam': cam, 'frame': frame_idx}
    return data
