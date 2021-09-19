
from tqdm import tqdm
import time
import torch
import numpy as np
import lmdb
import os.path as op
import cv2 as cv
from PIL import Image
import os
import tarfile
import hashlib

CUDA_THINGS = (torch.Tensor, torch.nn.utils.rnn.PackedSequence)


# define your own log_metric and log_image for your favourite logger
class Experiment():
    def __init__(self, args):
        self.args = args
        hash = hashlib.sha1()
        hash.update(str(time.time()).encode('utf-8'))
        key = hash.hexdigest()
        self.key = key[:9]

    def get_key(self):
        return self.key

    def log_metric(self, key, val, step):
        # e.g., self.experiment.log_metric(key, val, step=step)
        print(key, val, step)

    def log_image(self, im, fname, step):
        # e.g., self.experiment.log_image(im, name=fname, step=step)
        print(im, fname, step)


def push_images(experiment, all_im_list, step, no_tqdm=True, verbose=True):
    if experiment is None:
        return
    # push images to experiment
    if verbose:
        print("Pushing PIL images")
        tic = time.time()
    iterator = all_im_list if no_tqdm else tqdm(all_im_list)
    for im in iterator:
        if 'fig_name' in im.keys():
            experiment.log_image(im['im'], im['fig_name'], step)
        else:
            experiment.log_image(im['im'], 'unnamed', step)
    if verbose:
        toc = time.time()
        print("Done pushing PIL images (%.1fs)" % (toc - tic))


def log_dict(experiment, metric_dict, step, postfix=None):
    # push metrics to experiment
    if experiment is None:
        return
    for key, value in metric_dict.items():
        if postfix is not None:
            key = key + postfix
        if isinstance(value, torch.Tensor) and len(value.view(-1)) == 1:
            value = value.item()

        if isinstance(value, (int, float, np.float32)):
            experiment.log_metric(key, value, step=step)


# push things to a device
def things2dev(obj, dev):
    if isinstance(obj, CUDA_THINGS):
        return obj.to(dev)
    if isinstance(obj, list):
        return [things2dev(x, dev) for x in obj]
    if isinstance(obj, tuple):
        return tuple(things2dev(list(obj), dev))
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = things2dev(v, dev)
    return obj


def tensor2np(ten):
    """This function move tensor to cpu and convert to numpy"""
    return ten.cpu().detach().numpy()


def ld2dl(LD):
    assert isinstance(LD, list)
    assert isinstance(LD[0], dict)
    """
    A list of dict (same keys) to a dict of lists
    """
    dict_list = {k: [dic[k] for dic in LD] for k in LD[0]}
    return dict_list


def fetch_lmdb_reader(db_path):
    env = lmdb.open(db_path, subdir=op.isdir(db_path),
                    readonly=True, lock=False,
                    readahead=False, meminit=False)
    txn = env.begin(write=False)
    return txn


# get an image with key `fname` from a LMDB handler `txn`
def read_lmdb_image(txn, fname):
    image_bin = txn.get(fname.encode('ascii'))
    if image_bin is None:
        return image_bin
    image = np.fromstring(image_bin, dtype=np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    return image


def segm_iou(pred, target, n_classes, tol, background_cls=0):
    """
    Compute mean iou for a segmentation mask.
    pred: (dim, dim)
    target: (dim, dim)
    n_classes: including the background class
    tol: how many entries to ignore for union for noisy target map
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert pred.shape == target.shape
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(n_classes):
        if cls == background_cls:
            continue
        pred_cls = pred == cls
        target_cls = target == cls
        target_inds = target_cls.nonzero(as_tuple=True)[0]
        intersection = pred_cls[target_inds].long().sum()
        union = pred_cls.long().sum() + target_cls.long().sum() - intersection

        # number of pixels in the target for this class
        num_target_pixel = target_cls.sum().item()
        if num_target_pixel < tol:
            # ignore classes that do not show or just a few pixels in GT segm
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    # average over all classes
    return float(np.nanmean(np.array(ious)))


# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D
    numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image
    in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def make_tarfile(output_filename, source_dir):
    # package folder into a tar.gz file
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=op.basename(source_dir))


def mkdir_p(exp_path):
    os.makedirs(exp_path, exist_ok=True)


# DO NOT MODIFY THIS
def package_lmdb(lmdb_name, segm_l, segm_r, keys, write_frequency=5000):
    """
    Package segm files into a lmdb database.
    lmdb_name is the name of the lmdb database file
    segm_l, segm_r are segmentation predictions.
    map_size: recommended to set to len(fnames)*num_types_per_image*10
    keys: the key of each image in dict
    """
    map_size = len(keys) * 5130240 * 2
    db = lmdb.open(lmdb_name, map_size=map_size)
    txn = db.begin(write=True)

    for idx in tqdm(range(len(keys))):
        key = keys[idx] + '__l'
        curr_segm_l = segm_l[idx]
        curr_segm_l = curr_segm_l.numpy().astype(np.uint8)
        img = cv.cvtColor(curr_segm_l, cv.COLOR_BGR2RGB)
        status, encoded_image = cv.imencode(
            ".png", img, [cv.IMWRITE_JPEG_QUALITY, 100]
        )
        assert status
        txn.put(key.encode('ascii'), encoded_image.tostring())

        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    for idx in tqdm(range(len(keys))):
        key = keys[idx] + '__r'
        curr_segm_r = segm_r[idx]
        curr_segm_r = curr_segm_r.numpy().astype(np.uint8)

        img = cv.cvtColor(curr_segm_r, cv.COLOR_BGR2RGB)
        status, encoded_image = cv.imencode(
            ".png", img, [cv.IMWRITE_JPEG_QUALITY, 100]
        )
        assert status
        txn.put(key.encode('ascii'), encoded_image.tostring())

        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    db.sync()
    db.close()
