from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import data as data_utils
import utils
import torch
import numpy as np


class UnpairedSegmData(Dataset):
    # unpaired segmentation data for semi-supervised learning
    def __init__(self):
        super().__init__()

        # segmentatoin db
        self.txn_segm = utils.fetch_lmdb_reader(
            'data/InterHand2.6M/segm_train_val_mp.lmdb')

        # unpaired segmentation ids
        self.train_segm_ids = torch.load(
            'data/InterHand2.6M/unpaired_segm_ids.pt')

        print('Total number of unpaired segm: ' + str(len(self.train_segm_ids)))

    def __getitem__(self, idx):
        segm_key = str(self.train_segm_ids[idx])

        # fetch segmenation
        img_segm_l = torch.LongTensor(
                data_utils.load_segm(segm_key + '__l', self.txn_segm))
        img_segm_r = torch.LongTensor(
                data_utils.load_segm(segm_key + '__r', self.txn_segm))
        img_segm = torch.stack((img_segm_l, img_segm_l, img_segm_r), dim=0)

        # get bbox around segm
        _, yidx, xidx = img_segm.nonzero(as_tuple=True)

        # topleft and bottom right corners
        x1 = xidx.min()
        y1 = yidx.min()
        x2 = xidx.max()
        y2 = yidx.max()

        # center and max dim
        cx = (x1 + x2)//2
        cy = (y1 + y2)//2
        dim = max(x2 - x1, y2 - y1)

        # new topleft and bottom right corners for squared crops
        x3 = cx - dim//2
        y3 = cy - dim//2

        # crop with augmentation
        img_segm_crop = data_utils.rand_crop_segm(
            img_segm.permute(1, 2, 0).numpy(),
            [x3, y3, dim, dim], (128, 128))[:, :, 1:]
        return torch.LongTensor(img_segm_crop)

    def __len__(self):
        return len(self.train_segm_ids)


class UnpairedImageData(Dataset):
    # dataset for RGB hand images without segmentation annotation
    def __init__(self, transform, args):
        super().__init__()
        self.args = args  # see config.py
        self.transform = transform

        # LMDB handle for raw images in the entire InterHand2.6M dataset
        self.txn = utils.fetch_lmdb_reader('data/InterHand2.6M/interhand.lmdb')

        datalist_all = torch.load(
            'data/InterHand2.6M/datalist_train_complete.pt')
        datalist_mp = torch.load(
            'data/InterHand2.6M/datalist_train_mp.pt')

        # get datalist without annotation
        im_path_mp = set([data['img_path'] for data in datalist_mp])
        datalist_images = []
        for data in datalist_all:
            if data['img_path'] not in im_path_mp:
                datalist_images.append(data)
        self.datalist = datalist_images

        if args.semisplit == 'mini':
            self.datalist = data_utils.downsample(self.datalist, 'minitrain')
        print('Total number of images: ' + str(len(self.datalist)))

    def __getitem__(self, idx):
        args = self.args
        data = self.datalist[idx]

        # unpack meta data
        img_path = data['img_path']
        bbox = data['bbox']
        hand_type = data['hand_type']
        hand_type = self.handtype_str2array(hand_type)

        # image reading
        img = np.array(utils.read_lmdb_image(self.txn, img_path.replace('./', '')))
        hw_size = img.shape[:2]

        # augmentation
        img, _, hand_type, do_flip = data_utils.augmentation(
            img, None, bbox, hand_type, 'train', args.input_img_shape)

        # image normalization
        img = self.transform(img.astype(np.float32))/255.

        inputs = {'img': img}
        meta_info = {
            'capture': int(data['capture']), 'cam': int(data['cam']),
            'frame': int(data['frame']),
            'idx': idx, 'im_path': img_path,
            'hw_size': hw_size, 'flipped': do_flip}
        return inputs, meta_info

    def __len__(self):
        return len(self.datalist)

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)


class ImageDataset(Dataset):
    # You will use this dataset class for training and validataion
    # For testing and submission to the server, checkout ImageDatasetTest
    # This dataset class contains image-segmentation pairs
    # You probably don't need to change this class.
    # If you do, please make sure it doesn't break the code for your submission.

    def __init__(self, transform, mode, split, args):
        super().__init__()
        self.args = args  # see config.py
        self.mode = mode  # train, test, val
        self.transform = transform

        # LMDB handle for raw images in the entire InterHand2.6M dataset
        self.txn = utils.fetch_lmdb_reader('data/InterHand2.6M/interhand.lmdb')

        if mode in ['train', 'val']:
            # paired segmentation
            txn_segm = utils.fetch_lmdb_reader('data/InterHand2.6M/segm_train_val_mp.lmdb')
            self.txn_segm_l = txn_segm
            self.txn_segm_r = txn_segm
            self.path2id = torch.load('data/InterHand2.6M/path2id_%s.pt' % (mode))

            # meta data such as image path, bounding boxes for hands
            self.datalist = torch.load(
                    'data/InterHand2.6M/datalist_%s_mp.pt' % (mode))
            # downsample the dataset for fast development
            # it activates with --trainsplit minitrain
            # the default is --trainsplit train
            # same for valsplit
            self.datalist = data_utils.downsample(self.datalist, split)
        elif mode == 'test':
            # Please do not change this.
            # As our submission server expects the exact set of images for evaluation
            datalist = torch.load('data/InterHand2.6M/datalist_stud_test.pt')
            self.datalist = datalist
        else:
            assert False
        print('Total number of annotations: ' + str(len(self.datalist)))


    def __getitem__(self, idx):
        args = self.args
        data = self.datalist[idx]

        # unpack meta data
        img_path = data['img_path']
        bbox = data['bbox']
        hand_type = data['hand_type']
        hand_type = self.handtype_str2array(hand_type)

        # image reading
        img = np.array(utils.read_lmdb_image(self.txn, img_path.replace('./', '')))
        hw_size = img.shape[:2]

        # segm reading
        segm_key = img_path.replace('./data/InterHand2.6M/images/', '').replace('.jpg', '')
        segm_key = str(self.path2id[segm_key])
        img_segm_l = data_utils.load_segm(segm_key + '__l', self.txn_segm_l)
        img_segm_r = data_utils.load_segm(segm_key + '__r', self.txn_segm_r)
        img_segm = np.stack((img_segm_l, img_segm_l, img_segm_r), axis=2)

        # augmentation
        img, img_segm, hand_type, do_flip = data_utils.augmentation(
            img, img_segm, bbox, hand_type, self.mode, args.input_img_shape)
        img_segm = torch.FloatTensor(img_segm).permute(2, 0, 1)
        if do_flip:
            img_segm = data_utils.swap_lr_labels_segm_target_channels(img_segm)

        # the target segmentation dimension is 128x128
        # please do not change this and the interpolation method
        img_segm_128 = img_segm.clone().long()
        img_segm_128 = F.interpolate(
                img_segm_128[None, :, :, :].float(), 128, mode='nearest'
                ).long().squeeze().clone()
        fg_idx = img_segm_128[1].nonzero(as_tuple=True)
        img_segm_128[1][fg_idx[0], fg_idx[1]] -= 16
        img_segm_128 = img_segm_128[1:]

        # image normalization
        img = self.transform(img.astype(np.float32))/255.

        inputs = {'img': img}
        targets = {
            'segm_128': img_segm_128}
        meta_info = {
            'capture': int(data['capture']), 'cam': int(data['cam']),
            'frame': int(data['frame']),
            'idx': idx, 'im_path': img_path,
            'hw_size': hw_size, 'flipped': do_flip}
        return inputs, targets, meta_info

    def __len__(self):
        return len(self.datalist)

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)


class ImageDatasetTest(ImageDataset):
    # Image dataset class for the test set.
    # It only contains images and meta data and it does not contain segm. annotation
    # You probably don't need to change this class.
    # If you do, please make sure it doesn't break the code for your submission.
    def __init__(self, transform, mode, split, args):
        super().__init__(transform, mode, split, args)

    def __getitem__(self, idx):
        args = self.args
        data = self.datalist[idx]
        img_path = data['img_path']
        bbox = data['bbox']
        hand_type = data['hand_type']
        hand_type = self.handtype_str2array(hand_type)

        img = np.array(utils.read_lmdb_image(self.txn, img_path.replace('./', '')))
        hw_size = img.shape[:2]

        # augmentation
        img, _, hand_type, do_flip = data_utils.augmentation(
            img, None, bbox, hand_type, self.mode, args.input_img_shape)

        img = self.transform(img.astype(np.float32))/255.

        inputs = {'img': img}
        meta_info = {
            'capture': int(data['capture']), 'cam': int(data['cam']),
            'frame': int(data['frame']),
            'idx': idx, 'im_path': img_path,
            'hw_size': hw_size, 'flipped': do_flip
            }
        return inputs, meta_info


def fetch_dataloader(mode, split, args):
    # get data loader based on config
    b_size = args.batch_size
    if mode == 'train':
        DatasetClass = ImageDataset
        shuffle = True
        dmode = 'train'
        drop_last = True
    elif mode == 'val':
        DatasetClass = ImageDataset
        shuffle = False
        dmode = mode
        drop_last = True
    elif mode == 'test':
        DatasetClass = ImageDatasetTest
        shuffle = False
        dmode = mode
        drop_last = False
    elif mode == 'images':
        DatasetClass = UnpairedImageData
        shuffle = True
        drop_last = True
        dataset = DatasetClass(transforms.ToTensor(), args)
        b_size *= 2
    elif mode == 'segm':
        DatasetClass = UnpairedSegmData
        shuffle = True
        drop_last = True
        dataset = DatasetClass()
    else:
        assert False, 'Invalid mode: %s' % (mode)

    print("Creating %s dataset..." % (mode))
    if mode in ['train', 'val', 'test']:
        dataset = DatasetClass(
            transforms.ToTensor(), dmode, split, args)

    loader = DataLoader(
        dataset=dataset,
        batch_size=b_size,
        shuffle=shuffle, num_workers=args.num_workers, pin_memory=True,
        drop_last=drop_last)
    return loader
