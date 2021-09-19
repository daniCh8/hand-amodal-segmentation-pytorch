from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import data as data_utils
import utils
import torch
import numpy as np

class ImageDataset(Dataset):
    # You will use this dataset class for training and validataion
    # For testing and submission to the server, checkout ImageDatasetTest
    # This dataset class contains image-segmentation pairs
    # You probably don't need to change this class.
    # If you do, please make sure it doesn't break the code for your submission.

    def __init__(self, transform, mode, split): #, args):
        super().__init__()
        # self.args = args  # see config.py
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
        # args = self.args
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
        input_img_shape = tuple([128]*2)
        img, img_segm, hand_type, do_flip = data_utils.augmentation(
            img, img_segm, bbox, hand_type, self.mode, input_img_shape)
        img_segm = torch.FloatTensor(img_segm).permute(2, 0, 1)
        # if do_flip:
            # img_segm = data_utils.swap_lr_labels_segm_target_channels(img_segm)

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

image_db = ImageDataset(transforms.ToTensor(), 'train', 'train')
indices = np.random.randint(58515, size=200)

for index in indices:
    i, t, m = image_db[index]
    img = i['img']
    msk = t['segm_128']
    np.save('./data_sample/img_{}.npy'.format(index), img)
    np.save('./data_sample/msk_{}.npy'.format(index), msk)
