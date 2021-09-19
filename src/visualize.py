import training.trainer as tr
import datasets
from pprint import pprint
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from datasets import ImageDataset
import utils
import os.path as op
from tqdm import tqdm
import shutil

import torch
import numpy as np
from PIL import Image

def create_visualization(args):
    train_loader = datasets.fetch_dataloader('train', args.trainsplit, args)
    train_val_loader = {
        'loader': datasets.fetch_dataloader('train', args.trainsplit, args),
        'postfix': "__train"
    }
    train_val_loader['vis_batch'] = next(iter(train_val_loader['loader']))
    val_val_loader = {
            'loader': datasets.fetch_dataloader('val', args.valsplit, args),
            'postfix': "__val"}
    val_val_loader['vis_batch'] = next(iter(val_val_loader['loader']))

    trainer = tr.Trainer(train_loader, [train_val_loader, val_val_loader], args)

    train_loader = trainer.val_loaders[0]
    train_vis = trainer.visualize_batches(
        train_loader['vis_batch'], postfix=train_loader['postfix'], num_examples=args.batch_size)
    
    val_loader = trainer.val_loaders[1]
    val_vis = trainer.visualize_batches(
        val_loader['vis_batch'], postfix=val_loader['postfix'], num_examples=args.batch_size, no_tqdm=False)
    
    root_path = './visualize_results/'
    train_path = op.join(root_path, 'train/')
    val_path = op.join(root_path, 'val/')
    utils.mkdir_p(train_path)
    utils.mkdir_p(val_path)
    
    for i,tp in tqdm(enumerate(zip(train_vis,val_vis))):
        (tt,vt) = tp
        t_name = op.join(train_path,'vis_{}.png'.format(i))
        v_name = op.join(val_path,'vis_{}.png'.format(i))
        tt['im'].save(t_name)
        vt['im'].save(v_name)

    shutil.make_archive('./visualize_results', 'zip', root_path)


def create_minibatch_dump(args):
    train_loader = datasets.fetch_dataloader('train', args.trainsplit, args)
    val_loader = datasets.fetch_dataloader('val', args.trainsplit, args)
    
    utils.mkdir_p('./data_sample/train/img/')
    utils.mkdir_p('./data_sample/train/msk/')
    utils.mkdir_p('./data_sample/val/img/')
    utils.mkdir_p('./data_sample/val/msk/')

    counter_tr = 0
    counter_val = 0
    for i in tqdm(range(10)):
        dummy_tr = next(iter(train_loader))
        dummy_val = next(iter(val_loader))

        for j in range(args.batch_size):
            np.save('./data_sample/train/img/train_{}.npy'.format(counter_tr), dummy_tr[0]['img'][j].numpy().transpose((1,2,0)))
            np.save('./data_sample/train/msk/train_{}.npy'.format(counter_tr), dummy_tr[1]['segm_128'][j].numpy())
            counter_tr = counter_tr + 1

        for j in range(args.batch_size):
            np.save('./data_sample/val/img/val_{}.npy'.format(counter_val), dummy_val[0]['img'][j].numpy().transpose((1,2,0)))
            np.save('./data_sample/val/msk/val_{}.npy'.format(counter_val), dummy_val[1]['segm_128'][j].numpy())
            counter_val = counter_val + 1

    shutil.make_archive('./data_sample/', 'zip', root_path)


if __name__ == "__main__":
    from config import args
    create_visualization(args)
