import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import utils


# formating data from vis_dict for plotting
def prepare_data(vis_dict):
    img = vis_dict['input_img']
    im_path = vis_dict['im_path']

    raw_im_list = []
    for im_np in img:
        im_np = (im_np/im_np.max()*255).astype(np.uint8)
        im = Image.fromarray(np.moveaxis(im_np, (0, 1, 2), (2, 0, 1)))
        raw_im_list.append(im)
    del img

    im_path = [
            ip.replace(
                './data/InterHand2.6M/images/', ''
            ).replace('.jpg', '') for ip in im_path]

    plt_dict = {}
    plt_dict['raw_im_list'] = raw_im_list
    plt_dict['im_path'] = im_path
    plt_dict['segm_l_mask'] = vis_dict['segm_l_mask']
    plt_dict['segm_r_mask'] = vis_dict['segm_r_mask']
    plt_dict['segm_target_128'] = vis_dict['segm_target_128']
    return plt_dict


# visualize amodal segmentation with RGB images
def vis_amodal_rgb(
        curr_l_mask, curr_r_mask,
        curr_segm_target_l, curr_segm_target_r, curr_img):

    # do not plot background
    curr_r_mask_clone = curr_r_mask.clone()
    curr_r_mask_clone = curr_r_mask_clone.float()
    curr_r_mask_clone[curr_r_mask_clone == 0] = np.nan
    # pixels with incorrect prediction should be 1, else 0
    diff_r = (curr_r_mask != curr_segm_target_r).long() 

    # same for the left hand
    curr_l_mask_clone = curr_l_mask.clone()
    curr_l_mask_clone = curr_l_mask_clone.float()
    curr_l_mask_clone[curr_l_mask_clone == 0] = np.nan
    diff_l = (curr_l_mask != curr_segm_target_l).long()

    fig, ax = plt.subplots(2, 5, figsize=(15, 8))
    # plot right hand first
    # image with right hand prediction overlaid
    ax[0, 0].imshow(curr_img)
    ax[0, 0].imshow(curr_r_mask_clone, alpha=0.8, vmin=0, vmax=16)

    # right hand prediction
    ax[0, 1].imshow(curr_r_mask, vmin=0, vmax=16)

    # groundtruth
    ax[0, 2].imshow(curr_segm_target_r, vmin=0, vmax=16)

    # error map
    ax[0, 3].imshow(diff_r)

    # input image
    ax[0, 4].imshow(curr_img)

    # plot left hand
    ax[1, 0].imshow(curr_img)
    ax[1, 0].imshow(curr_l_mask_clone, alpha=0.8, vmin=0, vmax=16)
    ax[1, 1].imshow(curr_l_mask, vmin=0, vmax=16)
    ax[1, 2].imshow(curr_segm_target_l, vmin=0, vmax=16)
    ax[1, 3].imshow(diff_l)
    ax[1, 4].imshow(curr_img)
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.close()

    # turn figure into PIL.Image
    im = utils.fig2img(fig)
    return im


def visualize_all(
        vis_dict, max_examples, postfix, no_tqdm):
    # visualze samples from vis_dict

    # prepare plotting data
    plt_dict = prepare_data(vis_dict)
    del vis_dict

    im_list = []
    max_examples = min(max_examples, len(plt_dict['im_path']))
    myrange = tqdm(range(max_examples)) if not no_tqdm else range(max_examples)

    # for each sample
    for example_idx in myrange:
        curr_path = plt_dict['im_path'][example_idx]
        curr_im = plt_dict['raw_im_list'][example_idx]

        curr_l_mask = torch.LongTensor(plt_dict['segm_l_mask'][example_idx])
        curr_r_mask = torch.LongTensor(plt_dict['segm_r_mask'][example_idx])
        curr_segm_target = torch.LongTensor(plt_dict['segm_target_128'][example_idx])
        curr_segm_target_l = torch.LongTensor(curr_segm_target[0])
        curr_segm_target_r = torch.LongTensor(curr_segm_target[1])

        # right hand
        im = vis_amodal_rgb(
            curr_l_mask, curr_r_mask,
            curr_segm_target_l, curr_segm_target_r, curr_im)
        im_list.append(
                {'im': im,
                 'fig_name': curr_path + '_amodal_rgb'})

    return im_list
