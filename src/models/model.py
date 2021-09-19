import torch
import torch.nn as nn
import models.loss as loss
from models.loss import jaccard_loss
import torch.nn.functional as F
import numpy as np
import utils
import itertools
from torch.autograd import Variable
from utils import things2dev
from models.smp_models import DeepLabV3PlusB5, PSPNetL2, SPNResNet34, UXceptionNet, DeepLab_ResNext, USeResNext, UPlusRegNet, TryNet
from models.seresnext.seresnext import SEResNetx50_32
from models.discriminator.discriminator import Discriminator, DiscriminatorPix2Pix, DiscriminatorPatch
from torchsummary import summary
from models.resnet import GeneratorResNet, DiscriminatorCycle
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101

real_label = 1
fake_label = 0

class G_XY(nn.Module):
    def __init__(self, num_channels, num_classes, segmentation_model, saved_weights_path_g=None):
        super().__init__()

        if segmentation_model == 'deeplabv3':
            self.segm_model = DeepLabV3PlusB5(num_channels, num_classes)
        elif segmentation_model == 'deeplab_resnext':
            self.segm_model = DeepLab_ResNext(num_channels, num_classes)
        elif segmentation_model == 'uxnet':
            self.segm_model = UXceptionNet(num_channels, num_classes)
        elif segmentation_model == 'resnet':
            self.segm_model = GeneratorResNet(num_channels, num_classes)
        elif segmentation_model == 'seresnext':
            self.segm_model = USeResNext(num_channels, num_classes)
        elif segmentation_model == 'uplusregnet':
            self.segm_model = UPlusRegNet(num_channels, num_classes)
        elif segmentation_model == 'trynet':
            self.segm_model = TryNet(num_channels, num_classes)


        if saved_weights_path_g not in [None, 'None']:
            self.load_weights(saved_weights_path_g)

    def forward(self, img, uncertain=False):
        batch_size, _, im_dim, _ = img.shape

        # segmentation logits for prediction
        amodal_logits = self.segm_model(img)

        # split segmentation logits for left and right hands
        amodal_logits_l, amodal_logits_r = torch.split(amodal_logits, 17, 1)

        # convert logits to classes
        conf_l, segm_mask_l = self.map2labels(amodal_logits_l, True)
        conf_r, segm_mask_r = self.map2labels(amodal_logits_r, True)

        segm_dict = {}
        if uncertain:
            total_conf = conf_l + conf_r
            order = torch.argsort(total_conf, descending=True)
            segm_mask_l = segm_mask_l[order]
            segm_mask_r = segm_mask_r[order]
            segm_dict['order'] = order

        segm_dict['segm_mask_l'] = segm_mask_l  # segmentation classes
        segm_dict['segm_mask_r'] = segm_mask_r
        segm_dict['segm_logits'] = amodal_logits  # logits for the segmentaion
        segm_dict['segm_logits_l'] = amodal_logits_l
        segm_dict['segm_logits_r'] = amodal_logits_r

        out_dict = {}
        out_dict['segm_dict'] = segm_dict
        return out_dict

    def map2labels(self, segm_hand, return_conf=False):
        # convert segmentation logits to labels
        with torch.no_grad():
            segm_hand = segm_hand.permute(0, 2, 3, 1)
            # class with max response
            maxes, pred_segm_hand = segm_hand.max(dim=3)
            if return_conf:
                return maxes.sum((1,2)), pred_segm_hand
            return pred_segm_hand

    def load_weights(self, weights_path):
        self.segm_model.load_state_dict(torch.load(weights_path))
        self.segm_model.eval()


class ModelWrapper(nn.Module):
    def __init__(self, segmentation_model_g1, segmentation_model_g2, lr, segm_loss_weight, main_loss_weight, saved_weights_path_g1=None, discriminator_w=0, saved_weights_path_d=None, old_weights=False, main_loss='dice', use_weights=True):
        super(ModelWrapper, self).__init__()

        self.semi_sup_lmbd = 0.1
        self.semi_adv_lmbd = 0.001
        self.adv_lmbd = 0.01
        self.semi_treshold = 0.2
        self.lr = lr
        self.segm_loss_weight = segm_loss_weight
        self.main_loss_weight = main_loss_weight
        
        if discriminator_w > 0:
            self.discriminator_weight = discriminator_w
            self.discriminator = Discriminator()
            if saved_weights_path_d not in [None, 'None']:
                self.discriminator.load_state_dict(torch.load(saved_weights_path_d))
            self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
            self.use_discriminator = True
        else:
            self.use_discriminator = False

        self.G1 = G_XY(3, 2*17, segmentation_model_g1, saved_weights_path_g1) # generator
        self.G2 = G_XY(3, 2*17, segmentation_model_g2)

        Tensor = torch.cuda.FloatTensor
        self.optimizerG1 = torch.optim.Adam(self.G1.parameters(), lr=self.lr)
        self.optimizerG2 = torch.optim.Adam(self.G2.parameters(), lr=self.lr)

        def lin(epoch):
            return 0.0002 if epoch < 40 else 0.0002-0.000006666*(epoch-39)

        # loss functions
        if old_weights:
            class_weights = [0.1, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 0.35, 0.35, 0.8, 0.8, 1.0, 0.8, 0.8, 0.8, 0.8]
        else:
            class_weights = [0.07, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 0.4, 0.4, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2]
        self.segm_loss = loss.SegmLoss(class_weights, use_weights)

        if main_loss == 'dice':
            self.main_loss = loss.DiceLoss()
        elif main_loss == 'jaccard':
            self.main_loss = jaccard_loss
        else:
            self.main_loss = loss.SegmLoss(class_weights, use_weights)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward_test(self, inputs, meta_info):
        # this forward function is used for test set
        # as the test set does not contain segmentation annotation
        # please be careful when changing this function in case it breaks packaging code for submission
        input_img = inputs['img']
        model_dict = self.G2(input_img)
        segm_dict = model_dict['segm_dict']

        # images involved in this test set
        im_path = meta_info['im_path']
        im_path = [p.replace('./data/InterHand2.6M/images/', '').replace('.jpg', '') for p in im_path]

        # predictions
        return {'segm_l': segm_dict['segm_mask_l'],
                'segm_r': segm_dict['segm_mask_r'],
                'im_path': im_path}

    def forward(self, inputs, targets, meta_info, mode, unp_img, phase):
        loss_dict = {}

        if phase == "pre":
            ###### GENERATOR 
            input_img = inputs['img']    #(batch, 3, 128, 128)
            segm_target_128 = targets['segm_128']
            segm_target_l = torch.reshape(segm_target_128[:, 0], (-1, 128, 128, 1)).permute(0, 3, 1, 2)
            segm_target_r = torch.reshape(segm_target_128[:, 1], (-1, 128, 128, 1)).permute(0, 3, 1, 2)
            segm_target = torch.cat((segm_target_l, segm_target_r), axis=1)

            self.optimizerG1.zero_grad()
            self.optimizerD.zero_grad()

            model_dict = self.G1(input_img)
            segm_dict = model_dict['segm_dict']
            segm_dict['segm_128'] = segm_target_128  

            loss_G1 = self.segm_loss(segm_dict)

            if self.use_discriminator:
                disc_input_l = torch.reshape(segm_dict['segm_mask_l'], (-1, 128, 128, 1)).permute(0, 3, 1, 2)
                disc_input_r = torch.reshape(segm_dict['segm_mask_r'], (-1, 128, 128, 1)).permute(0, 3, 1, 2)
                disc_input = torch.cat((disc_input_l, disc_input_r), axis=1)

                disc_predictions = self.discriminator(disc_input.float())
                target = self.make_D_label(fake_label, disc_predictions)
                loss_dict['loss_disc'] = self.bce_loss(disc_predictions, target)
                
                #real 
                disc_predictions = self.discriminator(segm_target.float())
                target = self.make_D_label(real_label, disc_predictions)
                loss_dict['loss_disc'] += self.bce_loss(disc_predictions, target)

            
            if mode == 'train':
                loss_G1.backward()
                self.optimizerG1.step()

                loss_dict['loss_disc'].backward()
                self.optimizerD.step()
                loss_dict['loss_segm'] = loss_G1
                return loss_dict
                
        # G1 is already trained and generates good segmentation. We give this new pair to G2 for its training
        if phase == "init":
            loss_dict = {}
            real_X = unp_img['img']
            model_dict = self.G1(real_X, uncertain=True)
            segm_dict = model_dict['segm_dict']
            
            new_len = len(real_X) // 2
            real_X_filtered = (real_X[segm_dict['order']])[:new_len]
            fake_Y_l = (torch.reshape(segm_dict['segm_mask_l'], (-1, 128, 128, 1)).permute(0, 3, 1, 2))[:new_len]
            fake_Y_r = (torch.reshape(segm_dict['segm_mask_r'], (-1, 128, 128, 1)).permute(0, 3, 1, 2))[:new_len]
            fake_Y = torch.cat((fake_Y_l, fake_Y_r), 1)
            segm_dict['segm_128'] = fake_Y

            self.optimizerG2.zero_grad()

            model_dict2 = self.G2(real_X_filtered)
            segm_dict2 = model_dict2['segm_dict']
            segm_dict2['segm_128'] = fake_Y

            loss_dice = self.main_loss(segm_dict2['segm_logits_l'], fake_Y_l) + self.main_loss(segm_dict2['segm_logits_r'], fake_Y_r)

            loss_segm = self.segm_loss(segm_dict2)
            loss_G2 = self.segm_loss_weight * loss_segm + self.main_loss_weight * loss_dice
            if mode == 'train':
                loss_G2.backward()
                self.optimizerG2.step()
        
                loss_dict['loss_segm'] = loss_G2
                return loss_dict

        # #G2 is pretrained on our own generated paired dataset, now finetune it on real paired dataset
        if phase == "tr" or phase == None:
            input_img = inputs['img']    #(batch, 3, 128, 128)
            segm_target_128 = targets['segm_128']
            segm_target_l = torch.reshape(segm_target_128[:, 0], (-1, 128, 128, 1)).permute(0, 3, 1, 2)
            segm_target_r = torch.reshape(segm_target_128[:, 1], (-1, 128, 128, 1)).permute(0, 3, 1, 2)
            
            for g in self.optimizerG2.param_groups:
                g['lr'] = 0.00015

            self.optimizerG2.zero_grad()

            model_dict = self.G2(input_img)
            segm_dict = model_dict['segm_dict']
            segm_dict['segm_128'] = segm_target_128  
            
            loss_dice = self.main_loss(segm_dict['segm_logits_l'], segm_target_l) + self.main_loss(segm_dict['segm_logits_r'], segm_target_r)
            loss_segm = self.segm_loss(segm_dict)
            loss_G2 = self.segm_loss_weight * loss_segm + self.main_loss_weight * loss_dice

            if self.use_discriminator:
                disc_input_l = torch.reshape(segm_dict['segm_mask_l'], (-1, 128, 128, 1)).permute(0, 3, 1, 2)
                disc_input_r = torch.reshape(segm_dict['segm_mask_r'], (-1, 128, 128, 1)).permute(0, 3, 1, 2)
                disc_input = torch.cat((disc_input_l, disc_input_r), axis=1)
                disc_predictions = self.discriminator(disc_input.float())
                loss_adversarial = self.bce_loss(disc_predictions, self.make_D_label(real_label, disc_predictions))
                loss_G2 = loss_G2*(1-self.discriminator_weight) + self.discriminator_weight*loss_adversarial

            if mode == 'train':
                loss_G2.backward()
                self.optimizerG2.step()

                loss_dict['loss_segm'] = loss_G2
                return loss_dict
   
        if mode == 'vis':
            # if visualization, return vis_dict containing objects for visualization
            vis_dict = {}

            segm_l_mask_128 = F.interpolate(
                    segm_dict['segm_mask_l'].float()[:, None, :, :], 128, mode='nearest').long().squeeze()
            segm_r_mask_128 = F.interpolate(
                    segm_dict['segm_mask_r'].float()[:, None, :, :], 128, mode='nearest').long().squeeze()

            # packaging for visualization
            vis_dict['segm_l_mask'] = utils.tensor2np(segm_l_mask_128)
            vis_dict['segm_r_mask'] = utils.tensor2np(segm_r_mask_128)
            vis_dict['input_img'] = utils.tensor2np(input_img)

            # segmentation groundtruth
            vis_dict['segm_target_128'] = utils.tensor2np(
                    F.interpolate(segm_target_128.float(), 128, mode='nearest').long())
            vis_dict['im_path'] = meta_info['im_path']
            return vis_dict

        # for evaluation
        loss_dict = {k: loss_dict[k].mean() for k in loss_dict}
        loss_dict['total_loss'] = sum(loss_dict[k] for k in loss_dict)

        # predictions
        segm_l_mask_128 = segm_dict['segm_mask_l']
        segm_r_mask_128 = segm_dict['segm_mask_r']

        # GT
        segm_target = segm_target_128
        segm_target_l = segm_target[:, 0]
        segm_target_r = segm_target[:, 1]

        # evaluation loop
        #pass to numpy
        ious_l = []
        ious_r = []
        for idx in range(segm_target.shape[0]):
            # Warning: do not modify these two lines
            iou_l = utils.segm_iou(segm_l_mask_128[idx], segm_target_l[idx], 17, 20)
            iou_r = utils.segm_iou(segm_r_mask_128[idx], segm_target_r[idx], 17, 20)
            ious_l.append(iou_l)
            ious_r.append(iou_r)

        out = {}
        out['loss'] = loss_dict
        out['ious'] = np.array(ious_l + ious_r)
        return out


    def make_D_label(self, label, ignore_mask):
        D_label = torch.ones(ignore_mask.shape, dtype=torch.float)*label
        return things2dev(D_label, 'cuda')

