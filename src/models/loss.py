import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segmentation_models_pytorch.utils.losses import DiceLoss
from utils import things2dev
from torch.autograd import Function
from typing import Optional, List
from torchvision.transforms.functional import to_tensor


class SegmLoss(nn.Module):
    def __init__(self, class_weights, use_weights):
        super().__init__()
        if use_weights:
            self.weight = torch.tensor(class_weights)
            self.weight = things2dev(self.weight, 'cuda')
            self.loss = nn.CrossEntropyLoss(weight=self.weight, reduction='mean')
        else:
            self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, data_dict):
        segm_128 = data_dict['segm_128']
        segm_target_l = segm_128[:, 0]
        segm_target_r = segm_128[:, 1]
        segm_logits = data_dict['segm_logits']
        segm_logits_l, segm_logits_r = torch.split(segm_logits, 17, 1)

        # amodal loss
        dist_hand_l = self.loss(segm_logits_l, segm_target_l).view(-1)
        dist_hand_r = self.loss(segm_logits_r, segm_target_r).view(-1)

        total_loss = dist_hand_l + dist_hand_r
        return total_loss


class CrossEntropy2d(nn.Module):

    def _init_(self):
        super(CrossEntropy2d, self)._init_()
        self.weight = torch.tensor([0.1, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 0.35, 0.35, 0.8, 0.8, 1.0, 0.8, 0.8, 0.8, 0.8])
        self.weight = things2dev(self.weight, 'cuda')
        self.loss = nn.CrossEntropyLoss(weight=self.weight, reduction='mean')
    
    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        loss = self.loss(predict, target)
        return loss

class BCEWithLogitsLoss2d(nn.Module):

    def _init_(self):
        super(BCEWithLogitsLoss2d, self)._init_()
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        
        loss = self.loss(predict, target)
        return loss

def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


class DiceLoss(nn.Module):

    def __init__(
        self,
        mode: str = "multiclass",
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.1,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Implementation of Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error 
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {"binary", "multilabel", "multiclass"}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != "binary", "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == "multiclass":
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == "binary":
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == "multiclass":
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == "multilabel":
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()



class FocalBinaryTverskyFunc(Function):
    """
        Focal Tversky Loss as defined in `this paper <https://arxiv.org/abs/1810.07842>`_
        `Authors' implementation <https://github.com/nabsabraham/focal-tversky-unet>`_ in Keras.
        Params:
            :param alpha: controls the penalty for false positives.
            :param beta: penalty for false negative.
            :param gamma : focal coefficient range[1,3]
            :param reduction: return mode
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
            add focal index -> loss=(1-T_index)**(1/gamma)
    """

    def __init__(ctx, alpha=0.5, beta=0.7, gamma=1.0, reduction='mean'):
        """
        :param alpha: controls the penalty for false positives.
        :param beta: penalty for false negative.
        :param gamma : focal coefficient range[1,3]
        :param reduction: return mode
        Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
        add focal index -> loss=(1-T_index)**(1/gamma)
        """
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.epsilon = 1e-6
        ctx.reduction = reduction
        ctx.gamma = gamma
        sum = ctx.beta + ctx.alpha
        if sum != 1:
            ctx.beta = ctx.beta / sum
            ctx.alpha = ctx.alpha / sum

    # @staticmethod
    def forward(ctx, input, target):
        batch_size = input.size(0)
        _, input_label = input.max(1)

        input_label = input_label.float()
        target_label = target.float()

        ctx.save_for_backward(input, target_label)

        input_label = input_label.view(batch_size, -1)
        target_label = target_label.view(batch_size, -1)

        ctx.P_G = torch.sum(input_label * target_label, 1)  # TP
        ctx.P_NG = torch.sum(input_label * (1 - target_label), 1)  # FP
        ctx.NP_G = torch.sum((1 - input_label) * target_label, 1)  # FN

        index = ctx.P_G / (ctx.P_G + ctx.alpha * ctx.P_NG + ctx.beta * ctx.NP_G + ctx.epsilon)
        loss = torch.pow((1 - index), 1 / ctx.gamma)
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if ctx.reduction == 'none':
            loss = loss
        elif ctx.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

    # @staticmethod
    def backward(ctx, grad_out):
        """
        :param ctx:
        :param grad_out:
        :return:
        d_loss/dT_loss=(1/gamma)*(T_loss)**(1/gamma-1)
        (dT_loss/d_P1)  = 2*P_G*[G*(P_G+alpha*P_NG+beta*NP_G)-(G+alpha*NG)]/[(P_G+alpha*P_NG+beta*NP_G)**2]
                        = 2*P_G
        (dT_loss/d_p0)=
        """
        inputs, target = ctx.saved_tensors
        inputs = inputs.float()
        target = target.float()
        batch_size = inputs.size(0)
        sum = ctx.P_G + ctx.alpha * ctx.P_NG + ctx.beta * ctx.NP_G + ctx.epsilon
        P_G = ctx.P_G.view(batch_size, 1, 1, 1, 1)
        if inputs.dim() == 5:
            sum = sum.view(batch_size, 1, 1, 1, 1)
        elif inputs.dim() == 4:
            sum = sum.view(batch_size, 1, 1, 1)
            P_G = ctx.P_G.view(batch_size, 1, 1, 1)
        sub = (ctx.alpha * (1 - target) + target) * P_G

        dL_dT = (1 / ctx.gamma) * torch.pow((P_G / sum), (1 / ctx.gamma - 1))
        dT_dp0 = -2 * (target / sum - sub / sum / sum)
        dL_dp0 = dL_dT * dT_dp0

        dT_dp1 = ctx.beta * (1 - target) * P_G / sum / sum
        dL_dp1 = dL_dT * dT_dp1
        grad_input = torch.cat((dL_dp1, dL_dp0), dim=1)
        # grad_input = torch.cat((grad_out.item() * dL_dp0, dL_dp0 * grad_out.item()), dim=1)
        return grad_input, None


class MultiTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation
    Args
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each class. If given, it has to be a Tensor of size `C`
    """

    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, reduction='mean', weights=None):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`
        """
        super(MultiTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights

    def forward(self, inputs, targets):
        num_class = inputs.size(1)
        weight_losses = 0.0
        if self.weights is not None:
            assert len(self.weights) == num_class, 'number of classes should be equal to length of weights '
            weights = self.weights
        else:
            weights = [1.0 / num_class] * num_class
        input_slices = torch.split(inputs, [1] * num_class, dim=1)
        for idx in range(num_class):
            input_idx = input_slices[idx]
            input_idx = torch.cat((1 - input_idx, input_idx), dim=1)
            target_idx = (targets == idx) * 1
            loss_func = FocalBinaryTverskyFunc(self.alpha, self.beta, self.gamma, self.reduction)
            loss_idx = loss_func(input_idx, target_idx)
            weight_losses+=loss_idx * weights[idx]
        # loss = torch.Tensor(weight_losses)
        # loss = loss.to(inputs.device)
        # loss = torch.sum(loss)
        return weight_losses


class FocalBinaryTverskyLoss(MultiTverskyLoss):
    """
            Binary version of Focal Tversky Loss as defined in `this paper <https://arxiv.org/abs/1810.07842>`_
            `Authors' implementation <https://github.com/nabsabraham/focal-tversky-unet>`_ in Keras.
            Params:
                :param alpha: controls the penalty for false positives.
                :param beta: penalty for false negative.
                :param gamma : focal coefficient range[1,3]
                :param reduction: return mode
            Notes:
                alpha = beta = 0.5 => dice coeff
                alpha = beta = 1 => tanimoto coeff
                alpha + beta = 1 => F beta coeff
                add focal index -> loss=(1-T_index)**(1/gamma)
        """

    def __init__(self, alpha=0.5, beta=0.7, gamma=1.0, reduction='mean', **kwargs):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        """
        super().__init__(alpha, beta, gamma, reduction)

    def forward(self, inputs, targets):
        return super().forward(inputs, targets.unsqueeze(1))

class SegmLossDice(nn.Module):
    def _init_(self):
        super()._init_()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = BCEWithLogitsLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, data_dict):
        segm_128 = data_dict['segm_128']
        segm_target_l = segm_128[:, 0]
        segm_target_r = segm_128[:, 1]
        segm_logits = data_dict['segm_logits']
        segm_logits_l, segm_logits_r = torch.split(segm_logits, 17, 1)

        # amodal loss
        dist_hand_l = self.loss(segm_logits_l, segm_target_l).mean().view(-1)
        dist_hand_r = self.loss(segm_logits_r, segm_target_r).mean().view(-1)
        #calculate bce loss for 2 hands
        #dice loss
        segm_target_l = torch.unsqueeze(segm_target_l, 1)
        segm_target_r = torch.unsqueeze(segm_target_r, 1)
        dice_loss_l = self.dice_loss(segm_logits_l, segm_target_l)
        dice_loss_r = self.dice_loss(segm_logits_r, segm_target_r)

        #print(bce_loss_l, bce_loss_r)
        total_loss = 0.75*dist_hand_l + 0.75*dist_hand_r + 0.25*dice_loss_l + 0.25*dice_loss_r
        return total_loss

def jaccard_loss(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)
