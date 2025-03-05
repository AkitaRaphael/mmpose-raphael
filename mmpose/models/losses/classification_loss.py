# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


@MODELS.register_module()
class BCELoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            before output. Defaults to False.
    """

    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.,
                 reduction='mean',
                 use_sigmoid=False):
        super().__init__()

        assert reduction in ('mean', 'sum', 'none'), f'the argument ' \
            f'`reduction` should be either \'mean\', \'sum\' or \'none\', ' \
            f'but got {reduction}'

        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        criterion = F.binary_cross_entropy if use_sigmoid \
            else F.binary_cross_entropy_with_logits
        self.criterion = partial(criterion, reduction='none')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = (loss * target_weight)
        else:
            loss = self.criterion(output, target)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.loss_weight


@MODELS.register_module()
class JSDiscretLoss(nn.Module):
    """Discrete JS Divergence loss for DSNT with Gaussian Heatmap.

    Modified from `the official implementation
    <https://github.com/anibali/dsntnn/blob/master/dsntnn/__init__.py>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
    """

    def __init__(
        self,
        use_target_weight=True,
        size_average: bool = True,
    ):
        super(JSDiscretLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.size_average = size_average
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def kl(self, p, q):
        """Kullback-Leibler Divergence."""

        eps = 1e-24
        kl_values = self.kl_loss((q + eps).log(), p)
        return kl_values

    def js(self, pred_hm, gt_hm):
        """Jensen-Shannon Divergence."""

        m = 0.5 * (pred_hm + gt_hm)
        js_values = 0.5 * (self.kl(pred_hm, m) + self.kl(gt_hm, m))
        return js_values

    def forward(self, pred_hm, gt_hm, target_weight=None):
        """Forward function.

        Args:
            pred_hm (torch.Tensor[N, K, H, W]): Predicted heatmaps.
            gt_hm (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.

        Returns:
            torch.Tensor: Loss value.
        """

        if self.use_target_weight:
            assert target_weight is not None
            assert pred_hm.ndim >= target_weight.ndim

            for i in range(pred_hm.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)

            loss = self.js(pred_hm * target_weight, gt_hm * target_weight)
        else:
            loss = self.js(pred_hm, gt_hm)

        if self.size_average:
            loss /= len(gt_hm)

        return loss.sum()


@MODELS.register_module()
class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation.

    <https://github.com/leeyegy/SimCC>`_.
    Args:
        beta (float): Temperature factor of Softmax. Default: 1.0.
        label_softmax (bool): Whether to use Softmax on labels.
            Default: False.
        label_beta (float): Temperature factor of Softmax on labels.
            Default: 1.0.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        mask (list[int]): Index of masked keypoints.
        mask_weight (float): Weight of masked keypoints. Default: 1.0.
    """

    def __init__(self,
                 beta=1.0,
                 label_softmax=False,
                 label_beta=10.0,
                 use_target_weight=True,
                 mask=None,
                 mask_weight=1.0):
        super(KLDiscretLoss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.label_beta = label_beta
        self.use_target_weight = use_target_weight
        self.mask = mask
        self.mask_weight = mask_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.label_beta, dim=1)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        N, K, _ = pred_simcc[0].shape
        loss = 0

        if self.use_target_weight:
            weight = target_weight.reshape(-1)
        else:
            weight = 1.

        for pred, target in zip(pred_simcc, gt_simcc):
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))

            t_loss = self.criterion(pred, target).mul(weight)

            if self.mask is not None:
                t_loss = t_loss.reshape(N, K)
                t_loss[:, self.mask] = t_loss[:, self.mask] * self.mask_weight

            loss = loss + t_loss.sum()

        return loss / K


@MODELS.register_module()
class InfoNCELoss(nn.Module):
    """InfoNCE loss for training a discriminative representation space with a
    contrastive manner.

    `Representation Learning with Contrastive Predictive Coding
    arXiv: <https://arxiv.org/abs/1611.05424>`_.

    Args:
        temperature (float, optional): The temperature to use in the softmax
            function. Higher temperatures lead to softer probability
            distributions. Defaults to 1.0.
        loss_weight (float, optional): The weight to apply to the loss.
            Defaults to 1.0.
    """

    def __init__(self, temperature: float = 1.0, loss_weight=1.0) -> None:
        super(InfoNCELoss, self).__init__()
        assert temperature > 0, f'the argument `temperature` must be ' \
                                f'positive, but got {temperature}'
        self.temp = temperature
        self.loss_weight = loss_weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Computes the InfoNCE loss.

        Args:
            features (Tensor): A tensor containing the feature
                representations of different samples.

        Returns:
            Tensor: A tensor of shape (1,) containing the InfoNCE loss.
        """
        n = features.size(0)
        features_norm = F.normalize(features, dim=1)
        logits = features_norm.mm(features_norm.t()) / self.temp
        targets = torch.arange(n, dtype=torch.long, device=features.device)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        return loss * self.loss_weight


@MODELS.register_module()
class VariFocalLoss(nn.Module):
    """Varifocal loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        alpha (float): A balancing factor for the negative part of
            Varifocal Loss. Defaults to 0.75.
        gamma (float): Gamma parameter for the modulating factor.
            Defaults to 2.0.
    """

    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.,
                 reduction='mean',
                 alpha=0.75,
                 gamma=2.0):
        super().__init__()

        assert reduction in ('mean', 'sum', 'none'), f'the argument ' \
            f'`reduction` should be either \'mean\', \'sum\' or \'none\', ' \
            f'but got {reduction}'

        self.reduction = reduction
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.gamma = gamma

    def criterion(self, output, target):
        label = (target > 1e-4).to(target)
        weight = self.alpha * output.sigmoid().pow(
            self.gamma) * (1 - label) + target
        output = output.clip(min=-10, max=10)
        vfl = (
            F.binary_cross_entropy_with_logits(
                output, target, reduction='none') * weight)
        return vfl

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss = (loss * target_weight)
        else:
            loss = self.criterion(output, target)

        loss[torch.isinf(loss)] = 0.0
        loss[torch.isnan(loss)] = 0.0

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.loss_weight

@MODELS.register_module()
class paraLoss(nn.Module):
    """Limb Visibility Loss.

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=0.5):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), f"Invalid reduction mode: {reduction}"
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, kpt_vis_preds, vis_targets):
        limb_pairs = torch.tensor([
            [7, 17], [9, 19], [8, 18], [10, 20], [13, 21], [15, 23], [14, 22], [16, 24]
        ], device=kpt_vis_preds.device)

        pred_pairwise_confidence = torch.cat((kpt_vis_preds[:, limb_pairs[:, 0, None]], kpt_vis_preds[:, limb_pairs[:, 1, None]]), dim=-1).flatten(start_dim=1)
        visibilities = vis_targets[:, limb_pairs].flatten(start_dim=1)

        loss = self.criterion(pred_pairwise_confidence, visibilities)

        # pred_pairwise_confidence = torch.cat((kpt_vis_preds[:, limb_pairs[:, 0, None]], kpt_vis_preds[:, limb_pairs[:, 1, None]]), dim=-1)
        # visibilities = vis_targets[:, limb_pairs]  # (N, num_pairs, 2)
        #
        # indices = ~((visibilities == 0).all(dim=-1))
        #
        # pred_pairwise_confidence = [pred_pairwise_confidence[i][indices[i]] for i in range(pred_pairwise_confidence.size(0))]
        # pred_pairwise_confidence = torch.cat(pred_pairwise_confidence, dim=0)  # shape: [88, n, 2]
        #
        # visibilities = [visibilities[i][indices[i]] for i in range(visibilities.size(0))]
        # visibilities = torch.cat(visibilities, dim=0)
        #
        # loss = self.criterion(pred_pairwise_confidence, visibilities)


        # pred_diffs = kpt_vis_preds[:, limb_pairs[:, 0]] - kpt_vis_preds[:, limb_pairs[:, 1]]
        #
        # valid_mask = (visibilities.sum(dim=-1) == 1)  # 只有一个关键点可见
        # valid_pred_diffs = pred_diffs[valid_mask].sigmoid()
        # valid_vis = visibilities[valid_mask]
        # target_diff = valid_vis[:, 0] - valid_vis[:, 1]
        # # loss = F.mse_loss(valid_pred_diffs, target_diff)
        # loss = torch.log(torch.cosh(valid_pred_diffs - target_diff))

        # # gt可见的点 - 不可见的点
        # target_diff = torch.where(valid_vis[:, 0] == 1, valid_pred_diffs, -valid_pred_diffs)
        #
        # # Loss设计：差值越大Loss越小
        # loss = 1 / (1 + torch.exp(target_diff))

        # if self.reduction == 'sum':
        #     loss = loss.sum()
        # elif self.reduction == 'mean':
        #     loss = loss.mean()
        # else:
        #     loss = torch.tensor(0.0, device=kpt_vis_preds.device)

        return loss * self.loss_weight


    # def forward(self, kpt_vis_preds, vis_targets):
    #     """Forward function.
    #
    #     Args:
    #         kpt_vis_preds (torch.Tensor[N, K]): Keypoint visibility predictions.
    #         vis_targets (torch.Tensor[N, K]): Target keypoint visibility labels.
    #     """
    #     limb_pairs = torch.tensor([
    #         [7, 17], [9, 19], [8, 18], [10, 20], [13, 21], [15, 23], [14, 22], [16, 24]
    #     ], device=kpt_vis_preds.device)
    #
    #     visibilities = vis_targets[:, limb_pairs]  # (N, num_pairs, 2)
    #     pred_diffs = kpt_vis_preds[:, limb_pairs[:, 0]] - kpt_vis_preds[:, limb_pairs[:, 1]]
    #
    #     valid_mask = (visibilities.sum(dim=-1) > 0)  # 至少一个关键点可见
    #     valid_pred_diffs = pred_diffs[valid_mask]
    #     valid_vis = visibilities[valid_mask]
    #
    #     # 目标差异值： p1可见设为-1，p2可见设为1，两个都可见设为0
    #     target_diff = (valid_vis[:, 1] - valid_vis[:, 0])
    #
    #     # 用MSE或L1 loss约束预测差值接近目标差值
    #     loss = F.mse_loss(valid_pred_diffs, target_diff.float(), reduction=self.reduction)
    #
    #
    #     if self.reduction == 'sum':
    #         loss = loss.sum()
    #     elif self.reduction == 'mean':
    #         loss = loss.mean()
    #
    #     else:
    #         loss = torch.tensor(0.0, device=kpt_vis_preds.device)  # Avoid NaN issues
    #
    #     return loss * self.loss_weight

