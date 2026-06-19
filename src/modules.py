"""Core model, loss, and utility modules for the cryoEM project.

This file glues together three training stages:
    stage 1   : DETR style transformer (object queries -> class/box/mask proposals)
    stage 2   : Graph neural network reasoning over object-level embeddings
    stage mask: High‑resolution segmentation (refines object masks with ViT+UNet hybrid)

Composite / joint stages (e.g. "stage 1 + 2", "stage 1 + 2 + 3") execute these
sub-stages sequentially inside a single Lightning step via `_common_step`.

Key concepts:
    - Boxes sometimes represented as normalized (cx, cy, w, h) in [0,1].
    - `box_masks` / indices of valid boxes can be -1 for padding.
    - For mask training we sub‑sample objects (self.num) using class weights to
        mitigate imbalance.
    - Several auxiliary losses (dice, focal, TV, noise) combined in `CompositeSegBBoxLoss`.
    - `single_chunk_noise_loss` penalizes activations outside predicted boxes.

NOTE: Some experimental / legacy sections remain (commented) for reference.
"""

import numpy as np
import pandas as pd
import pycocotools
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.datasets
import torchvision.transforms.v2 as transforms
from PIL import Image
from pycocotools.coco import COCO
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch_geometric.nn import (
    AGNNConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    SAGEConv,
    TransformerConv,
)
from transformers.image_transforms import center_to_corners_format

import utils


class AdditionalInputLayer(nn.Module):
    """Tiny MLP to project auxiliary per-node features to model dimension."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, in_dim)
        self.layer2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):  # (N, in_dim)
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x  # (N, out_dim)


class EmptyContextManager:
    """No-op context manager used to unify code paths with and without torch.no_grad."""

    def __enter__(self):
        # No setup actions needed（()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # No cleanup actions needed
        pass


from torch import Tensor


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, warmup_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
        # self.base_lr = None
        # print(self.base_lrs)

    def get_lr(self):
        # if self.base_lr is None:
        # Get the base learning rate from the optimizer
        # self.base_lr = [group["lr"] for group in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_epochs:
            # print(self.last_epoch)
            # Linear warmup: Scale LR linearly based on the epoch
            return [self.warmup_lr for lr in self.base_lrs]
        else:
            # After warmup, return the base LR
            return self.base_lrs


# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# Copied from transformers.models.detr.sigmoid_focal_loss (lightly adjusted)
def sigmoid_focal_loss(
    inputs, targets, num_boxes=None, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    inputs = inputs.squeeze()
    targets = targets.squeeze()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if num_boxes is None:
        num_boxes = targets.shape[0]

    return loss.mean(1).sum() / num_boxes


import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-6):
    """Dice loss for binary masks"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def total_variation_loss(mask):
    """Total variation for smoothness"""
    loss = torch.mean(torch.abs(mask[:, :, :-1] - mask[:, :, 1:])) + torch.mean(
        torch.abs(mask[:, :-1, :] - mask[:, 1:, :])
    )
    return loss


# def smooth_l1_loss(pred_box, target_box):
#     return F.smooth_l1_loss(pred_box, target_box)


def single_chunk_noise_loss(pred_mask, target_mask, boxes, eps=0.02, warn=False):
    """Penalize predicted mask activation outside predicted bounding boxes.

    Args:
        pred_mask: (B, H, W) raw logits.
        target_mask: (B, H, W) ground truth (unused except for optional warning).
        boxes: (B, 4) normalized (cx, cy, w, h) in [0,1]. Can be None / empty.
        eps: float padding around each box (normalized) to tolerate minor misalign.
        warn: if True, can emit console warning when GT mostly outside the box.

    Returns:
        Scalar tensor: mean activation outside boxes after sigmoid.
    """
    # Early exit if no boxes
    if boxes is None or boxes.shape[0] == 0:
        return pred_mask.new_tensor(0.0)

    # Detach boxes to ensure no gradients are tracked for their ops
    boxes = boxes.detach()

    B, H, W = pred_mask.shape

    # Convert (cx, cy, w, h) normalized -> pixel index ranges with padding eps
    x_c, y_c, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = torch.clamp(x_c - bw / 2 - eps, min=0.0)
    x2 = torch.clamp(x_c + bw / 2 + eps, max=1.0)
    y1 = torch.clamp(y_c - bh / 2 - eps, min=0.0)
    y2 = torch.clamp(y_c + bh / 2 + eps, max=1.0)

    x1i = (x1 * W).long()
    x2i = (x2 * W).long()
    y1i = (y1 * H).long()
    y2i = (y2 * H).long()

    # Sigmoid predictions (no in-place ops afterwards)
    pm = torch.sigmoid(pred_mask)

    # inside_mask==1 inside the (expanded) box, 0 elsewhere
    inside_mask = torch.zeros_like(pm)
    for i in range(B):
        if x1i[i] < x2i[i] and y1i[i] < y2i[i]:  # valid box
            inside_mask[i, y1i[i] : y2i[i], x1i[i] : x2i[i]] = 1.0

    outside_mask = 1.0 - inside_mask

    # Optional sanity warning (no grad) if target largely outside predicted box
    # if warn and target_mask is not None:
    #     with torch.no_grad():
    #         overlap_ratio = (inside_mask * target_mask).sum() / (target_mask.sum() + 1e-6)
    #         if overlap_ratio < 0.9:
    #             print("warning: target less inside box (overlap {:.3f})".format(float(overlap_ratio)))

    # Mean activation outside the predicted box regions
    loss = (pm * outside_mask).mean()
    return loss


class CompositeSegBBoxLoss(nn.Module):
    def __init__(
        self, lambda_dice=1.0, lambda_bce=1.0, lambda_tv=0.5, lambda_noise=0.5
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_tv = lambda_tv
        self.lambda_noise = lambda_noise
        self.warn = False

    def forward(self, pred_mask, target_mask, pred_box):
        """Compute composite segmentation loss.

        Currently combines: Dice + focal (classification) + TV (smoothness).
        A noise suppression term (mask leakage outside boxes) is computed but
        excluded from total by default (can be re-enabled if desired).
        """
        loss_seg = dice_loss(pred_mask, target_mask)
        loss_bce = sigmoid_focal_loss(pred_mask, target_mask, None, 0.25)
        loss_tv = total_variation_loss(torch.sigmoid(pred_mask))
        loss_noise = single_chunk_noise_loss(
            pred_mask, target_mask, pred_box, warn=self.warn
        )  # not added now

        total_loss = (
            self.lambda_dice * loss_seg
            + self.lambda_bce * loss_bce
            + self.lambda_tv * loss_tv
            + self.lambda_noise * loss_noise  # optional
        )
        return total_loss


class DetrModel(L.LightningModule):
    """
    main model for object detection and segmentation.
    can be trained using different modes:
    stage 1: pretraining and training detr alone
    stage 2: training gnn alone
    stage mask: training mask head alone
    stage 1 + 2: training detr and gnn together
    stage 1 + 2 + 3: training detr, gnn and mask head together
    stage 1 + 2 + 3 mask: train mask head alone but with data augmentation from raw slice input

    all other modes are for debugging and testing purposes.

    Args:
        stage (str): the training stage, can be one of the following:
            - "stage 1": pretraining and training detr alone
            - "stage 2": training gnn alone
            - "stage mask": training mask head alone
            - "stage 1 + 2": training detr and gnn together
            - "stage 1 + 2 + 3": training detr, gnn and mask head together
            - "stage 1 + 2 + 3 mask": train mask head alone but with data augmentation from raw slice input
        model (nn.Module or dict): the detr to be trained, can be a single model or a dictionary of models.
        lr (float): learning rate for the optimizer.
        weight_decay (float): weight decay for the optimizer.
        feature_dim (int): dimension of the input features.
        output_dim (int): number of output classes.
        lr_detr (float, optional): learning rate for the DETR model (the transformer part).
        lr_backbone (float, optional): learning rate for the backbone model in the detr.
        additional_input_dim (int, optional): dimension of additional input features. Defaults to 10.
        additional_output_dim (int, optional): dimension of additional output features. Defaults to 16.
        layer_type (str, optional): type of GNN layer to use. Defaults to "GCNConv".
        dropout (bool, optional): whether to use dropout in GNN layers. Defaults to True.
        scheduler_step (int, optional): step size for the learning rate scheduler. Defaults to -1.
        warmup_epoches (int, optional): number of warmup epochs for the learning rate scheduler. Defaults to 1.
        pick_num (int, optional): number of objects to pick from each image for mask training. Defaults to 6.
        mask_alpha (float, optional): alpha value for focal loss in mask head. Defaults to 0.8.
    """

    def __init__(
        self,
        stage,
        model,
        lr,
        weight_decay,
        feature_dim,
        output_dim,
        lr_detr=None,
        lr_backbone=None,
        gnn_in_channel=10,
        layer_type="GCNConv",
        dropout=True,
        scheduler_step=-1,
        warmup_epoches=1,
        pick_num=6,
        mask_alpha=0.8,
        mask_in_channel=3,
        mask_out_channel=1,
        class_weights=None,
        consistency_regularization_coef=0.5,
        box_head="lora",
        nms = -1.0
    ):
        super().__init__()
        if isinstance(model, dict):
            self.is_dict = True
            self.model = nn.ModuleDict(model)
        else:
            self.is_dict = False
            self.model = model

        assert stage in [
            "stage 1",
            "stage 2",
            "stage 1 mask",
            "stage 1 + 2",
            "stage 1 + 2 + 3",
            "stage 1 + 2 + 3 mask",
            "stage pretrain mask",
            "stage mask",
        ]
        print("model at stage ", stage)
        self.stage = stage
        self.warmup_epoches = warmup_epoches
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.lr_detr = lr_detr
        self.weight_decay = weight_decay
        self.training_step_outputs = []
        self.val_step_outputs = []
        self.gnn_in_channel = gnn_in_channel
        self.feature_dim = feature_dim
        self.box_head = box_head
        self.nms = nms

        self.acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=output_dim, average="macro"
        )
        self.auroc = torchmetrics.AUROC(
            num_classes=output_dim, average="macro", task="multiclass"
        )
        self.mask_auroc = torchmetrics.AUROC(task="binary")

        self.gnn = GCN(
            feature_dim,
            gnn_in_channel,
            output_dim,
            layer_type=layer_type,
            dropout=dropout,
            box_head=box_head,
        )
        # self.class_weights = class_weights
        if class_weights is not None:
            t = torch.tensor(class_weights, dtype=torch.float32)
        else:
            t = torch.ones((output_dim))
            t[-1] = 0.1

        self.class_weights = (t - t[-1]).numpy()  # - t[-1]

        print("model with output classes", output_dim)
        print("model receiving class weights", t)

        self.box_loss = CompositeSegBBoxLoss()

        self.cri = nn.CrossEntropyLoss(weight=t)
        self.edge_cri = nn.BCEWithLogitsLoss(reduction="none")
        self.kv = nn.KLDivLoss(reduction="batchmean")

        print("using consistency regularization coef", consistency_regularization_coef)
        self.consistency_regularization_coef = consistency_regularization_coef

        self.output_dim = output_dim
        self.mask_head = VitForMask(
            embed_dim=feature_dim,
            sigmoid=False,
            c_in=mask_in_channel,
            c_out=mask_out_channel,
        )

        self.scheduler_step = scheduler_step

        self.num = pick_num
        self.mask_alpha = mask_alpha

        self.box_in_for_mask = True

    def forward(
        self,
        x=None,
        edge_index=None,
        pixel_values=None,
        pixel_mask=None,
        labels=None,
        mark=None,
        embed=None,
        box=None,
    ):
        if "stage 1" in self.stage:
            assert pixel_values is not None
            if self.is_dict:
                ret = {}
                if mark is not None:
                    ret[mark] = self.model[mark](
                        pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
                    )
                else:
                    for i in self.model:
                        ret[i] = self.model[i](
                            pixel_values=pixel_values,
                            pixel_mask=pixel_mask,
                            labels=labels,
                        )
                return ret
            else:
                return self.model(
                    pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
                )
        elif "stage 2" in self.stage:
            assert x is not None
            assert edge_index is not None
            # additional_input, model_feature = (
            #     x[:, : self.additional_input_dim],
            #     x[:, self.additional_input_dim :],
            # )
            # additional_feature = self.additional_input_layer(additional_input)
            # inputs = torch.cat([model_feature, additional_feature], dim=1)
            return self.gnn(x, edge_index)

        elif "stage mask" in self.stage:
            assert embed is not None
            assert pixel_values is not None
            if self.box_in_for_mask:
                assert box is not None
                return self.mask_head(
                    pixel_values,
                    embed,
                    box,  # .unsqueeze(1).repeat(1, self.num, 1),
                )
            else:
                return self.mask_head(pixel_values, embed)

        else:
            raise NotImplementedError

    def loss_boxes(self, source_boxes, targets, num_boxes):

        loss_bbox = nn.functional.mse_loss(source_boxes, targets)

        return loss_bbox  # + 2 * loss_giou

    def common_step_stage1(self, batch, return_outputs=False):
        if "stage 1" in self.stage:

            pixel_values = batch["pixel_values"]  # .to(self.device)
            b, c, h, w = pixel_values.shape
            if c > 3:
                pixel_values = pixel_values[:, :3, :, :]

            pixel_values = pixel_values.to(self.device)

            pixel_mask = batch["pixel_mask"].to(self.device)
            # if "mark" in batch:
            #     mark = batch["mark"][0]
            # else:
            #     mark = None

            # if mark == "":
            #     mark = None
            required_labels = []
            for t in batch["labels"]:
                sample = {}
                for q in ["class_labels", "boxes", "masks"]:
                    if q in t:
                        sample[q] = t[q].to(self.device)
                required_labels.append(sample)
            # labels = [
            #     {k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]
            # ]

            # print(pixel_values, pixel_mask, required_labels)

            outputs = self(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=required_labels,
                # mark=mark,
            )
            # if mark is not None:
            #     loss += outputs[mark].loss
            #     if loss_dict is None:
            #         loss_dict = outputs[mark].loss_dict.detach().cpu()
            #     else:
            #         for i in loss_dict:
            #             loss_dict[i] += outputs[mark].loss_dict[i].detach().cpu()
            # else:
            loss = outputs.loss
            loss_dict = outputs.loss_dict
            for i in loss_dict:
                loss_dict[i] = loss_dict[i].detach().cpu()
            loss_dict["loss"] = loss.detach().cpu()
            if return_outputs:
                return loss, loss_dict, outputs
            return loss, loss_dict
        else:
            raise ValueError("not in stage 1")

    def common_step_stage2(
        self,
        x,
        edge,
        mask,
        y,
        box=None,
        box_mask=None,
        edge_label=None,
        edge_mask=None,
        edge_type=None,
        return_outputs=False,
    ):

        x = x.to(self.device)
        edge = edge.to(self.device)
        mask = mask.to(self.device)
        y = y.to(self.device)

        ret_dict = self(x=x, edge_index=edge)
        # print(ret_dict["predict"][mask].shape, y[mask].shape)
        # print(torch.max(y[mask]))
        # print(ret_dict["predict"].shape)
        # print(y)
        # print(ret_dict["predict"][mask].shape)
        loss = self.cri(ret_dict["predict"][mask], y[mask])
        if edge_type is not None and sum(edge_type) > 0:
            # print(edge_type)
            # print(edge_type)
            # print(sum(edge_type))
            inter_edge = (edge.T)[edge_type]
            n1 = ret_dict["predict"][inter_edge[:, 0]]
            n2 = ret_dict["predict"][inter_edge[:, 1]]
            # print(n1.shape, n2.shape)
            loss_consistency = self.kv(
                torch.log_softmax(n1, dim=1),
                torch.softmax(n2, dim=1),
            ) + self.kv(
                torch.log_softmax(n2, dim=1),
                torch.softmax(n1, dim=1),
            )
            loss += self.consistency_regularization_coef * loss_consistency
        # self.auroc.update(ret_dict["predict"][mask].detach(), y[mask])
        # print("before auroc")
        loss_dict = {
            "loss": loss.detach().cpu(),
            "auroc": self.auroc(ret_dict["predict"][mask].detach(), y[mask]).cpu(),
        }
        # print("after auroc")
        self.auroc.reset()

        if box is not None:
            box_mask = box_mask.to(self.device)
            box = box.to(self.device)
            box_mask = box_mask & mask
            loss_boxes = self.loss_boxes(
                ret_dict["box"][box_mask], box[box_mask], box_mask.sum()
            )
            loss += loss_boxes
            loss_dict["loss_boxes"] = loss_boxes.detach().cpu()
        else:
            loss_boxes = 0
        # print("before edge loss")
        if edge_label is not None:
            edge_label = edge_label.to(self.device)
            ret_dict["edge"] = ret_dict["edge"].to(self.device)
            if edge_mask is not None:
                # edge_mask = torch.ones_like(edge_label, dtype=torch.bool)
                edge_mask = edge_mask.to(self.device)
                ret_dict["edge"] = ret_dict["edge"][edge_mask]
                edge_label = edge_label[edge_mask]
            # print(
            #     "shape before loss",
            #     ret_dict["edge"].squeeze().shape,
            #     edge_label.float().shape,
            # )
            edge_label = edge_label.float()
            loss_edge = self.edge_cri(ret_dict["edge"].squeeze(), edge_label)
            loss_edge = loss_edge * (edge_label + 0.1)
            # print(loss_edge.shape)
            loss_edge = loss_edge.mean()
            # print(loss_edge.shape)
            loss += loss_edge
            loss_dict["loss_edge"] = loss_edge.detach().cpu()
        # print("after edge loss")

        # print(loss, loss_dict)

        if return_outputs:
            return loss, loss_dict, ret_dict
        return loss, loss_dict

    def common_stage_mask(self, pixel_values, embed, mask, cal_auroc=False, box=None):
        pixel_values = pixel_values.to(self.device).float()
        embed = embed.to(self.device).float()
        mask = mask.to(self.device).float()
        outputs = self(pixel_values=pixel_values, embed=embed, box=box)
        loss = self.box_loss(
            outputs,
            mask.float(),
            box if box is not None else torch.zeros((1, 4)).to(self.device),
        )
        # loss = sigmoid_focal_loss(outputs, mask.float(), alpha=self.mask_alpha)
        # # print("after focal loss")

        # loss += 0.5 * (
        #     torch.mean(torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:]))
        #     + torch.mean(torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :]))
        # )

        if cal_auroc:
            auroc = self.mask_auroc(
                outputs.detach().view(-1), mask.detach().view(-1)
            ).cpu()
            self.mask_auroc.reset()
            torch.cuda.empty_cache()
            return loss, {"loss": loss.detach().cpu(), "mask_auroc": auroc}
        # loss = F.binary_cross_entropy(outputs, mask.float())
        return loss, {"loss": loss.detach().cpu()}

    def on_validation_epoch_end(self):
        # loss = torch.stack(self.val_step_outputs).mean()
        losses = np.mean([i["loss"] for i in self.val_step_outputs])
        if "mask" in self.stage or "stage 1 + 2 + 3" in self.stage:
            # auroc = self.mask_auroc.compute()
            auroc = np.nanmean(
                [i["mask_auroc"] for i in self.val_step_outputs if "mask_auroc" in i]
            )
            self.log("total_validate_mask_auroc", auroc, prog_bar=True)
            self.mask_auroc.reset()
        if "stage 2" in self.stage or "stage 1 + 2" in self.stage:
            # auroc = self.auroc.compute()
            auroc = np.mean([i["auroc"] for i in self.val_step_outputs])
            self.log("total_validate_auroc", auroc, prog_bar=True)
            self.auroc.reset()
            loss_boxes = np.mean(
                [i["loss_boxes"] for i in self.val_step_outputs if "loss_boxes" in i]
            )
            self.log("total_validate_loss_boxes", loss_boxes, prog_bar=True)
        if "loss_ce" in self.val_step_outputs[0]:
            ce_loss = np.mean(
                [i["loss_ce"] for i in self.val_step_outputs if "loss_ce" in i]
            )
            self.log("total_validate_loss_ce", ce_loss, prog_bar=True)
        if "loss_bbox" in self.val_step_outputs[0]:
            bbox_loss = np.mean(
                [i["loss_bbox"] for i in self.val_step_outputs if "loss_bbox" in i]
            )
            self.log("total_validate_loss_bbox", bbox_loss, prog_bar=True)
        self.log("total_validate_loss", losses, prog_bar=True)
        self.val_step_outputs.clear()

    def on_train_epoch_end(self):
        # loss = torch.stack(self.val_step_outputs).mean()
        losses = np.mean([i["loss"] for i in self.training_step_outputs])
        if "mask" in self.stage or "stage 1 + 2 + 3" in self.stage:
            auroc = np.nanmean(
                [
                    i["mask_auroc"]
                    for i in self.training_step_outputs
                    if "mask_auroc" in i
                ]
            )
            self.log("total_train_mask_auroc", auroc, prog_bar=True)
            self.mask_auroc.reset()

        if "stage 2" in self.stage or "stage 1 + 2" in self.stage:
            # auroc = self.auroc.compute()
            auroc = np.mean([i["auroc"] for i in self.training_step_outputs])
            self.log("total_train_auroc", auroc, prog_bar=True)
            self.auroc.reset()
            loss_boxes = np.mean(
                [
                    i["loss_boxes"]
                    for i in self.training_step_outputs
                    if "loss_boxes" in i
                ]
            )
            self.log("total_train_loss_boxes", loss_boxes, prog_bar=True)
        self.log("total_train_loss", losses, prog_bar=True)
        self.training_step_outputs.clear()

    def _common_step(self, batch):
        """Unified training / validation step dispatcher.

        Depending on current `self.stage`, executes one or more of:
          stage 1 (DETR), stage 2 (GNN), stage mask (segmentation).
        Composite stages chain these together while temporarily overriding
        `self.stage` to re-use sub-step methods, then restore it.
        """
        if "stage 1 + 2 + 3" in self.stage:
            temp = self.stage
            n, _, _, _ = batch[0]["pixel_values"].shape
            t = EmptyContextManager
            if self.lr_detr < 1e-6:
                t = torch.no_grad
            with t():
                loss, loss_dict, output = self.common_step_stage1(batch[0], True)
                retdict = utils.process(
                    output,
                    batch[0]["labels"],
                    need_mask=True,
                    empty=self.output_dim - 1,
                )
                data2 = utils.convertStage2Dataset(
                    retdict, num_classes=self.output_dim - 1, obj_thres=0.2
                )
                self.stage = "stage 2"
                x = data2.x
                y = data2.y
                edge_index = data2.edge_index
                mask = torch.ones_like(y, dtype=torch.bool)
                boxes = data2.boxes
                box_masks = data2.box_masks
                edge_type = data2.inter_edges if hasattr(data2, "inter_edges") else None
                edge_label = (
                    (data2.edge_label) if hasattr(batch, "edge_label") else None
                )
                loss2, loss_dict2, outputs = self.common_step_stage2(
                    x,
                    edge_index,
                    mask,
                    y,
                    boxes,
                    box_masks > -1,
                    edge_label,
                    return_outputs=True,
                    edge_type=edge_type,
                )
                loss += loss2
                self.stage = "stage mask"

                if "mask_input" in batch[0]["labels"][n // 2]:
                    img = batch[0]["labels"][n // 2]["mask_input"]
                else:
                    img = batch[0]["pixel_values"][n // 2]

                embeds = outputs["embeddings"]
                objects, _ = embeds.shape
                obj_per_image = objects // n
                sub_embeds = embeds[
                    (n // 2) * obj_per_image : (n // 2 + 1) * obj_per_image
                ]
                sub_box_masks = box_masks[
                    (n // 2) * obj_per_image : (n // 2 + 1) * obj_per_image
                ]
                pick_from = torch.where((sub_box_masks > -1))[0]
                boxes = outputs["box"]

            if len(pick_from) > 0:
                if len(pick_from) <= self.num:
                    stage_2_embeds = sub_embeds[pick_from]
                    box = boxes[pick_from]
                    masks = retdict["masks"]
                else:
                    tensor = torch.arange(len(pick_from))
                    indices = torch.randperm(tensor.size(0))[: self.num]
                    selected = pick_from[indices]
                    stage_2_embeds = sub_embeds[selected]
                    box = boxes[selected]
                    masks = retdict["masks"][indices]

                masks = masks.squeeze(1)
                num_masks = masks.sum(axis=[1, 2])
                num_masks = num_masks > 0
                if num_masks.any():
                    masks = masks[num_masks].to(self.device)
                    num_masks = num_masks.to(self.device)
                    stage_2_embeds = stage_2_embeds[num_masks]
                    box = box[num_masks]
                    img = img.repeat(stage_2_embeds.shape[0], 1, 1, 1)
                    loss3, loss_dict3 = self.common_stage_mask(
                        img, stage_2_embeds, masks, True, box
                    )
                    if self.lr_detr < 1e-6:
                        loss = loss3
                    else:
                        loss += loss3
                    loss_dict2["mask_auroc"] = loss_dict3["mask_auroc"]
                    loss_dict2["loss"] = loss.detach().cpu()
            else:
                loss_dict2["mask_auroc"] = np.nan

            self.stage = temp
            loss_dict = loss_dict2
        elif "stage 1 mask" in self.stage:
            self.stage = "stage 1"
            t = EmptyContextManager
            if self.lr_detr < 1e-6:
                t = torch.no_grad
            with t():
                loss, loss_dict, output = self.common_step_stage1(batch[0], True)
                if "mask_input" in batch[0]["labels"][0]:
                    img = batch[0]["labels"][0]["mask_input"]
                else:
                    img = batch[0]["pixel_values"][0]
                retdict = utils.process_stage1(output, batch[0]["labels"])
                masks = retdict["masks"]
                boxes = retdict["pred_boxes"]
                embeds = retdict["feature"]
                obj_pos = retdict["obj_pos"]
                label = retdict["label"].cpu().numpy()
                weights = self.class_weights[label]
                picked = utils.unique_random_sample_indices(weights, self.num)
                embeds = embeds[picked]
                boxes = boxes[picked]
                obj_pos = obj_pos[picked]
                masks = masks[obj_pos]
                self.stage = "stage mask"
                if masks.dim() == 4:
                    masks = masks.squeeze(1)
                num_masks = masks.sum(axis=[1, 2]) > 0
            if num_masks.any():
                masks = masks[num_masks]
                embeds = embeds[num_masks]
                boxes = boxes[num_masks]
                img = img.repeat(embeds.shape[0], 1, 1, 1)
                loss2, loss_dict2 = self.common_stage_mask(
                    img, embeds, masks, True, boxes
                )
                loss = loss2 if self.lr_detr < 1e-6 else loss + loss2
                if self.lr_detr < 1e-6:
                    loss_dict = loss_dict2
                else:
                    loss_dict["mask_auroc"] = loss_dict2["mask_auroc"]
            else:
                print(
                    "encountered empty masks. This could be caused by data augmentation"
                )
            self.stage = "stage 1 mask"
        elif "stage 1 + 2" in self.stage:
            t = EmptyContextManager
            if self.lr_detr < 1e-8:
                t = torch.no_grad
            with t():
                loss, loss_dict, output = self.common_step_stage1(batch[0], True)
                retdict = utils.process(
                    output, batch[0]["labels"], empty=self.output_dim - 1, nms=self.nms
                )
                data2 = utils.convertStage2Dataset(
                    retdict, obj_thres=0.15, num_classes=self.output_dim - 1
                )
            self.stage = "stage 2"
            x = data2.x
            y = data2.y
            edge_index = data2.edge_index
            mask = torch.ones_like(y, dtype=torch.bool)
            boxes = data2.boxes if hasattr(data2, "boxes") else None
            box_masks = data2.box_masks if hasattr(data2, "box_masks") else None
            edge_type = data2.inter_edges if hasattr(data2, "inter_edges") else None
            edge_label = None
            loss2, loss_dict2 = self.common_step_stage2(
                x,
                edge_index,
                mask,
                y,
                boxes,
                box_masks > -1,
                edge_label,
                edge_type=edge_type,
                return_outputs=False,
            )
            self.stage = "stage 1 + 2"
            loss = loss + loss2
            loss_dict = loss_dict2
        elif "stage 1" in self.stage:
            loss, loss_dict = self.common_step_stage1(batch[0])
        elif "stage 2" in self.stage:
            mask = batch.train_mask
            y = batch.y
            boxes = batch.boxes if hasattr(batch, "boxes") else None
            box_masks = batch.box_masks if hasattr(batch, "box_masks") else None
            edge_label = batch.edge_label if hasattr(batch, "edge_label") else None
            edge_mask = (
                batch.train_edge_mask if hasattr(batch, "train_edge_mask") else None
            )
            edge_type = batch.inter_edges if hasattr(batch, "inter_edges") else None
            loss, loss_dict = self.common_step_stage2(
                batch.x,
                batch.edge_index,
                mask,
                y,
                boxes,
                box_masks,
                edge_label,
                edge_mask,
                edge_type=edge_type,
            )
        elif "stage mask" in self.stage:
            pixel_values, stage_2_embeds, pixel_mask, box = batch
            loss, loss_dict = self.common_stage_mask(
                pixel_values, stage_2_embeds, pixel_mask, True, box=box
            )
        elif "stage pretrain mask" in self.stage:
            temp = self.stage
            self.stage = "stage mask"
            inputs = batch[0]["labels"][0]["mask_input"]
            boxes = batch[0]["labels"][0]["boxes"]
            b, _ = boxes.shape
            stage_2_embeds = torch.zeros((b, 256)).to(self.device)
            stage_2_embeds.requires_grad_(False)
            pick_from = torch.arange(b).to(self.device)
            pick_from.requires_grad_(False)
            masks = batch[0]["labels"][0]["masks"]
            if len(pick_from) > 0:
                if len(pick_from) > self.num:
                    indices = torch.randperm(pick_from.size(0))[: self.num]
                    stage_2_embeds = stage_2_embeds[indices]
                    boxes = boxes[indices]
                    masks = masks[indices]
                num_masks = masks.sum(axis=[1, 2, 3]) > 0
                if num_masks.any():
                    masks = masks[num_masks].to(self.device)
                    num_masks = num_masks.to(self.device)
                    stage_2_embeds = stage_2_embeds[num_masks]
                    boxes = boxes[num_masks]
                    img = inputs.repeat(stage_2_embeds.shape[0], 1, 1, 1)
                    loss3, loss_dict3 = self.common_stage_mask(
                        img, stage_2_embeds, masks, True, boxes
                    )
                    loss = loss3
                    loss_dict = loss_dict3
            else:
                loss = 0.0
                loss_dict = {"mask_auroc": np.nan}
            self.stage = temp
        return loss, loss_dict

    def training_step(self, batch, batch_idx=0, loader_idx=0):

        loss, loss_dict = self._common_step(batch)
        loss_dict["loss"] = loss.detach().cpu()
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss, prog_bar=True)
        # for k, v in loss_dict.items():
        #     self.log("train_" + k, v.item(), prog_bar=False)
        self.training_step_outputs.append(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx=0, loader_idx=0):
        # print(batch)
        loss, loss_dict = self._common_step(batch)
        res = {}
        res["loss"] = loss.detach().cpu()
        for k, v in loss_dict.items():
            res[k] = v.detach().cpu()
        self.val_step_outputs.append(res)
        self.log("validation_loss", res["loss"], prog_bar=True)
        # for k, v in loss_dict.items():
        #     self.log("validate_" + k, v.item(), prog_bar=False)

        return loss

    def configure_optimizers(self):
        """Create optimizer(s) with per-stage parameter grouping.

        Different stages freeze / unfreeze subsets (backbone, transformer, GNN,
        mask head) and may assign distinct LRs. Two schedulers can be attached:
        a warmup (custom) and an optional StepLR when `scheduler_step>0`.
        """
        optim = None
        if "stage 1 + 2 + 3 mask" in self.stage:
            d1 = []
            d2 = []
            for n, p in self.named_parameters():
                if "mask_head" in n and p.requires_grad:
                    d1.append(p)
                elif p.requires_grad:
                    d2.append(p)

            param_dicts = [
                {
                    "params": d1,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
            ]
            if self.lr_detr > 0.0000001:
                param_dicts.append(
                    {
                        "params": d2,
                        "lr": self.lr_detr,
                        "weight_decay": self.weight_decay * 0.01,
                    },
                )
            optim = torch.optim.AdamW(param_dicts)

        elif "stage 1 mask" in self.stage:
            # if self.lr_backbone is not None:

            d1 = []
            d2 = []
            d3 = []
            for n, p in self.named_parameters():
                if "backbone" in n and p.requires_grad:
                    d1.append(p)
                elif ".model" in n:
                    d2.append(p)
                elif "mask_head" in n:
                    d3.append(p)
            # self.lr_backbone = self.lr
            param_dicts = []
            if self.lr_backbone > 1e-6:
                param_dicts.append({"params": d1, "lr": self.lr_backbone})
            if self.lr_detr > 1e-6:
                param_dicts.append(
                    {
                        "params": d2,
                        "lr": self.lr_detr,
                    }
                )
            if self.lr > 1e-6:
                param_dicts.append(
                    {
                        "params": d3,
                        "lr": self.lr,
                    }
                )
            if self.weight_decay > 0:
                optim = torch.optim.AdamW(param_dicts, weight_decay=self.weight_decay)
            else:
                optim = torch.optim.Adam(param_dicts)

        elif "stage 1 + 2" in self.stage:
            # if self.lr_backbone is not None:

            d1 = []
            d2 = []
            d3 = []
            for n, p in self.named_parameters():
                if "backbone" in n and p.requires_grad:
                    d1.append(p)
                elif ".model" in n:
                    d2.append(p)
                elif "gnn" in n:
                    d3.append(p)
            # self.lr_backbone = self.lr
            param_dicts = []
            if self.lr_backbone > 1e-6:
                param_dicts.append({"params": d1, "lr": self.lr_backbone})
            if self.lr_detr > 1e-6:
                param_dicts.append(
                    {
                        "params": d2,
                        "lr": self.lr_detr,
                    }
                )
            if self.lr > 1e-6:
                param_dicts.append(
                    {
                        "params": d3,
                        "lr": self.lr,
                    }
                )
            if self.weight_decay > 0:
                optim = torch.optim.AdamW(param_dicts, weight_decay=self.weight_decay)
            else:
                optim = torch.optim.Adam(param_dicts)

        elif "stage 1" in self.stage:
            if self.lr_backbone is not None:

                d1 = []
                d2 = []
                d3 = []
                for n, p in self.named_parameters():
                    if "backbone" in n and p.requires_grad:
                        d1.append(p)
                    elif ".model" in n:
                        d2.append(p)
                    else:
                        d3.append(p)
                # self.lr_backbone = self.lr
                param_dicts = [
                    {"params": d1, "lr": self.lr_backbone},
                    {
                        "params": d2,
                        "lr": self.lr_detr,
                    },
                    {
                        "params": d3,
                        "lr": self.lr,
                    },
                ]
                if self.weight_decay > 0:
                    optim = torch.optim.AdamW(
                        param_dicts, weight_decay=self.weight_decay
                    )
                else:
                    optim = torch.optim.Adam(param_dicts)

        elif "stage 2" in self.stage:
            parameters = []
            for n, p in self.named_parameters():
                if ".model" in n or "mask_head" in n:
                    p.requires_grad = False
                else:
                    parameters.append(p)

            if self.weight_decay > 0:
                optim = torch.optim.AdamW(
                    parameters, lr=self.lr, weight_decay=self.weight_decay
                )
            else:
                optim = torch.optim.Adam(parameters, lr=self.lr)
        elif "stage mask" in self.stage:
            print("fixing all parameters except mask head")
            parameters = []
            for n, p in self.named_parameters():
                if not "mask_head" in n:
                    p.requires_grad = False
                else:
                    parameters.append(p)

            if self.weight_decay > 0:
                optim = torch.optim.AdamW(
                    parameters, lr=self.lr, weight_decay=self.weight_decay
                )
            else:
                optim = torch.optim.Adam(parameters, lr=self.lr)
        if optim is None:
            if self.weight_decay > 0:
                optim = torch.optim.AdamW(
                    self.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
            else:
                optim = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler_step > 0:
            return [optim], [
                {
                    "scheduler": WarmupScheduler(optim, self.warmup_epoches),
                    "interval": "epoch",
                    "frequency": 1,
                },
                {
                    "scheduler": torch.optim.lr_scheduler.StepLR(
                        optim, step_size=1, gamma=0.5
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                },
            ]
        else:
            return optim


class SimpleLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge):
        return self.linear(x)


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, dtype=torch.float):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).to(dtype))
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank, dtype=dtype) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim, dtype=dtype) * std_dev)
        # self.C = torch.nn.Parameter(torch.zeros(out_dim, dtype=dtype) * std_dev)
        # self.alpha = alpha

    def forward(self, x):
        x = x @ self.A @ self.B  # + self.C
        # print(x.shape)
        # x = x + self.C
        return x


class GCN(torch.nn.Module):
    """
    GNN module
    This module takes in all queries and output the final GNN features
    classification, box regression are integrated in this module, other prediction heads can be added


    Args:
        input_dim (int): dimension of the input features.
        additional_input_dim (int): dimension of additional input features. (such as predictions from stage 1)
        output_classes (int): number of output classes.
        layer_type (str, optional): type of GNN layer to use. Defaults to "GCNConv".
        dropout (bool, optional): whether to use dropout in GNN layers. Defaults to False.
        zpos (int, optional): positional encoding multiplier. Defaults to 50.
    """

    def __init__(
        self,
        input_dim,
        additional_input_dim,
        output_classes,
        layer_type="TransformerConv",
        dropout=False,
        zpos=500,
        box_head="lora",
        record=False,
    ):

        super().__init__()
        self.output_classes = output_classes
        self.additional_input_dim = additional_input_dim
        self.additional_input_layer = AdditionalInputLayer(
            additional_input_dim, input_dim
        )
        if layer_type == "GCNConv":
            Layer = GCNConv
        elif layer_type == "SAGEConv":
            Layer = SAGEConv
        elif layer_type == "GATConv":
            Layer = GATConv
        elif layer_type == "AGNNConv":
            Layer = AGNNConv
        elif layer_type == "GATv2Conv":
            Layer = GATv2Conv
        elif layer_type == "TransformerConv":
            Layer = TransformerConv
        elif layer_type == "SimpleLinear":
            Layer = SimpleLinear
        else:
            raise ValueError("Invalid layer type")

        pe = self.inipos(input_dim)
        pe.requires_grad = False
        self.register_buffer("pe", pe)

        self.conv1 = Layer(input_dim, input_dim)
        self.conv2 = Layer(input_dim, input_dim)
        self.conv3 = Layer(input_dim, input_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_classes),
        )
        # self.cls_head = nn.Sequential(
        #     nn.Linear(input_dim, input_dim // 2),
        #     nn.ReLU(),
        #     LoRALayer(input_dim // 2, output_classes, 4),
        # )
        self.box = box_head
        if box_head == "lora":
            self.box_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                LoRALayer(input_dim // 2, 4, 4),
                nn.Tanh(),
            )
        else:
            self.box_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 4),
                nn.Sigmoid(),
            )

        self.dropout = dropout

        self.edge_head = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.zpos = zpos
        if record:
            self.record = []
        else:
            self.record = None

    def inipos(self, channels):
        inv_freq = 1.0 / (
            (100 * 10) ** (torch.arange(0, channels, 2).float() / channels)
        )  # .to(self.device)
        t = torch.arange(0, 505)[:, None]  # .to(self.device)
        # print(t.shape, inv_freq.shape)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        # print(pos_enc_a.shape)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=1)
        return pos_enc

    def forward(self, inputs, edge_index):

        additional_input, model_feature = (
            inputs[:, : self.additional_input_dim],
            inputs[:, self.additional_input_dim :],
        )

        pos = (additional_input[:, 0] * self.zpos).long()
        pos_embed = self.pe[pos]

        # additional_feature = self.additional_input_layer(additional_input)
        # x = model_feature + additional_feature + pos_embed

        feature = self.additional_input_layer(inputs)
        x = feature + pos_embed
        # x = torch.cat([pos_embed, feature], dim=1)
        # if self.dropout:
        #     model_feature = F.dropout(model_feature, p=0.2)
        # x = torch.cat([model_feature, additional_feature], dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=0.2)
        x = self.conv2(x, edge_index)

        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=0.2)
        x = self.conv3(x, edge_index)

        # if self.dropout:
        #     x = F.dropout(x, p=0.2)

        predict = self.cls_head(x)
        # predict[:, : self.output_classes - 1] += inputs[:, 5 : self.output_classes + 4]

        if self.box == "lora":
            box = self.box_head(x) + inputs[:, 1:5]
        else:
            box = self.box_head(x)

        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=1)
        edge = self.edge_head(edge_embeddings)

        return {"predict": predict, "box": box, "edge": edge, "embeddings": x}


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return F.gelu(self.double_conv(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, pool_kernal=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_kernal),
            # DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.Linear(emb_dim, out_channels),
            nn.SiLU(),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, size, emb_dim=None):
        super().__init__()

        self.up = nn.Upsample(size=size, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            # DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels),
        )
        if emb_dim is not None:
            self.emb_layer = nn.Sequential(
                nn.Linear(emb_dim, out_channels),
                nn.SiLU(),
            )

    def forward(self, x, skip_x, t=None):
        x = self.up(x)
        # print("up", x.shape, skip_x.shape)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        if t is None:
            return x

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class VitForMask(nn.Module):
    def __init__(self, c_in=3, c_out=1, embed_dim=272, sigmoid=True, TF=0.5):
        super().__init__()
        # self.ini = DoubleConv()

        # 16 * 800 * 800
        self.inc = DoubleConv(c_in + 1, 16)

        # 16 * 400 * 400
        self.down1 = Down(16, 32, embed_dim)

        # 64 * 200 * 200
        self.down2 = Down(32, 64, embed_dim)

        # 128 * 100 * 100
        self.down3 = Down(64, 128, embed_dim)

        # 256 * 50 * 50
        self.down4 = Down(128, 512, embed_dim)

        # 512 * 25 * 25
        # self.down5 = Down(256, 512, embed_dim)
        self.l = nn.Sequential(nn.Linear(embed_dim, 512), nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(1, 2500 + 1, 512))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="gelu",
            ),
            num_layers=6,
        )

        # 128 * 100 * 100
        self.up3 = Up(512 + 128, 64, (100, 100), embed_dim)

        # 64 * 200 * 200
        self.up4 = Up(128, 32, (200, 200), embed_dim)

        # 32 * 400 * 400
        self.up5 = Up(64, 16, (400, 400), embed_dim)

        # 32 * 800 * 800
        self.up6 = Up(32, 32, (800, 800), embed_dim)

        # 1 * 800 * 800
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

        self.sigmoid = sigmoid

        self.TF = TF

    def forward(self, x, t, boxes):
        B, C, H, W = x.shape
        mask = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)
        if self.TF > 0:
            if torch.rand(1) < self.TF:
                # return torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)
                b1 = boxes[:, 0] - boxes[:, 2] / 2
                b1 = torch.clamp(b1, min=0.0)
                b2 = boxes[:, 0] + boxes[:, 2] / 2
                b2 = torch.clamp(b2, max=1.0)
                b3 = boxes[:, 1] - boxes[:, 3] / 2
                b3 = torch.clamp(b3, min=0.0)
                b4 = boxes[:, 1] + boxes[:, 3] / 2
                b4 = torch.clamp(b4, max=1.0)
                b1, b2, b3, b4 = (
                    (b1 * W).long(),
                    (b2 * W).long(),
                    (b3 * H).long(),
                    (b4 * H).long(),
                )
                for i in range(B):
                    mask[i, 0, b3[i] : b4[i], b1[i] : b2[i]] = 1.0

        mask.requires_grad_(False)

        x = torch.cat((x, mask), dim=1)

        # if self.record is not None:
        #     self.record.append(x.detach().cpu().numpy())

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)

        x5 = x5.view(-1, 512, 2500).transpose(1, 2)
        l = self.l(t).unsqueeze(1)
        x = torch.cat((l, x5), dim=1)
        x += self.pos_embed
        # x5 = x5.transpose()
        x = self.transformer(x)
        x = x[:, 1:, :]
        x = x.transpose(1, 2).view(-1, 512, 50, 50)

        x = self.up3(x, x4, t)
        x = self.up4(x, x3, t)
        x = self.up5(x, x2, t)
        x = self.up6(x, x1, t)
        output = self.outc(x).squeeze(1)

        if self.sigmoid:
            output = torch.sigmoid(output)

        return output


class ParticleID3DNet_Binary(L.LightningModule):

    def __init__(self, num_classes=1, regression=False, small=False):
        super().__init__()
        self.save_hyperparameters()
        
        # 3D convolutional backbone
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv1_1 = nn.Conv3d(16, 16, kernel_size=5, padding=2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv2_1 = nn.Conv3d(32, 32, kernel_size=5, padding=2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv3_1 = nn.Conv3d(64, 64, kernel_size=5, padding=2)

        self.pool = nn.MaxPool3d(2)
        
        if not small:
            # After 3 poolings: 65 -> 32 -> 16 -> 8
            self.fc1 = nn.Linear(64 * 8 * 8 * 8, 128)
        else:
            
            self.fc1 = nn.Linear(64 * 5 * 5 * 5, 128)

        self.num_classes = num_classes
        self.regression = regression
        if regression:
            self.fc2 = nn.Linear(128, num_classes)  # output dim = regression dim
            self.criterion = nn.MSELoss()
        elif num_classes == 1:
            self.fc2 = nn.Linear(128, 1)
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.fc2 = nn.Linear(128, num_classes)
            self.criterion = nn.CrossEntropyLoss()

        self.training_step_outputs = []
        self.val_step_outputs = []

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.conv1_1(x))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.conv2_1(x))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.conv3_1(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        if self.regression:
            return out  # shape (B, num_regression)
        elif self.num_classes == 1:
            return out.squeeze(1)  # (B,)
        else:
            return out  # (B, num_classes)

    def shared_step(self, batch, stage, l):
        x, y = batch
        logits = self(x)
        if self.regression:
            # y shape: (B, num_regression)
            loss = self.criterion(logits, y)
            l.append({"y": y.cpu().numpy(), "pre": logits.detach().cpu().numpy()})
            return loss
        elif self.num_classes == 1:
            # Binary classification
            loss = self.criterion(logits, y.float())
            preds = torch.sigmoid(logits)
            predicted_classes = (preds > 0.5).float()
            acc = (predicted_classes == y).float().mean()
            l.append({"y": y.cpu().numpy(), "pre": preds.detach().cpu().numpy()})
            return loss
        else:
            # Multiclass classification
            loss = self.criterion(logits, y.long())
            preds = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(preds, dim=1)
            acc = (predicted_classes == y).float().mean()
            l.append({"y": y.cpu().numpy(), "pre": preds.detach().cpu().numpy()})
            return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train", self.training_step_outputs)

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val", self.val_step_outputs)

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        preds = np.concatenate([x["pre"] for x in self.val_step_outputs], axis=0)
        trues = np.concatenate([x["y"] for x in self.val_step_outputs], axis=0)
        if self.regression:
            # For regression, print MSE
            mse = np.mean((preds - trues) ** 2)
            print("val_mse", mse)
        elif self.num_classes == 1:
            from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

            precision, recall, thresholds = precision_recall_curve(trues, preds)
            aupr = auc(recall, precision)
            auroc = roc_auc_score(trues, preds)
            print("val_aupr", aupr, "val_auroc", auroc)
            print("logging")
            self.log_dict({"val_auroc":auroc, "val_aupr": aupr})
            # self.log("val_auroc", auroc)
            # self.log("val_aupr", aupr)
        else:
            from sklearn.metrics import accuracy_score, log_loss

            acc = accuracy_score(trues, np.argmax(preds, axis=1))
            ce = log_loss(trues, preds)
            print("val_acc", acc, "val_ce", ce)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

class Particle3DNet(L.LightningModule):
    """Residual 3D CNN for multiclass particle identification in tomograms.

    Design:
    - 6 Conv3D layers total (implemented as 3 residual blocks, each block has 2 convs).
    - Spatial downsampling by factor 2 after every residual block.
    - Supports 65^3 and 41^3 inputs in one implementation via adaptive pooling.
    """

    def __init__(
        self,
        num_classes=13,
        input_size=65,
        channels=(32, 64, 128),
        dropout=0.2,
        lr=2e-4,
        weight_decay=1e-4,
        class_weights=None,
        label_smoothing=0.05,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        if input_size not in (41, 65):
            raise ValueError("input_size must be either 41 or 65")
        if len(channels) != 3:
            raise ValueError("channels must provide 3 stages, e.g. (32, 64, 128)")

        c1, c2, c3 = channels
        self.num_classes = num_classes
        self.input_size = input_size
        self.lr = lr
        self.weight_decay = weight_decay

        # 6 conv layers in total: 2 convs per block x 3 blocks.
        self.block1 = Residual3DBlock(1, c1, dropout=dropout)
        self.block2 = Residual3DBlock(c1, c2, dropout=dropout)
        self.block3 = Residual3DBlock(c2, c3, dropout=dropout)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, c3 // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(c3 // 2, num_classes),
        )

        cw = None
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32)
            if cw.numel() != num_classes:
                raise ValueError("class_weights length must match num_classes")

        self.register_buffer(
            "class_weights_buffer",
            cw if cw is not None else torch.ones(num_classes, dtype=torch.float32),
            persistent=True,
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights_buffer,
            label_smoothing=label_smoothing,
        )

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError("Expected input of shape (B, C, D, H, W)")
        if x.shape[1] != 1:
            raise ValueError("Particle3DNet expects single-channel input (C=1)")

        s = x.shape[-1]
        if s not in (41, 65):
            raise ValueError("Particle3DNet supports cubic inputs of 41 or 65")

        x = self.block1(x)
        x = self.pool(x)

        x = self.block2(x)
        x = self.pool(x)

        x = self.block3(x)
        x = self.pool(x)

        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def shared_step(self, batch, stage="train"):
        x, y = batch
        logits = self(x)
        y = y.long()
        loss = self.criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        if stage == "train":
            acc = self.train_acc(probs, y)
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        elif stage == "val":
            acc = self.val_acc(probs, y)
            f1 = self.val_f1(probs, y)
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=self.lr * 0.05,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
