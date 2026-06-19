import json
import os
import pickle
import random
import re
from typing import List

import mrcfile
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as transforms
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torchvision.tv_tensors import BoundingBoxes, Mask
from tqdm import tqdm
from transformers import (
    ConditionalDetrConfig,
    ConditionalDetrForObjectDetection,
    ConditionalDetrForSegmentation,
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    DetrConfig,
    DetrForObjectDetection,
    DetrForSegmentation,
    DetrImageProcessor,
)
from transformers.image_transforms import center_to_corners_format

import modules

# helper function for post processing and visualization
int_colors = [
    "#afba03",
    "#1f7c92",
    "#c70039",
    "#4d2e7f",
    "#03a678",
    "#f5f5f5",
    "#ff5733",
    "#000000",
]


def  drawannotation(image, target, box=True, mask=True, font_size=30, color=None):
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    if isinstance(image, torch.Tensor):
        # image = image.numpy()
        # if np.max(image) < 10:
        image -= torch.min(image)
        image /= torch.max(image)
        # image = torch.tensor(image * 255).type(torch.uint8)
        image = (image * 255).type(torch.uint8)

    if "masks" in target and mask:
        print("masks")
        if target["masks"].dim() > 3:
            target["masks"] = target["masks"].squeeze()
        if target["masks"].dtype != torch.bool:
            target["masks"] = target["masks"] > 0.5
        annotated_tensor = draw_segmentation_masks(
            image=image,
            masks=target["masks"],
            alpha=0.5,
            colors=[int_colors[i] for i in target["class_labels"]],
        )
    else:
        annotated_tensor = image
        annotated_tensor = torch.tensor(annotated_tensor, dtype=torch.uint8)
    # Annotate the sample image with labels and bounding boxes
    # if "names" in target:
    if box:
        if color is not None:
            color = [int_colors[color]] * len(target["bboxes"])
        b = target["bboxes"]
        b[:, 2:] += 5
        b[:, :2] -= 5
        annotated_tensor = draw_bounding_boxes(
            image=annotated_tensor,
            boxes=b,  # target["bboxes"],
            # labels=target["names"] if "names" in target else target["labels"],
            # font_size=font_size,
            width=5,
            colors="blue",
            # colors=[int_colors[i] for i in target["class_labels"]],
        )

    res = annotated_tensor.numpy()
    plt.imshow(np.moveaxis(res, 0, -1))
    for i in range(len(target["bboxes"])):
        box = target["bboxes"][i]
        if "names" in target:
            name = target["names"][i]
        else:
            name = str(target["labels"][i])
        plt.text(
            box[0] + 10,
            box[1] + 10,
            name,
            color="red",
            fontsize=font_size,
            # bbox=dict(facecolor="red", alpha=0.5),
        )


def mask_iou(mask1, mask2):
    """
    Compute IoU between two binary masks.
    """
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()
    return intersection / union


def donms(threshold, masks, scores):
    s = torch.zeros_like(scores, dtype=torch.float)
    seq = torch.argsort(scores, descending=True)
    keep = []
    for i in range(len(scores)):
        if s[seq[i]] < threshold:
            keep.append(seq[i])

        for j in range(i + 1, len(scores)):
            s[seq[j]] = max(s[seq[j]], mask_iou(masks[seq[i]], masks[seq[j]]))
    return torch.tensor(keep)


def nms(threshold, masks, scores, classids):
    allkeep = []
    for i in np.unique(classids):
        keep = donms(threshold, masks[classids == i], scores[classids == i])
        ids = torch.where(classids == i)[0]
        allkeep.append(ids[keep])
    keep = torch.cat(allkeep)
    return keep


def bbnms(threshold, boxes, scores, classids):
    allkeep = []
    # print(boxes.shape, scores.shape, classids.shape)
    for i in torch.unique(classids):
        # print(i)
        keep = torchvision.ops.nms(
            boxes[classids == i], scores[classids == i], threshold
        )
        if isinstance(classids, torch.Tensor):
            ids = torch.where(classids == i)[0]
        else:
            ids = np.where(classids == i)[0]
            ids = torch.tensor(ids)
        if ids.dim() < 1:
            ids = ids.unsqueeze(0)
        # if len(keep) >= 1:
        # allkeep.append(torch.tensor([]))
        allkeep.append(ids[keep])
    # print(allkeep)
    keep = torch.cat(allkeep)
    return keep


def convertBoxes(boxes):
    center_x, center_y, width, height = boxes.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [
            (center_x - 0.5 * width),
            (center_y - 0.5 * height),
            (center_x + 0.5 * width),
            (center_y + 0.5 * height),
        ],
        dim=-1,
    )
    bbox_corners[bbox_corners < 0] = 0
    bbox_corners[bbox_corners > 1] = 1
    return bbox_corners


def to_tuple(tup):
    if isinstance(tup, tuple):
        return tup
    return tuple(tup.cpu().long().tolist())


# inspired by image_processor.post_process
def postSegmentationTreatment(
    outputs, threshold, target_sizes, mask_threshold=0.5, style="without None"
):
    if mask_threshold is not None:
        out_logits, out_bbox, masks = (
            outputs.logits,
            outputs.pred_boxes,
            outputs.pred_masks,
        )
    else:
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

    if target_sizes is not None:
        if len(out_logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    if style == "without None":
        prob = nn.functional.sigmoid(out_logits)
        scores, labels = prob.max(-1)
        is_nones = torch.zeros_like(scores)
    else:
        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., 1:].max(-1)
        is_nones = prob[..., 0]

    boxes = convertBoxes(out_bbox)
    if isinstance(target_sizes, List):
        img_h = torch.Tensor([i[0] for i in target_sizes])
        img_w = torch.Tensor([i[1] for i in target_sizes])
        target_sizes = torch.stack([img_w, img_h], dim=1).to(boxes.device)
    else:
        img_h, img_w = target_sizes.unbind(1)

    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    boxes = boxes * scale_fct[:, None, :]

    results = []

    if mask_threshold is not None:
        for s, l, b, mask, size, is_none in zip(
            scores, labels, boxes, masks, target_sizes, is_nones
        ):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            mask = mask[s > threshold]
            is_none = is_none[s > threshold]
            print(mask.shape, size)
            if np.any(mask.shape[1:2] != size):
                mask = nn.functional.interpolate(
                    mask[:, None], size=to_tuple(size), mode="bilinear"
                )
                mask = mask > mask_threshold

            results.append(
                {
                    "scores": score,
                    "labels": label,
                    "boxes": box,
                    "masks": mask,
                    "is_none": is_none,
                }
            )
    else:
        for s, l, b, size, is_nones in zip(
            scores, labels, boxes, target_sizes, is_nones
        ):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            is_none = is_nones[s > threshold]
            results.append(
                {"scores": score, "labels": label, "boxes": box, "is_none": is_none}
            )
    return results


def toTarget(result, size):
    if "masks" in result:
        return {
            "labels": result["labels"],
            "bboxes": BoundingBoxes(result["boxes"], format="xyxy", canvas_size=size),
            "masks": Mask(result["masks"].squeeze(1)),
            "names": [str(i.item()) for i in result["labels"]],
        }
    else:
        return {
            "labels": result["labels"],
            "bboxes": BoundingBoxes(result["boxes"], format="xyxy", canvas_size=size),
            "names": [str(i.item()) for i in result["labels"]],
        }


# helper functions for stage 2 processing
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrHungarianMatcher,
)

matcher = DeformableDetrHungarianMatcher(1.0, 5.0, 2.0)


def processSingle(model, label, data, target_size, thres, has_none, empty=5, nms=-1.0):
    ret = []
    boxeses = []
    box_masks = []
    l = []
    item_ids = []
    masks = []
    model.eval()
    output = model(
        pixel_values=data["pixel_values"].unsqueeze(0).float().to(model.device),
        pixel_mask=data["pixel_mask"].unsqueeze(0).float().to(model.device),
    )
    # for idx, i in enumerate(model_names):
    #     # print(i)
    #     if i!="none":
    #         _output = output[i]
    #     else:
    #         _output = output
    logits = output["logits"].squeeze(0)
    pred_boxes = output["pred_boxes"].squeeze(0)
    if nms > 0.1:
        keep = bbnms(nms, convertBoxes(pred_boxes), logits[:, :empty].max(axis=1).values, torch.argmax(logits[:, :empty], axis=1))
        pred_boxes = pred_boxes[keep]
        logits = logits[keep]
    if has_none:
        # print(logits, logits.shape)
        prob = torch.softmax(logits, -1)
        prob = prob[:, :-1]
    else:
        prob = torch.sigmoid(logits)

    if isinstance(thres, float):
        v, pos = torch.max(prob, dim=1)
        reserve = v > thres
    else:
        reserve = reserve > thres
        reserve = reserve.any(dim=1)

    if label["class_labels"] is not None and len(label["class_labels"]) == 0:
        label["class_labels"] = None

    if label["class_labels"] is not None:
        o = {}
        o["pred_boxes"] = pred_boxes.unsqueeze(0)
        o["logits"] = logits.unsqueeze(0)
        match_res = matcher(o, [label])
        match_res = match_res[0]
        r = torch.zeros_like(reserve, dtype=torch.bool)
        r[match_res[0]] = True
        reserve2 = reserve | r
        # print("check whether reserve ", sum(reserve), sum(reserve2))
        reserve = reserve2
    # else:
    #     reserve = v > thres

    # v, pos = torch.max(prob, dim=1)
    logits = logits[reserve]
    pred_boxes = pred_boxes[reserve]
    embed = output["last_hidden_state"].squeeze(0)
    embed = embed[reserve]

    pos = torch.zeros((pred_boxes.shape[0], 5))
    pos[:, 0] = label["pos"]
    # if label["pos"] > 249 and label["pos"] < 255:
    #     print(label["pos"])
    if "zposmax" in label:
        if label["zposmax"] == 0:
            label["zposmax"] = 500
        pos[:, 0] /= label["zposmax"]
    else:
        pos[:, 0] /= 500
    # print(pred_boxes)
    pos[:, 1] = pred_boxes[:, 0]  # * (label["size"][0] / target_size[0])
    pos[:, 2] = pred_boxes[:, 1]  # * (label["size"][1] / target_size[1])
    pos[:, 3] = pred_boxes[:, 2]  # * (label["size"][0] / target_size[0])
    pos[:, 4] = pred_boxes[:, 3]  # * (label["size"][1] / target_size[1])
    input = torch.concat([pos, logits, embed], dim=1)
    ret.append(input)
    targets = torch.zeros((pred_boxes.shape[0]), dtype=torch.long)
    targets.fill_(empty)
    box_mask = torch.zeros((pred_boxes.shape[0]), dtype=torch.bool)
    boxes = torch.zeros((pred_boxes.shape[0], 4))
    item_id = ["" for i in range(pred_boxes.shape[0])]

    if label["class_labels"] is not None:
        if len(label["class_labels"]) > 0:
            o = {}
            o["pred_boxes"] = pred_boxes.unsqueeze(0)
            o["logits"] = logits.unsqueeze(0)
            # print(o, label[i])
            match_res = matcher(o, [label])
            match_res = match_res[0]
            target = label["class_labels"]
            target = target[match_res[1]]
            # print(targets, target)
            target_boxes = label["boxes"]
            target_boxes = target_boxes[match_res[1]]

            if "item_id" in label:
                for s, t in zip(match_res[0], match_res[1]):
                    item_id[s] = label["item_id"][t]

            boxes[match_res[0]] = target_boxes
            box_mask[match_res[0]] = True
            # print(box_mask.shape, boxes.shape)
            targets[match_res[0]] = target

            if "masks" in label:
                masks.append(label["masks"][match_res[1]])

    l.append(targets)
    box_masks.append(box_mask)
    boxeses.append(boxes)
    item_ids.append(item_id)

    return {
        "feature": ret,
        "label": l,
        "box_mask": box_masks,
        "boxes": boxeses,
        "item_id": item_ids,
        "masks": masks,
    }
    
def pickout(dict, pos, gap, length, min_id = None, max_id = None):
    keys = np.array(sorted(dict.keys()))

    if length <= 0 or len(keys) == 0:
        return [], -1

    if pos not in dict:
        pos = keys[np.argmin(np.abs(keys - pos))].item()

    center_idx = length // 2
    start = pos - gap * center_idx
    if min_id is None:
        min_id = keys.min().item()
    if max_id is None:
        max_id = keys.max().item()

    while start < min_id:
        start += gap

    end = start + gap * (length - 1)
    if end > max_id:
        shift_steps = (end - max_id + gap - 1) // gap
        start -= shift_steps * gap
        while start < min_id:
            start += gap

    needID = []
    p = start
    for i in range(length):
        needID.append(p)
        p += gap
        if p > max_id:
            break

    if pos in needID:
        pos_idx = needID.index(pos)
    else:
        pos_idx = int(np.argmin(np.abs(np.array(needID) - pos)))

    return needID, pos_idx


def runStage1(
    model, 
    dataset,
    target_size,
    has_none=False,
    empty=5,
    seed=None,
    nms = -1.0
):
    ret = {}
    for i in tqdm(range(dataset.start_pos, dataset.end_pos)):
        if seed is not None:
            data = dataset.__getitem__(pos=i, seed=seed)
        else:
            data = dataset.__getitem__(pos=i)
        label = data["labels"]
        with torch.no_grad():
            _ret_dict = processSingle(
                model, label, data, target_size, 0.0, has_none, empty, nms
            )

        ret[i] = _ret_dict
    # print("finish dataset")
    return ret

def buildStage2(
    model,
    dataset,
    target_size,
    thres,
    offset=0,
    gap=5,
    # model_names=["other", "ribo"],
    has_none=False,
    empty=5,
    seed=None,
    nms = -1.0,
):
    ret = []
    l = []
    box_masks = []
    masks = []
    boxes = []
    item_ids = []
    images = []
    pixel_masks = []
    sample_mapping = {}
    ret_dict = {
        "feature": ret,
        "label": l,
        "masks": masks,
        "box_mask": box_masks,
        "boxes": boxes,
        "item_id": item_ids,
        "images": images,
        "pixel_masks": pixel_masks,
        "sample_mapping": sample_mapping,
    }
    cnts = 0
    for i in tqdm(range(dataset.start_pos + offset, dataset.end_pos, gap)):
        # print("slice", i)
        if seed is not None:
            data = dataset.__getitem__(pos=i, seed=seed)
        else:
            data = dataset.__getitem__(pos=i)
        label = data["labels"]
        with torch.no_grad():
            _ret_dict = processSingle(
                model, label, data, target_size, thres, has_none, empty
            )

        for j in _ret_dict:
            ret_dict[j].extend(_ret_dict[j])
        ret_dict["images"].append(data["pixel_values"])
        ret_dict["pixel_masks"].append(data["pixel_mask"])
        # print(_ret_dict["item_id"])
        for i in _ret_dict["item_id"]:
            for j in range(len(i)):
                sample_mapping[cnts] = len(ret_dict["images"]) - 1
                cnts += 1

        # print("obj_cnts:", cnts)
    print("finish dataset")
    ret_dict["feature"] = torch.cat(ret_dict["feature"], dim=0)
    ret_dict["label"] = torch.cat(ret_dict["label"], dim=0)
    ret_dict["box_mask"] = torch.cat(ret_dict["box_mask"], dim=0)
    ret_dict["boxes"] = torch.cat(ret_dict["boxes"], dim=0)
    if len(ret_dict["masks"]) > 0:
        ret_dict["masks"] = torch.cat(ret_dict["masks"], dim=0)
    # else:
    #     ret_dict["masks"] = None
    item_ids = []
    for i in ret_dict["item_id"]:
        item_ids.extend(i)
    ret_dict["item_id"] = item_ids
    return ret_dict
    # return ret, l, box_masks, boxes
    # return ret, l


def process_stage1(outputs, labels):
    logits = outputs["logits"]
    device = logits.device
    for l in range(len(labels)):
        for i in ["class_labels", "boxes", "masks"]:
            if i in labels[l]:
                labels[l][i] = labels[l][i].to(device)

    match_res = matcher(outputs, labels)

    _, num_obj, obj_num = logits.shape
    pred_boxes = outputs["pred_boxes"]
    label = labels[0]
    # print(label)
    # print(label.keys())
    embed = outputs["last_hidden_state"][0]
    targets = torch.zeros((num_obj), dtype=torch.long).to(device)
    targets.fill_(obj_num)
    targets[match_res[0][0]] = label["class_labels"][match_res[0][1]]
    box_mask = torch.zeros((num_obj), dtype=torch.long).to(device)
    box_mask.fill_(-1)
    t = torch.arange(0, len(match_res[0][0])).long().to(device)
    box_mask[match_res[0][0]] = t
    # boxes = torch.zeros((num_obj, 4)).to(device)
    # boxes[match_res[0][0]] = label["boxes"][match_res[0][1]]
    masks = label["masks"][match_res[0][1]]

    return {
        "masks": masks,
        "feature": embed,
        "label": targets,
        "pred_boxes": pred_boxes[0].detach(),
        "obj_pos": box_mask,
    }


def unique_random_sample_indices(weights, num_samples):
    """
    Randomly sample up to `num_samples` elements from a list based on weights
    without replacement, and return the indices of the selected elements.
    Elements with a weight of 0 are excluded from sampling.

    Args:
        elements (list): List of elements to sample from.
        weights (list): List of weights corresponding to each element.
        num_samples (int): The number of samples to select.

    Returns:
        list: Indices of the selected elements.
    """
    # Filter elements with weights > 0
    filtered_indices = [i for i, w in enumerate(weights) if w > 0]

    # If no elements have a valid weight, return an empty list
    if not filtered_indices:
        return []

    # If there aren't enough valid elements, return all valid indices
    if len(filtered_indices) <= num_samples:
        return filtered_indices

    # Otherwise, randomly sample indices from the valid ones
    sampled_indices = random.sample(population=filtered_indices, k=num_samples)

    return sampled_indices



def process(outputs, labels, empty=4, need_mask=False, nms = -1.0):
    ret = []
    boxeses = []
    box_masks = []
    l = []
    item_ids = []
    mask = []
    logits = outputs["logits"]
    device = logits.device
    slice_num, num_obj, _ = logits.shape
    pred_boxes = outputs["pred_boxes"]

    # match_res = matcher(outputs, labels)

    cnts = 0
    for i in range(slice_num):
        pred_boxes_i = pred_boxes[i]
        logits_i = logits[i]
        embed = outputs["last_hidden_state"][i]
        if nms > 0.1:
            # print("using nms")
            keep = bbnms(nms, convertBoxes(pred_boxes_i), logits_i[:, :empty].max(axis=1).values, torch.argmax(logits_i[:, :empty], axis=1))
            pred_boxes_i = pred_boxes_i[keep]
            logits_i = logits_i[keep]
            embed = embed[keep]
            # print(pred_boxes_i)
            # print(logits_i)
        match_res = matcher({"pred_boxes": pred_boxes_i.unsqueeze(0), "logits": logits_i.unsqueeze(0)}, [labels[i]])
        label = labels[i]
        num_obj, _ = logits_i.shape
        pos = torch.zeros((num_obj, 5)).to(device)
        pos[:, 0] = label["pos"]
        if "zposmax" in label:
            pos[:, 0] /= label["zposmax"]
        else:
            pos[:, 0] /= 500.0
        pos[:, 1:5] = pred_boxes_i[:, 0:4]

        logit = logits_i
        # print(pos.shape, logit.shape, embed.shape)
        
        input = torch.concat([pos, logit, embed], dim=1)
        ret.append(input)

        targets = torch.zeros((num_obj), dtype=torch.long).to(device)
        targets.fill_(empty)
        targets[match_res[0][0]] = label["class_labels"][match_res[0][1]]
        box_mask = torch.zeros((num_obj), dtype=torch.long).to(device)
        box_mask.fill_(-1)
        if len(match_res[0][0]) > 0:
            t = torch.arange(0, len(match_res[0][0])).long().to(device)
            box_mask[match_res[0][0]] = t + cnts
            cnts += len(match_res[0][0])
        if need_mask and "masks" in label and (i == slice_num // 2):
            mask.append(label["masks"][match_res[0][1]])
        boxes = torch.zeros((num_obj, 4)).to(device)
        boxes[match_res[0][0]] = label["boxes"][match_res[0][1]]

        item_id = ["" for i in range(num_obj)]
        if "item_id" in label:
            for s, t in zip(match_res[0][0], match_res[0][1]):
                item_id[s] = label["item_id"][t]

        boxeses.append(boxes)
        l.append(targets)
        item_ids.append(item_id)
        box_masks.append(box_mask)
    ret = torch.cat(ret, dim=0)
    l = torch.cat(l, dim=0)
    box_masks = torch.cat(box_masks, dim=0)
    boxeses = torch.cat(boxeses, dim=0)
    item_ids = [i for j in item_ids for i in j]
    if need_mask:
        # print(mask)
        mask = torch.cat(mask, dim=0)
        # print(mask.shape)
        # print(mask)
        # print(cnts, mask.shape)
        return {
            "feature": ret,
            "label": l,
            "box_mask": box_masks,
            "boxes": boxeses,
            "item_id": item_ids,
            "masks": mask,
        }
    return {
        "feature": ret,
        "label": l,
        "box_mask": box_masks,
        "boxes": boxeses,
        "item_id": item_ids,
    }


def get_iou(X):
    iou, _ = modules.box_iou(
        center_to_corners_format(torch.tensor(X)),
        center_to_corners_format(torch.tensor(X)),
    )
    iou = iou.numpy()
    return iou


# def get_neighbors(
#     X, z_thres1=0.03, z_thres2=0.5, iou_thres=0.4, num_classes=5, obj_thres=0.2
# ):
#     X2 = X[:, 1:5].clone()
#     # expand a little so that boxes may overlap
#     X2[:, 2:] += 0.05

#     iou, _ = modules.box_iou(center_to_corners_format(X2), center_to_corners_format(X2))
#     x, y = torch.where(iou > 0.01)

#     obj = X[:, 5 : 5 + num_classes] > obj_thres
#     obj = obj.any(dim=1)

#     iou -= 1 - obj.float()
#     iou = iou.T
#     iou -= 1 - obj.float()

#     zposx = X[:, 0][x]
#     zposy = X[:, 0][y]
#     x = x[torch.abs(zposx - zposy) < z_thres1]
#     y = y[torch.abs(zposx - zposy) < z_thres1]

#     x1, y1 = torch.where(iou > iou_thres)
#     zposx = X[:, 0][x1]
#     zposy = X[:, 0][y1]
#     need = (zposx != zposy) & (torch.abs(zposx - zposy) < z_thres2)
#     x1 = x1[need]
#     y1 = y1[need]
#     x = torch.cat([x, x1])
#     y = torch.cat([y, y1])
#     return x, y


def get_neighbors(
    X,
    z_thres1=0.001,
    z_thres2=0.5,
    iou_thres=0.4,
    num_classes=4,
    obj_thres=0.2,
    max_edges=None,
):
    X2 = X[:, 1:5].clone()
    # expand a little so that boxes may overlap
    # X2[:, 2:] += 0.05
    # print("calculate node iou")
    # with open("/home/feity/cryoem/temp/results/iou.pt", "wb") as f:
    #     pickle.dump(X2, f)
    iou, _ = modules.box_iou(center_to_corners_format(X2), center_to_corners_format(X2))
    # print("calculate node iou done")
    obj = X[:, 5 : 5 + num_classes]
    # print(obj)
    obj = torch.sigmoid(obj)  # .float()
    obj = obj > obj_thres
    obj = obj.any(dim=1)

    iou -= 1 - obj.float()
    iou = iou.T
    iou -= 1 - obj.float()
    if max_edges is None:
        x, y = torch.where(iou > 0.01)
    else:
        values, indices = torch.topk(iou, k=min(max_edges, iou.shape[1]), dim=1)
        mask = values > 0.01
        x = torch.arange(0, iou.shape[0]).unsqueeze(1).repeat(1, values.shape[1])
        x = x[mask]
        y = indices[mask]

    zposx = X[:, 0][x]
    zposy = X[:, 0][y]
    x = x[torch.abs(zposx - zposy) < z_thres1]
    y = y[torch.abs(zposx - zposy) < z_thres1]

    if max_edges is None:
        x1, y1 = torch.where(iou > iou_thres)
    else:
        values, indices = torch.topk(iou, k=min(max_edges, iou.shape[1]), dim=1)
        mask = values > iou_thres
        x1 = torch.arange(0, iou.shape[0]).unsqueeze(1).repeat(1, values.shape[1])
        x1 = x1[mask]
        y1 = indices[mask]

    zposx = X[:, 0][x1]
    zposy = X[:, 0][y1]
    need = (zposx != zposy) & (torch.abs(zposx - zposy) < z_thres2)
    x1 = x1[need]
    y1 = y1[need]
    # print(len(x), len(y), len(x1), len(y1))
    x1 = torch.cat([x, x1])
    y1 = torch.cat([y, y1])
    z = torch.ones_like(x1, dtype=torch.bool, requires_grad=False)
    z[: len(x)] = False
    return x1, y1, z

    # dis = 1 - iou
    # t = t.numpy().copy()
    # t[:, 0] *= z
    # t[:, 1] *= x
    # # print(t, y)
    # t[:, 2] *= y
    # t[:, 3] *= x
    # t[:, 4] *= y
    # xs = []
    # ys = []
    # for i in range(t.shape[0] - 1):
    #     for j in range(i + 1, t.shape[0]):
    #         # d = np.linalg.norm(t[i, :3] - t[j, :3])
    #         # d2 = np.linalg.norm(t[i, 3:5])
    #         # d3 = np.linalg.norm(t[j, 3:5])
    #         # d1 = max(d2, d3) / 2
    #         if np.abs(t[i, 0] - t[j, 0]) < 0.00001:
    #             if dis[i, j] < 0.99:
    #                 xs.append(i)
    #                 ys.append(j)
    #                 xs.append(j)
    #                 ys.append(i)
    #         else:
    #             if dis[i, j] + np.abs(t[i, 0] - t[j, 0]) < radius:
    #                 xs.append(i)
    #                 ys.append(j)
    #                 xs.append(j)
    #                 ys.append(i)
    # # radius_neighbors = NearestNeighbors(radius=radius)
    # # radius_neighbors.fit(t)
    # # neighbors = radius_neighbors.radius_neighbors(t, return_distance=False)
    # return xs, ys


def convertStage2Dataset(
    retdict,
    z_thres1=0.0001,
    z_thres2=0.1,
    iou_thres=0.4,
    num_classes=4,
    obj_thres=0.2,
    max_edges=None,
):
    X = retdict["feature"]

    # print("building up neighboring graph")
    xs, ys, edge_type = get_neighbors(
        X.clone().detach(),
        z_thres1=z_thres1,
        z_thres2=z_thres2,
        iou_thres=iou_thres,
        obj_thres=obj_thres,
        num_classes=num_classes,
        max_edges=max_edges,
    )
    # print(xs, ys, len(xs), len(ys))

    # print("building up neighboring graph done")

    y = retdict["label"]
    box_masks = retdict["box_mask"]
    boxes = retdict["boxes"]
    item_ids = retdict["item_id"]

    # item_mapping = {"": -1}
    # ids = []
    # for i in item_ids:
    #     if i not in item_mapping:
    #         item_mapping[i] = len(item_mapping)
    #     ids.append(item_mapping[i])
    # item_ids = [item_mapping[i] for i in item_ids]
    # item_ids = torch.tensor(item_ids, dtype=torch.long).to(X.device)
    # item_idx = item_ids[xs]
    # item_idy = item_ids[ys]
    # edge_label = (item_idx == item_idy) & (item_idx != -1)
    # edge_label = edge_label.long()
    # source = []
    # dest = []
    # edge_label = []
    # for i, j in zip(xs, ys):
    #     source.append(i)
    #     dest.append(j)
    #     if item_ids[i] == item_ids[j] and item_ids[i] != "":
    #         edge_label.append(1)
    #     else:
    #         edge_label.append(0)
    edges = torch.stack([xs, ys])
    # print(edges.shape)

    t = Data(x=X, edge_index=edges, y=y)
    t.inter_edges = edge_type
    t.box_masks = box_masks
    t.boxes = boxes
    t.item_id = item_ids
    # t.edge_label = edge_label  # torch.tensor(edge_label, dtype=torch.long)
    # if "sample_mapping" in retdict:
    #     t.sample_mapping = torch.tensor(
    #         [i for i in retdict["sample_mapping"].values()], dtype=int
    #     )
    return t


# def get_neighbors(
#     X, z_thres1=0.01, z_thres2=0.15, iou_thres=0.4, num_classes=5, obj_thres=0.1
# ):
#     X2 = X[:, 1:5].clone()
#     # expand a little so that boxes may overlap
#     X2[:, 2:] +=0.05

#     iou, _ = modules.box_iou(center_to_corners_format(X2), center_to_corners_format(X2))

#     obj = X[:, 5 : 5 + num_classes]
#     obj = torch.sigmoid(obj)
#     obj = obj > obj_thres
#     obj = obj.any(dim=1)

#     # queries with less than obj_thres are not considered
#     iou -= 2 - 2 * obj.float()
#     iou = iou.T
#     iou -= 2 - 2 * obj.float()

#     # remove self-edges
#     # iou -= torch.eye(iou.shape[0], dtype=torch.float32, device=X.device)
#     zpos = X[:, 0]
#     zpos_same = (zpos[:, None] == zpos[None, :]).float()
#     iou += zpos_same

#     x, y = torch.where(iou > 0.01)
#     zposx = X[:, 0][x]
#     zposy = X[:, 0][y]
#     x = x[torch.abs(zposx - zposy) < z_thres1]
#     y = y[torch.abs(zposx - zposy) < z_thres1]

#     x1, y1 = torch.where(iou > iou_thres)
#     zposx = X[:, 0][x1]
#     zposy = X[:, 0][y1]
#     need = (zposx != zposy) & (torch.abs(zposx - zposy) < z_thres2)
#     x1 = x1[need]
#     y1 = y1[need]
#     x = torch.cat([x, x1])
#     y = torch.cat([y, y1])
#     return x, y


# def convertStage2Dataset(
#     retdict, z_thres1=0.01, z_thres2=0.15, iou_thres=0.4, num_classes=5, obj_thres=0.2
# ):
#     X = retdict["feature"]

#     # print("building up neighboring graph")
#     xs, ys = get_neighbors(
#         X.clone().detach(),
#         z_thres1=z_thres1,
#         z_thres2=z_thres2,
#         iou_thres=iou_thres,
#         obj_thres=obj_thres,
#         num_classes=num_classes,
#     )
#     # print(xs, ys, len(xs), len(ys))

#     # print("building up neighboring graph done")

#     y = retdict["label"]
#     box_masks = retdict["box_mask"]
#     boxes = retdict["boxes"]
#     # item_ids = retdict["item_id"]

#     # item_mapping = {"": -1}
#     # for i in item_ids:
#     #     if i not in item_mapping:
#     #         item_mapping[i] = len(item_mapping)
#     # item_ids = [item_mapping[i] for i in item_ids]
#     # item_ids = torch.tensor(item_ids, dtype=torch.long).to(X.device)
#     # item_idx = item_ids[xs]
#     # item_idy = item_ids[ys]
#     # edge_label = (item_idx == item_idy) & (item_idx != -1)
#     # edge_label = edge_label.long()
#     # source = []
#     # dest = []
#     # edge_label = []
#     # for i, j in zip(xs, ys):
#     #     source.append(i)
#     #     dest.append(j)
#     #     if item_ids[i] == item_ids[j] and item_ids[i] != "":
#     #         edge_label.append(1)
#     #     else:
#     #         edge_label.append(0)
#     edges = torch.stack([xs, ys])
#     # print(edges.shape)

#     t = Data(x=X, edge_index=edges, y=y)
#     t.box_masks = box_masks
#     t.boxes = boxes
#     # t.edge_label = edge_label  # torch.tensor(edge_label, dtype=torch.long)
#     # if "sample_mapping" in retdict:
#     #     t.sample_mapping = torch.tensor(
#     #         [i for i in retdict["sample_mapping"].values()], dtype=int
#     #     )
#     return t


# helper functions for model building
def buildModel(configs, args, checkpoint=None):
    if configs["model"]["name"] == "conditional_detr":
        if configs["model"]["task"] == "segmentation":
            config = ConditionalDetrConfig(use_pretrained_backbone=False, **args)
            seg_model = ConditionalDetrForSegmentation(config)
            if len(configs["model"]["pretrained"]) > 0:
                model = ConditionalDetrForObjectDetection.from_pretrained(
                    configs["model"]["pretrained"], ignore_mismatched_sizes=True, **args
                )
                seg_model.conditional_detr.load_state_dict(model.state_dict())
            model = seg_model
        elif configs["model"]["task"] == "detection":
            if (
                "pretrained" in configs["model"]
                and len(configs["model"]["pretrained"]) > 0
            ):
                model = ConditionalDetrForObjectDetection.from_pretrained(
                    configs["model"]["pretrained"], ignore_mismatched_sizes=True, **args
                )
            else:
                config = ConditionalDetrConfig(use_pretrained_backbone=False, **args)
                model = ConditionalDetrForObjectDetection(config)

    elif configs["model"]["name"] == "deformable_detr":
        if "pretrained" in configs["model"] and len(configs["model"]["pretrained"]) > 0:
            model = DeformableDetrForObjectDetection.from_pretrained(
                configs["model"]["pretrained"], ignore_mismatched_sizes=True, **args
            )
        else:
            config = DeformableDetrConfig(use_pretrained_backbone=False, **args)
            model = DeformableDetrForObjectDetection(config)

    elif configs["model"]["name"] == "detr":
        if configs["model"]["task"] == "segmentation":
            if len(configs["model"]["pretrained"]) > 0:
                model = DetrForSegmentation.from_pretrained(
                    configs["model"]["pretrained"], ignore_mismatched_sizes=True, **args
                )
            else:
                config = DetrConfig(use_pretrained_backbone=False, **args)
                model = DetrForSegmentation(config)
        elif configs["model"]["task"] == "detection":
            if len(configs["model"]["pretrained"]) > 0:
                model = DetrForObjectDetection.from_pretrained(
                    configs["model"]["pretrained"], ignore_mismatched_sizes=True, **args
                )
            else:
                config = DetrConfig(use_pretrained_backbone=False, **args)
                model = DetrForObjectDetection(config)
    else:
        raise NotImplementedError

    if (
        "renew_position_embeddings" in configs["model"]
        and configs["model"]["renew_position_embeddings"]
    ):
        print("renew position embeddings")
        state_dict = model.state_dict()
        for i in state_dict:
            if "query_position_embeddings" in i:
                state_dict[i] = torch.randn(state_dict[i].shape)
        model.load_state_dict(state_dict)

    if checkpoint is not None and len(checkpoint) > 0:
        if checkpoint.endswith(".ckpt"):
            ckpt = torch.load(checkpoint)["state_dict"]
        else:
            ckpt = torch.load(checkpoint)
        for i, j in model.named_parameters():
            if i in ckpt and j.shape != ckpt[i].shape:
                if "query_position_embeddings" not in i:
                    del ckpt[i]
                else:
                    t = j.clone()
                    t[: min(t.shape[0], ckpt[i].shape[0])] = ckpt[i][
                        : min(t.shape[0], ckpt[i].shape[0])
                    ]
                    # t = torch.randn_like(j)
                    ckpt[i] = t
        need_del = []
        p = [j for j, _ in model.named_parameters()]
        for i in ckpt:
            if i not in p or ckpt[i].shape != model.state_dict()[i].shape:
                need_del.append(i)
        for i in need_del:
            del ckpt[i]
        model.load_state_dict(ckpt, strict=False)

    return model


def getModel(configs):

    if "stage" not in configs["model"]:
        configs["model"]["stage"] = "stage 1"

    if "checkpoint" not in configs["model"]:
        configs["model"]["checkpoint"] = None

    # if isinstance(configs["data"]["require_mask"], dict):

    #     models = {}

    #     for i in configs["model"]["args"]:
    #         models[i] = buildModel(
    #             configs, configs["model"]["args"][i], configs["model"]["checkpoint"]
    #         )
    # else:
    models = buildModel(
        configs, configs["model"]["args"], configs["model"]["checkpoint"]
    )

    if "lr_backbone" not in configs["training"]:
        configs["training"]["lr_backbone"] = None

    if "lr_detr" not in configs["training"]:
        configs["training"]["lr_detr"] = None

    if "dropout" not in configs["model"]:
        configs["model"]["dropout"] = False

    if "scheduler_step" not in configs["training"]:
        configs["training"]["scheduler_step"] = -1

    if "warmup_epoches" not in configs["training"]:
        configs["training"]["warmup_epoches"] = 1

    if "pick_num" not in configs["training"]:
        configs["training"]["pick_num"] = 8

    if "mask_alpha" not in configs["training"]:
        configs["training"]["mask_alpha"] = 0.5

    if "mask_in_channel" not in configs["model"]:
        configs["model"]["mask_in_channel"] = 3

    if "mask_out_channel" not in configs["model"]:
        configs["model"]["mask_out_channel"] = 1

    if "class_weight" not in configs["model"]:
        configs["model"]["class_weight"] = None

    if "consistency_regularization_coef" not in configs["model"]:
        configs["model"]["consistency_regularization_coef"] = 0.5

    if "box_head" not in configs["model"]:
        configs["model"]["box_head"] = "lora"

    if "feature_dim" not in configs["model"]:
        configs["model"]["feature_dim"] = 256

    if "C_in" not in configs["model"]:
        configs["model"]["C_in"] = (
            configs["model"]["feature_dim"] + configs["model"]["output_dim"] + 4
        )
    # print(configs["model"]["feature_dim"], configs["model"]["C_in"])

    if "lr" not in configs["training"]:
        configs["training"]["lr"] = 1e-4

    if "weight_decay" not in configs["training"]:
        configs["training"]["weight_decay"] = 0.0
        
    if "nms" not in configs["model"]:
        configs["model"]["nms"] = -1.0

    model = modules.DetrModel(
        configs["model"]["stage"],
        models,
        lr=configs["training"]["lr"],
        lr_backbone=configs["training"]["lr_backbone"],
        lr_detr=configs["training"]["lr_detr"],
        weight_decay=configs["training"]["weight_decay"],
        feature_dim=configs["model"]["feature_dim"],
        gnn_in_channel=configs["model"]["C_in"],
        output_dim=configs["model"]["output_dim"],
        layer_type=configs["model"]["layer_type"],
        dropout=configs["model"]["dropout"],
        scheduler_step=configs["training"]["scheduler_step"],
        warmup_epoches=configs["training"]["warmup_epoches"],
        pick_num=configs["training"]["pick_num"],
        mask_alpha=configs["training"]["mask_alpha"],
        mask_in_channel=configs["model"]["mask_in_channel"],
        mask_out_channel=configs["model"]["mask_out_channel"],
        class_weights=configs["model"]["class_weight"],
        consistency_regularization_coef=configs["model"][
            "consistency_regularization_coef"
        ],
        box_head=configs["model"]["box_head"],
        nms=configs["model"]["nms"],
    )

    if "load" in configs["model"] and configs["model"]["load"] is not None:
        t = torch.load(configs["model"]["load"], map_location="cpu")
        if configs["model"]["load"].endswith(".ckpt"):
            ckpt = t["state_dict"]
        else:
            ckpt = t
        need_del = []
        p = list(model.state_dict().keys())  # [j for j, _ in model.named_parameters()]
        # print("model parameters ", p)
        for i in ckpt:
            if i not in p or ckpt[i].shape != model.state_dict()[i].shape:
                need_del.append(i)
        for i in need_del:
            del ckpt[i]
        print("incompatible parameters", need_del)
        model.load_state_dict(ckpt, strict=False)
        print("finish loading parameters")

    return model


def get_stage2_model_embeddings(config_path, dataset, checkpoint_path="last.ckpt"):
    model = loadModel(config_path, checkpoint_path)

    ds = torch.load(dataset)

    with torch.no_grad():
        model.eval()
        res = model(x=ds.x, edge_index=ds.edge_index)
        # for i in ds:
        #     i = i.to("cuda")
        #     model(i.x, i.edge_index, i.train_mask)
        #     model(i.x, i.edge_index, i.val_mask)
    return res


def loadModel(path, checkpoint="last.ckpt"):
    with open(os.path.join(path, "config.json"), "r") as f:
        configs = json.load(f)

    model = getModel(configs)
    # print(path, checkpoint)
    ckpt = torch.load(os.path.join(path, checkpoint), map_location="cpu")["state_dict"]

    p = list(model.state_dict().keys())  # [j for j, _ in model.named_parameters()]
    # print("model parameters ", p)
    need_del = []
    for i in ckpt:
        if i not in p or ckpt[i].shape != model.state_dict()[i].shape:
            need_del.append(i)
    for i in need_del:
        del ckpt[i]
    print("incompatible parameters", need_del)
    model.load_state_dict(ckpt, strict=False)
    print("finish loading parameters")

    # need_del = []
    # p = [j for j, _ in model.named_parameters()]
    # for i in ckpt:
    #     if i not in p or ckpt[i].shape != model.state_dict()[i].shape:
    #         need_del.append(i)
    # for i in need_del:
    #     del ckpt[i]
    # model.load_state_dict(ckpt, strict=False)

    return model


def prepareMaskDataset(retdict, resdict):
    masks = retdict["masks"]
    images = retdict["images"]
    images = torch.stack(images, dim=0)
    embeds = resdict["embeddings"][retdict["box_mask"]]
    sapple_mapping = retdict["sample_mapping"]
    return {
        "embed": embeds,
        "pixel_values": images,
        "masks": masks,
        "sample_mapping": sapple_mapping,
    }


def generateMasks(model, embed, image, pixel_mask=None, batch_size=5, device="cpu"):
    model.stage = "stage mask"
    model = model.to(device)
    assert image.dim() == 3
    assert embed.dim() == 2
    res = []
    image = image[None, :, :, :].repeat(batch_size, 1, 1, 1)
    if pixel_mask is not None:
        pixel_mask = pixel_mask[None, :, :, :].repeat(batch_size, 1, 1, 1)
        pixel_mask = pixel_mask.to(device)
    image = image.float().to(device)
    embed = embed.to(device)
    for i in tqdm(range(0, embed.shape[0], batch_size)):
        with torch.no_grad():
            input_embed = embed[i : i + batch_size]
            res.append(
                model(
                    pixel_values=image[: input_embed.shape[0]].float(),
                    pixel_mask=pixel_mask,
                    stage_2_embeds=input_embed,
                )
            )
    res = torch.cat(res, dim=0)
    res = res.cpu()
    model = model.cpu()
    return res


def calculate_iou(pred_bbox, gt_bbox):
    """Calculate Intersection over Union (IoU) between predicted and ground truth bounding boxes."""
    # Compute the coordinates of the intersection rectangle
    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[2], gt_bbox[2])
    y2 = min(pred_bbox[3], gt_bbox[3])

    # Compute the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute the area of both bounding boxes
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

    # Compute the area of the union
    union_area = pred_area + gt_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou


def calculate_precision_recall(pred_bboxes_list, gt_bboxes_list, iou_threshold=0.5):
    """Calculate Precision and Recall for a given IoU threshold."""
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    # Keep track of which ground truth boxes are matched
    for pred_bboxes, gt_bboxes in zip(pred_bboxes_list, gt_bboxes_list):
        matched_gt = []
        for pred_bbox in pred_bboxes:
            ious = [calculate_iou(pred_bbox, gt_bbox) for gt_bbox in gt_bboxes]
            if len(ious) == 0:
                max_iou = 0
            else:
                max_iou = max(ious)

            if max_iou >= iou_threshold:
                idx = ious.index(max_iou)
                if idx not in matched_gt:
                    tp += 1
                    matched_gt.append(idx)
                # tp += 1
                # matched_gt.append(ious.index(max_iou))
            else:
                fp += 1

        fn += len(gt_bboxes) - len(matched_gt)

    print(tp, fp, fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


def calculate_ap(pred_bboxes, gt_bboxes, iou_thresholds=[0.5, 0.75]):
    """Calculate AP at different IoU thresholds (AP50, AP75)."""
    aps = {}
    for iou_threshold in iou_thresholds:
        precision, recall = calculate_precision_recall(
            pred_bboxes, gt_bboxes, iou_threshold
        )
        ap = (
            precision * recall
        )  # Simplified AP calculation (you may use interpolation methods)
        aps[f"AP@{iou_threshold*100}"] = ap

    return aps


def get_target_df(graph):
    z = graph.x[:, [0]]
    boxes = graph.boxes
    labels = graph.y.unsqueeze(1)
    df = torch.cat([z, boxes, labels], dim=1)

    target_df = pd.DataFrame(df.numpy(), columns=["z", "x", "y", "w", "h", "label"])
    target_df["z"] = target_df["z"] * 500
    target_df["z"] = target_df["z"].astype(int)
    target_df["label"] = target_df["label"].astype(int)
    target_df = target_df[target_df["label"] != 5]
    target_df["name"] = target_df["label"].map(
        {
            0: "ribo",
            1: "microtubule",
            2: "nucleus",
            3: "endoplasmic_reticulum_or_Golgi",
            4: "mitochondria",
        }
    )

    bboxes = target_df[["x", "y", "w", "h"]].values
    bboxes = convertBoxes(torch.tensor(bboxes))
    bboxes = torch.tensor(bboxes) * torch.tensor([800, 800, 800, 800])
    target_df[["x1", "y1", "x2", "y2"]] = bboxes.numpy()
    target_df["mask_id"] = range(len(target_df))
    return target_df


def get_result_df(model, graph):
    model.stage = "stage 2"
    model.eval()
    with torch.no_grad():
        res = model(graph.x, graph.edge_index)
    z = graph.x[:, [0]]
    prob = torch.softmax(res["predict"], 1)
    pred_boxes = res["box"]
    df = torch.cat([z, pred_boxes, prob], 1)
    df = df.detach().numpy()
    df = pd.DataFrame(
        df,
        columns=[
            "z",
            "x",
            "y",
            "w",
            "h",
            "ribo",
            "microtubule",
            "nucleus",
            "endoplasmic_reticulum_or_Golgi",
            "mitochondria",
            "None",
        ],
    )
    df["z"] = df["z"] * 500
    df["z"] = df["z"].astype(int)
    bboxes = df[["x", "y", "w", "h"]].values
    bboxes = convertBoxes(torch.tensor(bboxes))
    bboxes = torch.tensor(bboxes) * torch.tensor([800, 800, 800, 800])
    df[["x1", "y1", "x2", "y2"]] = bboxes.numpy()
    subdf = df[
        [
            "ribo",
            "microtubule",
            "nucleus",
            "endoplasmic_reticulum_or_Golgi",
            "mitochondria",
        ]
    ]
    df["max"] = subdf.max(axis=1)
    df["largest"] = subdf.idxmax(axis=1)
    return df


def post_process_df(
    df, thres=0.5, nms=0.5, min_sample=3, eps=0.4, dis_penalty=0.01 * 0.05
):
    df = df[df["max"] > thres]
    dfs = []
    for i, subdf in df.groupby("z"):
        keep = bbnms(
            nms,
            subdf[["x1", "y1", "x2", "y2"]].values,
            subdf["max"].values,
            subdf["largest"].values,
        )
        dfs.append(subdf.iloc[keep.numpy()])

    df = pd.concat(dfs, ignore_index=True)

    dbscan = DBSCAN(min_samples=min_sample, eps=eps, metric="precomputed")
    iou = get_iou(df[["x", "y", "w", "h"]].values)
    iou = 1 - iou
    mask = df["z"].values
    mask = mask[:, None] == mask[None, :]
    iou[mask] = 2
    mask = df["z"].values
    z_penalty = (np.abs(mask[:, None] - mask[None, :]) ** 2) * dis_penalty
    iou = iou + z_penalty
    dbscan.fit(iou)
    df["label"] = list(dbscan.labels_)

    return df


def cal_ap(df, target_df, iou_thresholds=[0.5, 0.75]):
    pres = []
    gts = []
    slices = list(target_df["z"])
    slices.extend(list(df["z"]))
    slices = np.unique(slices)
    for j in slices:
        s1 = []
        s2 = []
        sub1 = target_df[target_df["z"] == j]
        sub2 = df[df["z"] == j]
        for i, r in sub1.iterrows():
            s1.append(np.array([r["x1"], r["y1"], r["x2"], r["y2"]]))
        for i, r in sub2.iterrows():
            s2.append(np.array([r["x1"], r["y1"], r["x2"], r["y2"]]))
        pres.append(s2)
        gts.append(s1)

    return calculate_ap(pres, gts, iou_thresholds=iou_thresholds)


def readTomogram(filename):
    with mrcfile.open(filename, permissive=True) as m:
        return m.data


def pickAndLoadBest(model, path):
    pattern = re.compile(
        r"epoch=(\d+)"
        r"(?:-total_validate_auroc=([0-9.]+))?"
        r"-total_validate_loss=([0-9.]+)"
        r"(?:-v(\d+))?"
        r"\.ckpt$"
    )
    for i in os.listdir(path):
        best = ""
        loss_best = 1e100
        if i.endswith(".ckpt"):
            m = pattern.match(i)
            if m is not None:
                epoch = int(m.group(1))
                auroc = float(m.group(2)) if m.group(2) is not None else 0
                loss = float(m.group(3))
                v = int(m.group(4)) if m.group(4) is not None else 0
                if loss_best > loss:
                    loss_best = loss
                    best = i
    a = torch.load(os.path.join(path, best), map_location="cpu")["state_dict"]
    model.load_state_dict(a, strict=False)
    return model
