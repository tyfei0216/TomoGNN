import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN

import utils


def getLabels(
    subdf, min_samples=3, eps=0.4, dis_penalty_coef=1.0, dis_penalty_cutoff=2.0
):
    # print(min_samples, eps, dis_penalty_coef, dis_penalty_cutoff)
    # print("df len", len(subdf))
    hdb = DBSCAN(min_samples=min_samples, eps=eps, metric="precomputed")
    iou = utils.get_iou(subdf[["x", "y", "w", "h"]].values)
    iou = 1 - iou
    mask = subdf["z"].values
    dis_penalty = np.abs(mask[:, None] - mask[None, :]).astype(np.float32)
    # print(dis_penalty)
    # print(iou)
    mask = mask[:, None] == mask[None, :]
    iou[mask] = 2.0
    # mask = subdf["z"].values

    dis_penalty[dis_penalty > dis_penalty_cutoff] = 2.0 / dis_penalty_coef
    # print(dis_penalty, dis_penalty_coef)
    dis_penalty *= dis_penalty_coef
    iou = iou + dis_penalty
    hdb.fit(iou)
    # print(iou)
    return hdb.labels_
    # subdf["label"] = hdb.labels_


# np.unique(hdb.labels_).tolist()


def processClass(df, classname, min_prob, nms, dbscan_prams={}):
    df["class_value"] = df[classname] * 2 - df["max"]
    subdf = df[df["class_value"] > min_prob]
    subdf = subdf[subdf["unlabeled"]]
    # print(len(subdf))
    # print(utils.convertBoxes(torch.tensor(subdf[["x", "y", "w", "h"]].values)).shape)
    alldfs = []
    for i, r in subdf.groupby("z"):
        keep = utils.bbnms(
            nms,
            utils.convertBoxes(torch.tensor(r[["x", "y", "w", "h"]].values)),
            torch.tensor(r["class_value"].values),
            np.array(["same" for i in range(len(r))]),
        )
        r["bbnms"] = False
        r.loc[r.index[keep.numpy()], "bbnms"] = True
        r = r[r["bbnms"]]
        alldfs.append(r)
    subdf = pd.concat(alldfs)
    # print(subdf)
    # print(len(subdf), "after bbnms")
    # print(dbscan_prams)
    labels = getLabels(subdf, **dbscan_prams)
    print("number of instances in class ", len(np.unique(labels)))
    subdf["label"] = list(labels)
    return subdf


def findClosestIndex(lst, target):
    lst = np.array(lst)  # Convert to NumPy array
    closest_index = np.abs(lst - target).argmin()
    return closest_index


def markFilter(df, classname, targetdf, max_cnt=100, remove_iou=0.5):
    label = []
    weight = []
    for i, r in targetdf.groupby("label"):
        if i == -1:
            continue
        label.append(i)
        weight.append((r[classname]).sum())
    label = np.array(label)
    weight = np.array(weight)
    k = np.argsort(-weight)[:max_cnt]
    required_label = label[k]
    for label in required_label:
        subdf2 = targetdf[targetdf["label"] == label]
        df.loc[subdf2.index, "label_id"] = label
        df.loc[subdf2.index, "label"] = classname
        for i in range(np.min(subdf2["z"]), np.max(subdf2["z"]) + 5, 5):
            j = findClosestIndex(subdf2["z"].values, i)
            # print(i, j)
            posz_df = df[df["z"] == i]
            required_indexes = list(posz_df.index)
            required_indexes.append(subdf2.index[j])
            required_df = df.loc[required_indexes]
            iou = utils.get_iou(required_df[["x", "y", "w", "h"]].values)
            iou = iou[-1]
            iou = iou > remove_iou
            required_df = required_df[iou]
            df.loc[required_df.index, "unlabeled"] = False


def pickTopCountors(countors, n, min_threshold):

    # Combine items and weights into a list of tuples
    weights = [cv2.contourArea(cnt) for cnt in countors]
    item_weight_pairs = list(zip(countors, weights))

    # Filter items based on the minimum threshold
    filtered_items = [
        (item, weight) for item, weight in item_weight_pairs if weight >= min_threshold
    ]

    # Sort filtered items by weight in descending order
    sorted_items = sorted(filtered_items, key=lambda x: x[1], reverse=True)

    # Select the top n items (or all if there are fewer than n items)
    selected_items = sorted_items[:n]

    # Return only the item names
    return [item for item, weight in selected_items]


def refineMask(
    mask,
    threshold=0.5,
    morph_open_kernal_size=(5, 5),
    morph_open_iterations=2,
    morph_close_kernal_size=(5, 5),
    morph_close_iterations=2,
    blur_ksize=(15, 15),
    blur_sigma=0,
    contour_area_threshold=50,
    max_contours=1,
    apply_contours=True,
    apply_convex_hull=False,
):
    # Step 1: Convert to binary
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    # _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Step 2: Apply morphological operations
    kernel = np.ones(morph_open_kernal_size, np.uint8)
    opened_mask = cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iterations
    )
    kernel = np.ones(morph_close_kernal_size, np.uint8)
    closed_mask = cv2.morphologyEx(
        opened_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iterations
    )

    # Step 3: Smooth edges
    blurred_mask = cv2.GaussianBlur(closed_mask, blur_ksize, blur_sigma)

    if not apply_contours:
        return blurred_mask
    # Step 4: Contour filtering
    contours, _ = cv2.findContours(
        blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filtered_contours = pickTopCountors(contours, max_contours, contour_area_threshold)
    # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    # Step 5: Draw filtered contours
    refined_mask = np.zeros_like(binary_mask)
    cv2.drawContours(refined_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    if not apply_convex_hull:
        return refined_mask
    # Step 6: Convex Hull (optional)
    hull_mask = np.zeros_like(binary_mask)
    for cnt in filtered_contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    return hull_mask


def getMaskInputs(
    df,
    model,
    dataset,
    class_id,
    min_length,
    extend,
    stage1_outputs={},
    zpos_min=0,
    zpos_max=500,
    iou_thres=0.5,
):
    model.stage = "stage 1"
    df = df.sort_values(by="z")
    zmin = df["z"].min() - 2
    zmax = df["z"].max() + 2
    zmid = (zmin + zmax) // 2
    if zmin > zmid - min_length:
        zmin = zmid - min_length
    if zmax < zmid + min_length:
        zmax = zmid + min_length
    zmin -= extend
    zmax += extend
    zmin = max(zmin, zpos_min)
    zmax = min(zmax, zpos_max)
    inputs = {}
    for i in range(zmin, zmax + 1):
        inputs[i] = {}
        data = dataset.__getitem__(pos=i)
        t = data["pixel_values"].unsqueeze(0)
        inputs[i]["image"] = t
        q = findClosestIndex(df["z"].values, i)
        inputs[i]["bboxes"] = torch.tensor(
            df.iloc[q][["x", "y", "w", "h"]].astype(float).values
        ).unsqueeze(0)
        q = df.index[q]
        if i not in stage1_outputs:
            with torch.no_grad():
                # data = dataset2.__getitem__(pos=i)
                data["pixel_values"] = (
                    data["pixel_values"].unsqueeze(0).float().to(model.device)
                )
                data["pixel_mask"] = (
                    data["pixel_mask"].unsqueeze(0).float().to(model.device)
                )
                outputs = model(
                    pixel_values=data["pixel_values"],
                    pixel_mask=data["pixel_mask"],
                )
            stage1_outputs[i] = {}
            stage1_outputs[i]["pred_boxes"] = (
                outputs["pred_boxes"].cpu().squeeze(0).numpy()
            )
            logits = outputs["logits"].squeeze(0)
            prob = torch.sigmoid(logits)
            stage1_outputs[i]["prob"] = prob.cpu().numpy()
            stage1_outputs[i]["embeddings"] = (
                outputs["last_hidden_state"].cpu().squeeze(0)
            )

        boxes = stage1_outputs[i]["pred_boxes"]
        boxes = np.concatenate([inputs[i]["bboxes"], boxes], 0)
        iou = utils.get_iou(boxes)
        iou = iou[0, 1:]
        score = stage1_outputs[i]["prob"][:, class_id]
        score[iou < iou_thres] -= 1.0
        score = score + iou
        take_id = np.argmax(score)

        # inputs[i]["embed"] = res["embeddings"][q].detach().cpu().unsqueeze(0)
        inputs[i]["embed"] = stage1_outputs[i]["embeddings"][[take_id]]
        # print(data["pixel_values"].shape)
        inputs[i]["input_mask"] = xywh2mask(
            inputs[i]["bboxes"].squeeze().numpy(),
            (inputs[i]["image"].shape[2], inputs[i]["image"].shape[3]),
        )
        # print(sum(inputs[i]["input_mask"].flatten()))

    return inputs, stage1_outputs


def getMasks(model, inputs, sigma=1):
    model.stage = "stage mask"
    masks = {}
    for i in inputs:
        with torch.no_grad():
            outputs = model(
                pixel_values=inputs[i]["image"].float().to(model.device),
                box=inputs[i]["bboxes"].float().to(model.device),
                embed=inputs[i]["embed"].float().to(model.device),
            )
        mask = torch.sigmoid(outputs).detach().cpu().numpy()

        # mask = refineMask(mask)
        masks[i] = mask

    aligned_masks = []
    for i in range(min(inputs.keys()), max(inputs.keys()) + 1):
        if i not in masks:
            continue
        aligned_masks.append(masks[i])

    aligned_masks = np.concatenate(aligned_masks, axis=0)
    smoothed_masks = gaussian_filter(aligned_masks, sigma=sigma)
    return smoothed_masks


def xywh2mask(box, img_size):
    # masks = []
    # for box in boxes:
    x, y, w, h = box
    mask = np.zeros(img_size, dtype=np.uint8)
    x1 = max(int((x - w / 2) * img_size[1]), 0)
    x2 = min(int((x + w / 2) * img_size[1]), img_size[1] - 1)
    y1 = max(int((y - h / 2) * img_size[0]), 0)
    y2 = min(int((y + h / 2) * img_size[0]), img_size[0] - 1)
    mask[y1:y2, x1:x2] = 1
    return mask
