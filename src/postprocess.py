import io
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.ndimage import center_of_mass, gaussian_filter, gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import sort_graph_by_row_values
from tqdm import tqdm

import utils


def generatedf(
    model,
    dataset,
    gap,
    columns,
    has_none=False,
    empty=4,
    num_classes=4,
    z_multiply=None,
    filter_prob=0.05,
    # return_mask=False,
    return_embeds=False,
    max_edges=50,
    nms=0.4,
    iou_thres=0.4,
):
    # silence dubugging outputs
    # saved_stdout = sys.stdout
    # sys.stdout = io.StringIO()
    if z_multiply is None:
        z_multiply = dataset.zpos_max
    t = [
        "z",
        "x",
        "y",
        "w",
        "h",
    ]
    t.extend(columns)
    with torch.no_grad():
        alldfs = []
        features = []
        # all_masks = []
        embeds = []
        for i in range(gap):
            model.stage = "stage 1"
            retdict = utils.buildStage2(
                model,
                dataset,
                (800, 800),
                filter_prob,
                gap=gap,
                offset=i,
                has_none=has_none,
                empty=empty,
                nms=nms,
            )
            # if return_mask:
            #     all_masks.append(retdict["masks"])
            graph = utils.convertStage2Dataset(
                retdict,
                num_classes=num_classes,
                obj_thres=0.1,
                max_edges=max_edges,
                iou_thres=iou_thres,
            )
            # print("build graph")
            features.append(graph.x.cpu().numpy())
            model.stage = "stage 2"
            graph = graph.to(model.device)
            res = model(graph.x, graph.edge_index)
            embeds.append(res["embeddings"].cpu().numpy())
            z = graph.x[:, [0]].cpu() * z_multiply
            # z = z.long()
            prob = torch.softmax(res["predict"], 1).cpu()
            pred_boxes = res["box"].cpu()
            # print("prepare df")
            df = torch.cat([z, pred_boxes, prob], 1)
            df = df.numpy()
            df = pd.DataFrame(df, columns=t)
            df["z"] = df["z"].round(0).astype(int)
            # df["z"] = df["z"].astype(int)
            subdf = df[columns]
            df["max"] = subdf.max(axis=1)
            df["largest"] = subdf.idxmax(axis=1)
            # print(df.shape, graph.x.shape)
            alldfs.append(df)
    # sys.stdout = saved_stdout
    embeds = np.concatenate(embeds, 0)
    alldfs = pd.concat(alldfs, ignore_index=True)
    if return_embeds:
        return alldfs, features
    # if return_mask:
    #     print(all_masks)
    #     all_masks = torch.cat(all_masks, 0)
    #     return alldfs, graphs, all_masks.detach().cpu().numpy()
    return alldfs


def generatedfBySlice(
    model,
    dataset,
    gap,
    columns,
    has_none=False,
    empty=4,
    num_classes=4,
    z_multiply=500,
    length=15,
    nms=0.4,
    iou_thres=0.4,
    return_mask=False,
    return_embeds=False,
    return_retdict=False,
):
    t = [
        "z",
        "x",
        "y",
        "w",
        "h",
    ]
    t.extend(columns)
    with torch.no_grad():
        alldfs = []
        # graphs = []
        all_masks = []
        embeds = []
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        use_gpu = model_device.type != "cpu"

        model.stage = "stage 1"
        retdict = utils.runStage1(
            model, dataset, (800, 800), has_none=has_none, empty=empty, nms=nms
        )
        print("finish stage1")

        slice_ids = sorted(retdict.keys())
        model.stage = "stage 2"

        for slice_id in tqdm(slice_ids, desc="Stage 2 slices"):
            needIDs, pos = utils.pickout(retdict, slice_id, gap, length)
            center_id = needIDs[pos]

            merged = {
                "feature": [],
                "label": [],
                "box_mask": [],
                "boxes": [],
                "item_id": [],
            }
            if return_mask:
                merged["masks"] = []

            for need_id in needIDs:
                if need_id not in retdict:
                    continue
                for key in ["feature", "label", "box_mask", "boxes", "item_id"]:
                    merged[key].extend(retdict[need_id][key])
                if return_mask and "masks" in retdict[need_id]:
                    merged["masks"].extend(retdict[need_id]["masks"])

            if len(merged["feature"]) == 0:
                continue

            merged["feature"] = torch.cat(merged["feature"], dim=0)
            merged["label"] = torch.cat(merged["label"], dim=0)
            merged["box_mask"] = torch.cat(merged["box_mask"], dim=0)
            merged["boxes"] = torch.cat(merged["boxes"], dim=0)

            graph = utils.convertStage2Dataset(
                merged, num_classes=num_classes, obj_thres=0.1, iou_thres=iou_thres
            )
            # graphs.append(graph)

            # Keep retdict/merged tensors on CPU. Move only stage-2 graph tensors when needed.
            if use_gpu:
                graph_x = graph.x.to(model_device, non_blocking=True)
                graph_edge_index = graph.edge_index.to(model_device, non_blocking=True)
            else:
                graph_x = graph.x
                graph_edge_index = graph.edge_index

            res = model(graph_x, graph_edge_index)
            slice_mask = torch.round(graph_x[:, 0] * z_multiply).long() == int(
                center_id
            )
            if not torch.any(slice_mask):
                continue

            prob = torch.softmax(res["predict"], 1)[slice_mask]
            pred_boxes = res["box"][slice_mask]
            z = graph_x[slice_mask][:, [0]] * z_multiply

            df = torch.cat([z, pred_boxes, prob], 1)
            df = pd.DataFrame(df.detach().cpu().numpy(), columns=t)
            df["z"] = df["z"].round(0).astype(int)
            subdf = df[columns]
            df["max"] = subdf.max(axis=1)
            df["largest"] = subdf.idxmax(axis=1)
            df["center_slice"] = center_id
            alldfs.append(df)

            if return_embeds:
                embeds.append(res["embeddings"][slice_mask].detach().cpu().numpy())
            if return_mask and len(merged["masks"]) > 0:
                all_masks.append(torch.cat(merged["masks"], dim=0).detach().cpu())

        alldfs = (
            pd.concat(alldfs, ignore_index=True)
            if len(alldfs) > 0
            else pd.DataFrame(columns=t + ["max", "largest"])
        )

        if return_retdict:
            return alldfs, retdict

        if return_embeds:
            embeds = np.concatenate(embeds, 0) if len(embeds) > 0 else np.empty((0,))
            return alldfs, embeds
        if return_mask:
            all_masks = (
                torch.cat(all_masks, 0) if len(all_masks) > 0 else torch.empty(0)
            )
            return alldfs, all_masks.detach().cpu().numpy()

        return alldfs


def myscan(iou_matrix, zpos, eps=0.6, max_z_diff=2):
    num = iou_matrix.shape[0]
    # print(zpos, max_z_diff)
    labels = -1 * np.ones(num, dtype=int)
    current_label = 1
    max_pos = np.max(zpos)
    for i in range(num):
        if labels[i] != -1:
            continue
        current = i
        while True:
            labels[current] = current_label
            ifbreak = True
            # print(
            #     zpos[current] + 1, min(int(zpos[current] + max_z_diff + 1), max_pos + 1)
            # )
            for z in range(
                zpos[current] + 1, min(int(zpos[current] + max_z_diff + 1), max_pos + 1)
            ):
                neighbors = np.where((zpos == z) & (iou_matrix[current] < eps))[0]
                # print(neighbors)
                # break
                if len(neighbors) == 0:
                    continue
                neighbors = neighbors[labels[neighbors] == -1]
                # print(neighbors)
                if len(neighbors) == 0:
                    continue
                get_id = np.argmin(iou_matrix[current][neighbors])
                current = neighbors[get_id]
                ifbreak = False
                break
            if ifbreak:
                break
        current_label += 1
    return labels


def getLabels(
    subdf,
    min_samples=3,
    eps=0.4,
    dis_penalty_coef=0.02,
    dis_penalty_cutoff=2.0,
    cutoff=None,
    enlarge=0.0,
    use_myscan=False,
):
    print(min_samples, eps, dis_penalty_coef, dis_penalty_cutoff)
    # print("df len", len(subdf))
    if min_samples is None and cutoff is None:
        raise ValueError("Either min_samples or cutoff must be provided.")

    if min_samples is None:
        min_samples = cutoff + 1

    x = subdf[["x", "y", "w", "h"]].values
    if enlarge > 0.0:
        x[:, 2:] += enlarge
    z_values = subdf["z"].to_numpy(dtype=np.int32, copy=False)
    iou = utils.get_iou_numpy(x).tocoo()
    upper = iou.row < iou.col
    rows = iou.row[upper]
    cols = iou.col[upper]
    iou_values = iou.data[upper].astype(np.float32, copy=False)

    z_diff = np.abs(z_values[rows] - z_values[cols]).astype(np.float32)
    z_limit = cutoff if cutoff is not None else dis_penalty_cutoff
    keep = z_diff <= z_limit

    if cutoff is None:
        distances = (1.0 - iou_values) + z_diff * dis_penalty_coef
    else:
        distances = 1.0 - iou_values

    tiny = np.finfo(np.float32).tiny
    distances = np.maximum(distances, tiny)
    keep &= distances <= eps

    rows = rows[keep]
    cols = cols[keep]
    distances = distances[keep]

    diag = np.arange(len(subdf), dtype=np.int64)
    diag_dist = np.full(len(subdf), tiny, dtype=np.float32)
    sym_rows = np.concatenate([rows, cols, diag])
    sym_cols = np.concatenate([cols, rows, diag])
    sym_distances = np.concatenate([distances, distances, diag_dist])

    distance_graph = csr_matrix(
        (sym_distances, (sym_rows, sym_cols)), shape=(len(subdf), len(subdf))
    )
    distance_graph = sort_graph_by_row_values(
        distance_graph, warn_when_not_sorted=False
    )
    hdb = DBSCAN(min_samples=min_samples, eps=eps, metric="precomputed")
    if use_myscan:
        dense = distance_graph.toarray()
        return myscan(dense, z_values, eps=eps, max_z_diff=dis_penalty_cutoff)
    hdb.fit(distance_graph)
    return hdb.labels_
    # subdf["label"] = hdb.labels_


# np.unique(hdb.labels_).tolist()


def _pairwise_iou_xywh(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ax1 = boxes_a[:, 0] - boxes_a[:, 2] / 2.0
    ay1 = boxes_a[:, 1] - boxes_a[:, 3] / 2.0
    ax2 = boxes_a[:, 0] + boxes_a[:, 2] / 2.0
    ay2 = boxes_a[:, 1] + boxes_a[:, 3] / 2.0

    bx1 = boxes_b[:, 0] - boxes_b[:, 2] / 2.0
    by1 = boxes_b[:, 1] - boxes_b[:, 3] / 2.0
    bx2 = boxes_b[:, 0] + boxes_b[:, 2] / 2.0
    by2 = boxes_b[:, 1] + boxes_b[:, 3] / 2.0

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a[:, None] + area_b[None, :] - inter_area
    union = np.maximum(union, 1e-8)
    return inter_area / union


def _build_sparse_edges_by_z(boxes, z, cutoff=5, iou_thres=0.3):
    if len(boxes) == 0:
        return np.empty((0, 2), dtype=np.int64)

    cutoff = int(cutoff)
    if cutoff < 1:
        return np.empty((0, 2), dtype=np.int64)

    z = z.astype(int)
    z_to_ids = {}
    for idx, zi in enumerate(z):
        z_to_ids.setdefault(int(zi), []).append(idx)

    edge_rows = []
    edge_cols = []
    all_z = sorted(z_to_ids.keys())
    all_z_set = set(all_z)
    for zi in all_z:
        ids_i = np.asarray(z_to_ids[zi], dtype=np.int64)
        box_i = boxes[ids_i]
        for dz in range(1, cutoff + 1):
            zj = zi + dz
            if zj not in all_z_set:
                continue
            ids_j = np.asarray(z_to_ids[zj], dtype=np.int64)
            box_j = boxes[ids_j]
            iou_ij = _pairwise_iou_xywh(box_i, box_j)
            take_i, take_j = np.where(iou_ij >= iou_thres)
            if len(take_i) == 0:
                continue
            edge_rows.append(ids_i[take_i])
            edge_cols.append(ids_j[take_j])

    if len(edge_rows) == 0:
        return np.empty((0, 2), dtype=np.int64)

    rows = np.concatenate(edge_rows)
    cols = np.concatenate(edge_cols)
    edges = np.stack([rows, cols], axis=1)
    return edges.astype(np.int64)


def getLabelsByGraph(
    subdf, method, enlarge=0.0, cutoff=5, iou_thres=0.3, leiden_resolution=1.0
):
    x = subdf[["x", "y", "w", "h"]].to_numpy(dtype=np.float32, copy=True)
    z = subdf["z"].to_numpy(dtype=np.int32)
    if enlarge > 0.0:
        x[:, 2:] += enlarge

    method = str(method).lower()
    edges = _build_sparse_edges_by_z(x, z, cutoff=cutoff, iou_thres=iou_thres)
    n = len(subdf)

    if method in ["connected_components", "connected", "cc"]:
        if len(edges) == 0:
            return np.arange(n, dtype=np.int32)
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(rows), dtype=np.int8)
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        _, labels = connected_components(csgraph=graph, directed=False)
        return labels.astype(np.int32)

    if method == "leiden":
        if len(edges) == 0:
            return np.arange(n, dtype=np.int32)
        try:
            import igraph as ig
            import leidenalg as la
        except ImportError as e:
            raise ImportError(
                "Leiden clustering requires `python-igraph` and `leidenalg`. "
                "Install them before using method='leiden'."
            ) from e

        g = ig.Graph(n=n, edges=[tuple(x) for x in edges.tolist()], directed=False)
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            seed=0,
            resolution_parameter=leiden_resolution,
        )
        labels = np.empty(n, dtype=np.int32)
        for community_id, members in enumerate(partition):
            labels[np.asarray(members, dtype=np.int64)] = community_id
        return labels

    if method == "graph":
        if len(edges) == 0:
            return csr_matrix((n, n), dtype=np.int8)
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(rows), dtype=np.int8)
        return csr_matrix((data, (rows, cols)), shape=(n, n))

    raise ValueError(
        "Unsupported graph method. Use one of: "
        "'connected_components'/'cc', 'leiden', or 'graph'."
    )


def getLabelByDirectScan(subdf, iou_thres=0.3, enlarge=0.02, max_count=30):
    if len(subdf) == 0:
        return np.empty((0,), dtype=np.int32)

    # Preserve original index order in the returned labels.
    ordered = subdf.sort_values(by="z")
    idx = ordered.index.to_numpy()
    x = ordered[["x", "y", "w", "h"]].to_numpy(dtype=np.float32, copy=True)
    z = ordered["z"].to_numpy(dtype=np.int32)
    if enlarge > 0.0:
        x[:, 2:] += float(enlarge)

    labels_sorted = np.empty(len(ordered), dtype=np.int32)
    label_sizes = {}
    next_label = 0

    pos_by_z = {}
    for pos, zi in enumerate(z):
        pos_by_z.setdefault(int(zi), []).append(pos)

    all_z = sorted(pos_by_z.keys())
    prev_z = None
    prev_pos = np.empty((0,), dtype=np.int64)

    for zi in all_z:
        curr_pos = np.asarray(pos_by_z[zi], dtype=np.int64)
        curr_boxes = x[curr_pos]

        if prev_z is None or zi != prev_z + 1 or len(prev_pos) == 0:
            for p in curr_pos:
                labels_sorted[p] = next_label
                label_sizes[next_label] = 1
                next_label += 1
        else:
            prev_boxes = x[prev_pos]
            iou_mat = _pairwise_iou_xywh(curr_boxes, prev_boxes)

            for local_i, p in enumerate(curr_pos):
                if iou_mat.shape[1] == 0:
                    best_iou = -1.0
                    best_prev_label = None
                else:
                    best_j = int(np.argmax(iou_mat[local_i]))
                    best_iou = float(iou_mat[local_i, best_j])
                    best_prev_label = int(labels_sorted[prev_pos[best_j]])

                use_prev = (
                    best_prev_label is not None
                    and best_iou >= float(iou_thres)
                    and label_sizes.get(best_prev_label, 0) < int(max_count)
                )

                if use_prev:
                    labels_sorted[p] = best_prev_label
                    label_sizes[best_prev_label] += 1
                else:
                    labels_sorted[p] = next_label
                    label_sizes[next_label] = 1
                    next_label += 1

        prev_z = zi
        prev_pos = curr_pos

    out = np.empty(len(subdf), dtype=np.int32)
    out_index = {k: i for i, k in enumerate(subdf.index.to_numpy())}
    for p, src_idx in enumerate(idx):
        out[out_index[src_idx]] = labels_sorted[p]
    return out


def processClass(
    df,
    classname,
    min_prob,
    nms,
    max_area=None,
    min_area=None,
    cluster_params={},
    use_myscan=False,
    method="DBSCAN",
    MAX_NUM=50000,
):
    df["class_value"] = df[classname]  # * 2 - df["max"]
    subdf = df[df["class_value"] > min_prob]
    subdf = subdf[subdf["unlabeled"]]
    # print(len(subdf))
    # print(utils.convertBoxes(torch.tensor(subdf[["x", "y", "w", "h"]].values)).shape)
    alldfs = []
    for i, r in subdf.groupby("z"):
        # keep = utils.bbnms(
        #     nms,
        #     utils.convertBoxes(torch.tensor(r[["x", "y", "w", "h"]].values)),
        #     torch.tensor(r["class_value"].values),
        #     np.array(["same" for i in range(len(r))]),
        # )
        keep = torchvision.ops.nms(
            utils.convertBoxes(torch.tensor(r[["x", "y", "w", "h"]].values)),
            torch.tensor(r["class_value"].values),
            nms,
        )
        # if i == 190:
        #     print("keep", keep)
        r["bbnms"] = False
        # print(keep, list(r.columns))
        r.iloc[keep.numpy(), list(r.columns).index("bbnms")] = True
        r = r[r["bbnms"]]
        alldfs.append(r)
    if len(alldfs) == 0:
        return None
    subdf = pd.concat(alldfs)
    subdf["area"] = subdf["w"] * subdf["h"]
    if max_area is not None:
        subdf = subdf[subdf["area"] < max_area]
    if min_area is not None:
        subdf = subdf[subdf["area"] > min_area]

    if method == "DBSCAN":
        if len(subdf) > MAX_NUM:
            raise ValueError(
                f"Too many instances after filtering: received {len(subdf)} instances, raise thresholds for screening"
            )
        labels = getLabels(subdf, use_myscan=False, **cluster_params)
    elif method == "myscan":
        if len(subdf) > MAX_NUM:
            raise ValueError(
                f"Too many instances after filtering: received {len(subdf)} instances, raise thresholds for screening"
            )
        labels = getLabels(subdf, use_myscan=True, **cluster_params)
    elif method.lower() in ["connected_components", "connected", "cc", "leiden"]:
        labels = getLabelsByGraph(subdf, method=method, **cluster_params)
    elif method.lower() in ["direct_scan", "directscan", "scan"]:
        labels = getLabelByDirectScan(subdf, **cluster_params)
    else:
        raise ValueError(
            "Unsupported method. Use one of: 'DBSCAN', 'myscan', "
            "'connected_components'/'cc', 'leiden', or 'direct_scan'."
        )
    print("number of instances in class ", len(np.unique(labels)))
    subdf["label"] = list(labels)
    return subdf


def findClosestIndex(lst, target):
    lst = np.array(lst)  # Convert to NumPy array
    closest_index = np.abs(lst - target).argmin()
    return closest_index


def max_sum_subarray_position(arr, k):
    if k > len(arr):
        return None  # k is too large

    # Compute sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    max_start_index = 0

    # Slide the window
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]  # Add new element, remove old
        if window_sum > max_sum:
            max_sum = window_sum
            max_start_index = i - k + 1

    return max_start_index


def markFilter(
    df,
    classname,
    targetdf,
    max_cnt=100,
    remove_iou=0.5,
    min_samples=12,
    max_samples=500,
    min_length=12,
    extend=0,
    threshold=0.15,
):
    if targetdf is None:
        print("targetdf is None, skip")
        return df
    zpos_min = df["z"].min()
    zpos_max = df["z"].max()
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
        if len(subdf2) < min_samples:
            continue
        if len(subdf2) > max_samples:
            pos = max_sum_subarray_position(list(subdf2[classname]), max_samples)
            subdf2 = subdf2.iloc[pos : pos + max_samples]
        # df.loc[subdf2.index, "label_id"] = label
        # df.loc[subdf2.index, "label"] = classname

        zmin = subdf2["z"].min()
        zmax = subdf2["z"].max()
        zmid = (zmin + zmax) // 2

        if zmin > zmid - min_length:
            zmin = zmid - min_length
        if zmax < zmid + min_length:
            zmax = zmid + min_length

        zmin -= extend
        zmax += extend
        zmin = max(zmin, zpos_min)
        zmax = min(zmax, zpos_max)

        for i in range(zmin, zmax + 1):
            j = findClosestIndex(subdf2["z"].values, i)
            # print(i, j)
            posz_df = df[df["z"] == i]
            required_indexes = list(posz_df.index)
            required_indexes.append(subdf2.index[j])
            required_df = df.loc[required_indexes]
            iou = utils.get_iou(required_df[["x", "y", "w", "h"]].values)
            iou = iou[-1]
            iou = iou > remove_iou
            iou[-1] = False
            required_df = required_df[iou]

            df.loc[required_df.index, "unlabeled"] = False
            required_df = required_df.sort_values(by=classname, ascending=False)
            if len(required_df) == 0:
                continue
            if required_df.iloc[0][classname] > threshold:
                df.loc[required_df.index[0], "label_id"] = label
                df.loc[required_df.index[0], "label"] = classname


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


def refineMembMask(
    mask,
    threshold,
    sigma=0,
    kernal_size=3,
    morph_open_iterations=2,
    morph_close_iterations=1,
    **kwargs,
):
    if sigma > 0:
        mask = gaussian_filter(mask, sigma=sigma)
    # Step 1: Convert to binary
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # Step 2: Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernal_size))
    opened_mask = cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iterations
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal_size, 1))
    opened_mask = cv2.morphologyEx(
        opened_mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iterations
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal_size, 1))
    closed_mask = cv2.morphologyEx(
        opened_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iterations
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernal_size))
    closed_mask = cv2.morphologyEx(
        opened_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iterations
    )

    return closed_mask


def refineMask(
    mask,
    threshold=0.5,
    morph_open_kernal_size=(5, 5),
    morph_open_iterations=2,
    morph_close_kernal_size=(5, 5),
    morph_close_iterations=2,
    apply_blur=True,
    blur_ksize=(15, 15),
    blur_sigma=0,
    contour_area_threshold=50,
    max_contours=1,
    apply_contours=True,
    apply_convex_hull=False,
    **kwargs,
):
    # Step 1: Convert to binary
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    # _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if np.sum(binary_mask) < 5:
        return binary_mask

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
    if apply_blur:
        blurred_mask = cv2.GaussianBlur(closed_mask, blur_ksize, blur_sigma)
    else:
        blurred_mask = closed_mask

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


# get the embeddings and the boxes both from GNN
# make sure the model is trained in the correct way to use this function
def getMaskInputsStage2(
    df,
    dataset,
    min_length,
    extend,
    res=None,
    zpos_min=0,
    zpos_max=500,
    iou_thres=0.5,
    class_id=None,
):
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
        inputs[i]["embed"] = res["embeddings"][q].detach().cpu().unsqueeze(0)

        # boxes = stage1_outputs[i]["pred_boxes"]
        # boxes = np.concatenate([inputs[i]["bboxes"], boxes], 0)
        # iou = utils.get_iou(boxes)
        # iou = iou[0, 1:]
        # score = stage1_outputs[i]["prob"][:, class_id]
        # score[iou < iou_thres] -= 1.0
        # score = score + iou
        # take_id = np.argmax(score)

        # inputs[i]["embed"] = res["embeddings"][q].detach().cpu().unsqueeze(0)
        # inputs[i]["embed"] = stage1_outputs[i]["embeddings"][[take_id]]
        # print(data["pixel_values"].shape)
        inputs[i]["input_mask"] = xywh2mask(
            inputs[i]["bboxes"].squeeze().numpy(),
            (inputs[i]["image"].shape[2], inputs[i]["image"].shape[3]),
        )
        # print(sum(inputs[i]["input_mask"].flatten()))

    return inputs


# def getMaskInputFromStage1()


# the default version
# the box comes from GNN, but the embeds comes from Detr.
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
    box_stage_1=False,
    expand=0.0,
):
    model.stage = "stage 1"
    df = df.sort_values(by="z")
    zmin = df["z"].min()
    zmax = df["z"].max()
    zmid = (zmin + zmax) // 2
    if zmin > zmid - min_length:
        zmin = zmid - min_length
    if zmax < zmid + min_length:
        zmax = zmid + min_length
    zmin -= extend
    zmax += extend
    zmin = max(zmin, zpos_min)
    zmax = min(zmax, zpos_max - 1)
    inputs = {}
    # print("get output")
    for i in range(zmin, zmax + 1):
        inputs[i] = {}

        if i not in stage1_outputs:
            with torch.no_grad():
                data = dataset.__getitem__(pos=i)
                data["pixel_values"] = (
                    data["pixel_values"].unsqueeze(0).float().to(model.device)
                )
                data["pixel_mask"] = (
                    data["pixel_mask"].unsqueeze(0).float().to(model.device)
                )
                # print("get input image")
                outputs = model(
                    pixel_values=data["pixel_values"],
                    pixel_mask=data["pixel_mask"],
                )

            stage1_outputs[i] = {}
            stage1_outputs[i]["pixel_values"] = data["pixel_values"].cpu()
            stage1_outputs[i]["pred_boxes"] = outputs["pred_boxes"].cpu().squeeze()
            logits = outputs["logits"].squeeze(0)
            prob = torch.sigmoid(logits)
            stage1_outputs[i]["prob"] = prob.cpu().numpy()
            stage1_outputs[i]["embeddings"] = (
                outputs["last_hidden_state"].cpu().squeeze(0)
            )
        # print("get inputs")
        # data = dataset.__getitem__(pos=i)
        # t = data["pixel_values"].unsqueeze(0)
        inputs[i]["image"] = stage1_outputs[i]["pixel_values"]
        q = findClosestIndex(df["z"].values, i)
        inputs[i]["bboxes"] = torch.tensor(
            df.iloc[q][["x", "y", "w", "h"]].astype(float).values
        ).unsqueeze(0)
        if expand > 0.0:
            inputs[i]["bboxes"] = inputs[i]["bboxes"].clone()
            inputs[i]["bboxes"][:, 2:] += expand
            inputs[i]["bboxes"][:, 2:] = np.minimum(inputs[i]["bboxes"][:, 2:], 1.0)
            inputs[i]["bboxes"][:, 2:] = np.maximum(inputs[i]["bboxes"][:, 2:], 0.0)
        q = df.index[q]
        boxes = stage1_outputs[i]["pred_boxes"]
        # print(boxes.shape, inputs[i]["bboxes"].shape)
        boxes = np.concatenate([inputs[i]["bboxes"], boxes], 0)
        # print("get iou")
        iou = utils.get_iou(boxes)
        iou = iou[0, 1:]
        score = stage1_outputs[i]["prob"][:, class_id]
        score[iou < iou_thres] -= 1.0
        score = score + iou

        take_id = np.argmax(score)

        if box_stage_1:
            inputs[i]["bboxes"] = stage1_outputs[i]["pred_boxes"][[take_id]]

        # inputs[i]["embed"] = res["embeddings"][q].detach().cpu().unsqueeze(0)
        inputs[i]["embed"] = stage1_outputs[i]["embeddings"][[take_id]]
        # print(data["pixel_values"].shape)
        inputs[i]["input_mask"] = xywh2mask(
            inputs[i]["bboxes"].squeeze().numpy(),
            (inputs[i]["image"].shape[2], inputs[i]["image"].shape[3]),
        )
        # print(sum(inputs[i]["input_mask"].flatten()))

    return inputs, stage1_outputs


def getMasks(model, inputs, sigma=1, on_z_only=False):
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
    if sigma > 0.0:
        if on_z_only:
            aligned_masks = gaussian_filter1d(aligned_masks, sigma=sigma, axis=0)
        else:
            aligned_masks = gaussian_filter(aligned_masks, sigma=sigma)
    return aligned_masks


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


def getPredictionCenters(df, classname, top_n=None, image_size=1024, thres=None):
    subdf = df[df["label"] == classname]
    subdf = subdf[subdf["label_id"] != -1]
    score = []
    predict_center = []
    ids = []
    for i, subdf2 in subdf.groupby("label_id"):
        value = subdf2[classname].values
        if thres is not None:
            value = value[value > thres]
        if top_n is None:
            score.append(np.sum(value))
        else:
            score.append(np.sum(-np.sort(-value)[:top_n]))
        predict_center.append(np.mean(subdf2[["z", "y", "x"]].values, axis=0))
        ids.append(i)
    ids = np.array(ids)
    score = np.array(score)
    predict_center = np.array(predict_center)
    if predict_center.size == 0:
        predict_center = np.empty((0, 3), dtype=float)
        return score, predict_center, ids
    predict_center *= np.array([1, image_size, image_size])
    return score, predict_center, ids


def match_and_find_closest(pred_centers, label_centers):
    """
    Matches predicted centers to label centers using bipartite matching (Hungarian algorithm).
    Also finds the closest prediction for each label.

    Args:
        pred_centers: np.ndarray of shape (N_pred, 3) — predicted (z, y, x) coordinates
        label_centers: np.ndarray of shape (N_label, 3) — ground truth (z, y, x) coordinates

    Returns:
        match_pairs: list of (label_index, pred_index)
        match_distances: np.ndarray of distances for the matched pairs
        closest_preds: np.ndarray of shape (N_label, 2), where each row = (closest_pred_index, distance)
    """
    pred_centers = np.asarray(pred_centers)
    label_centers = np.asarray(label_centers)

    if label_centers.size == 0:
        return [], np.array([], dtype=float), np.empty((0, 2), dtype=float)

    if pred_centers.size == 0:
        closest_preds = np.stack(
            [
                np.full(len(label_centers), -1, dtype=int),
                np.full(len(label_centers), np.inf, dtype=float),
            ],
            axis=1,
        )
        return [], np.array([], dtype=float), closest_preds

    # Compute distance matrix between labels and predictions
    dist_matrix = cdist(label_centers, pred_centers, metric="euclidean")

    # --- Bipartite matching (Hungarian algorithm) ---
    label_idx, pred_idx = linear_sum_assignment(dist_matrix)
    match_pairs = list(zip(label_idx, pred_idx))
    match_distances = dist_matrix[label_idx, pred_idx]

    # --- Closest prediction for each label ---
    closest_pred_indices = np.argmin(dist_matrix, axis=1)
    closest_pred_distances = dist_matrix[
        np.arange(len(label_centers)), closest_pred_indices
    ]
    closest_preds = np.stack(
        [closest_pred_indices, closest_pred_distances], axis=1
    )  # (N_label, 2)

    return match_pairs, match_distances, closest_preds


def get_labels_and_centers(label_matrix, labels=None):
    """
    Find all object labels and their centers in a 3D label matrix.

    Args:
        label_matrix: 3D NumPy array with 0 = background, 1,2,... = objects

    Returns:
        labels_1d: 1D NumPy array of object labels
        centers_2d: 2D NumPy array of shape (n_objects, 3)
                    with (z, y, x) coordinates for each object
    """
    # label_matrix = np.asarray(label_matrix)
    if labels is None:
        labels_id = np.unique(label_matrix)
        labels_id = labels_id[labels_id != 0]  # remove background
    else:
        labels_id = np.asarray(labels)

    centers_list = [
        center_of_mass(label_matrix == label_val) for label_val in labels_id
    ]

    centers_2d = np.array(centers_list)  # shape (n_objects, 3)

    return labels_id, centers_2d


def calculate_metrics(match_pairs, match_distances, y_pred_prob, threshold=20):
    """
    Calculate AUROC, precision, and recall for binary classification.

    Args:
        y_true: array-like of shape (n_samples,) — ground truth labels (0 or 1)
        y_pred_prob: array-like of shape (n_samples,) — predicted probabilities
        threshold: float — probability threshold for converting to binary predictions

    Returns:
        auc_score: float — AUROC score
        precision: float — Precision score
        recall: float — Recall score
    """
    y_pred_prob = np.asarray(y_pred_prob)
    if y_pred_prob.size == 0:
        return {
            "auroc": float("nan"),
            "aupr": 0.0,
            "aupr2": 0.0,
            "cnts": 0,
        }

    matches = np.array(match_pairs, dtype=int)
    find_matches = np.array(match_distances) < threshold
    score_label = np.zeros(len(y_pred_prob), dtype=int)
    if matches.size > 0:
        matched = matches[find_matches]
        if matched.size > 0:
            score_label[matched[:, 1]] = 1

    precision, recall, _ = precision_recall_curve(score_label.astype(int), y_pred_prob)
    aupr = auc(recall, precision)

    aupr2 = average_precision_score(score_label.astype(int), y_pred_prob)

    try:
        auroc = roc_auc_score(score_label.astype(int), y_pred_prob)
    except ValueError:
        auroc = float("nan")

    return {
        "auroc": auroc,
        "aupr": aupr,
        "aupr2": aupr2,
        "cnts": sum(find_matches),
        # "precision": precision_score(score_label.astype(int), y_pred_prob > 0.5),
        # "recall": recall_score(score_label.astype(int), y_pred_prob > 0.5),
    }

    return roc_auc_score(score_label.astype(int), y_pred_prob), sum(find_matches)


def calc_iou(pred_mask, true_mask):
    # Ensure binary masks (0 or 1)
    pred_mask = pred_mask > 0  # .float()
    true_mask = true_mask > 0  # .float()

    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask) - intersection

    if union == 0:
        return float("nan")  # Handle no-object case
    return (intersection / union).item()
