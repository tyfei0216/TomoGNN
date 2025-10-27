import sys

sys.path.append("/home/feity/cryoem/")
import importlib
import os
import pickle

import numpy as np
import torch
from scipy.ndimage import center_of_mass
from scipy.sparse import csr_matrix
from tqdm import tqdm

import data
import postprocess
import utils

importlib.reload(data)
import os

# l = [
#     "24092002_9.62Apx.mrc",
#     "24092013_9.62Apx.mrc",
#     "24092016_9.62Apx.mrc",
#     "24092020_9.62Apx.mrc",
#     "24092024_9.62Apx.mrc",
#     "24092026_9.62Apx.mrc",
#     "24092030_9.62Apx.mrc",
#     "24092009_9.62Apx.mrc",
#     "24092014_9.62Apx.mrc",
#     "24092017_9.62Apx.mrc",
#     "24092023_9.62Apx.mrc",
#     "24092025_9.62Apx.mrc",
#     "24092029_9.62Apx.mrc",
#     "24092031_9.62Apx.mrc",
# ]

# l = [
#     "24092009_9.62Apx.mrc",
#     "24092014_9.62Apx.mrc",
#     "24092017_9.62Apx.mrc",
#     "24092023_9.62Apx.mrc",
#     "24092025_9.62Apx.mrc",
#     "24092029_9.62Apx.mrc",
#     "24092031_9.62Apx.mrc",
# ]

l = [
    "24092002_9.62Apx.mrc",
    "24092013_9.62Apx.mrc",
    "24092016_9.62Apx.mrc",
    "24092020_9.62Apx.mrc",
    "24092024_9.62Apx.mrc",
    "24092026_9.62Apx.mrc",
    "24092030_9.62Apx.mrc",
]

paths = [
    "/data/transformer_project/transforemer_model/train_data/training/results/conditional_detr_mrc_clean/fold2_ribo",
    "/data/transformer_project/transforemer_model/train_data/training/results/conditional_detr_mrc_clean/fold2_ribo",
    "/data/transformer_project/transforemer_model/train_data/training/results/conditional_detr_mrc_clean/fold2_ribo",
    "/data/transformer_project/transforemer_model/train_data/training/results/conditional_detr_mrc_clean/fold2_ribo",
    "/data/transformer_project/transforemer_model/train_data/training/results/conditional_detr_mrc_clean/fold2_ribo",
    "/data/transformer_project/transforemer_model/train_data/training/results/conditional_detr_mrc_clean/fold2_ribo",
    "/data/transformer_project/transforemer_model/train_data/training/results/conditional_detr_mrc_clean/fold2_ribo",
]

models = [
    "stage2/last.ckpt",
    "stage2/last.ckpt",
    "stage2/last.ckpt",
    "stage2/last.ckpt",
    "stage2/last.ckpt",
    "stage2/last.ckpt",
    "stage2/last.ckpt",
]

params = {
    "endoplasmic_reticulum_or_Golgi": {
        "min_prob": 0.5,
        "nms": 0.6,
        "dbscan_prams": {
            "min_samples": 20,
            "eps": 0.5,
            "dis_penalty_coef": 1.0 * (1 / 500),
            "dis_penalty_cutoff": 20,
        },
    },
    "mitochondria": {
        "min_prob": 0.4,
        "nms": 0.2,
        "dbscan_prams": {
            "min_samples": 25,
            "eps": 0.5,
            "dis_penalty_coef": 1.0 * (1 / 500),
            "dis_penalty_cutoff": 20,
        },
    },
    "nucleus": {
        "min_prob": 0.4,
        "nms": 0.2,
        "dbscan_prams": {
            "min_samples": 25,
            "eps": 0.5,
            "dis_penalty_coef": 1.0 * (1 / 500),
            "dis_penalty_cutoff": 20,
        },
    },
    "ribo": {
        "min_prob": 0.4,
        "nms": 0.2,
        "max_area": 0.05 * 0.05,
        "dbscan_prams": {
            "min_samples": 6,
            "eps": 0.5,
            "dis_penalty_coef": 2.0 * (1 / 500),
            "dis_penalty_cutoff": 5,
        },
    },
    "hsp60": {
        "min_prob": 0.5,
        "nms": 0.2,
        "max_area": 0.03 * 0.03,
        "dbscan_prams": {
            "min_samples": 4,
            "eps": 0.5,
            "dis_penalty_coef": 2.0 * (1 / 500),
            "dis_penalty_cutoff": 3,
            "enlarge": 0.03,
        },
    },
}

mark_and_filter = {
    "nucleus": {"max_cnt": 1, "remove_iou": 0.4, "min_samples": 20},
    "mitochondria": {"max_cnt": 10, "remove_iou": 0.6, "min_samples": 20},
    "endoplasmic_reticulum_or_Golgi": {
        "max_cnt": 20,
        "remove_iou": 0.8,
        "min_samples": 20,
    },
    "ribo": {
        "max_cnt": 80000,
        "remove_iou": 0.5,
        "min_samples": 12,
        "min_length": 16,
        "extend": 0,
        "max_samples": 30,
    },
    "hsp60": {
        "max_cnt": 5000,
        "remove_iou": 0.5,
        "min_samples": 5,
        "min_length": 10,
        "extend": 0,
        "max_samples": 30,
    },
}

mask_thres = {
    "ribo": 0.75,
    "hsp60": 0.75,
    "mitochondria": 0.6,
    "nucleus": 0.6,
    "endoplasmic_reticulum_or_Golgi": 0.6,
}
prepare_mask_input = {
    "nucleus": {"class_id": 2, "min_length": 50, "extend": 10},
    "mitochondria": {"class_id": 1, "min_length": 40, "extend": 5},
    "endoplasmic_reticulum_or_Golgi": {"class_id": 3, "min_length": 30, "extend": 5},
    "ribo": {"class_id": 0, "min_length": 12, "extend": 0},
    "hsp60": {"class_id": 0, "min_length": 12, "extend": 0},
}
refine_mask = {
    "nucleus": {
        "smooth": 2.0,
        "morph_open_kernal_size": (7, 7),
        "morph_close_kernal_size": (7, 7),
        "blur_ksize": (25, 25),
        "max_contours": 1,
        "apply_convex_hull": True,
    },
    "mitochondria": {
        "smooth": 2.0,
        "morph_open_kernal_size": (5, 5),
        "morph_close_kernal_size": (5, 5),
        "blur_ksize": (15, 15),
        "max_contours": 1,
        "apply_convex_hull": False,
    },
    "endoplasmic_reticulum_or_Golgi": {
        "smooth": 1.0,
        "morph_open_kernal_size": (5, 5),
        "morph_close_kernal_size": (5, 5),
        "blur_ksize": (11, 11),
        "max_contours": 1,
        "contour_area_threshold": 250,
        "apply_convex_hull": False,
    },
    "ribo": {
        "smooth": 0.5,
        "morph_open_kernal_size": (3, 3),
        "morph_close_kernal_size": (3, 3),
        "blur_ksize": (3, 3),
        "max_contours": 1,
        "contour_area_threshold": 10,
        "apply_convex_hull": False,
    },
    "hsp60": {
        "smooth": 0.5,
        "morph_open_kernal_size": (3, 3),
        "morph_close_kernal_size": (3, 3),
        "blur_ksize": (3, 3),
        "max_contours": 1,
        "contour_area_threshold": 10,
        "apply_convex_hull": False,
    },
}


# model = utils.loadModel(
#     "/data/transformer_project/transforemer_model/train_data/training/results/conditional_detr_mrc_clean/whole_ribo_gap1",
#     "stage3/epoch=5-total_validate_loss=0.0273.ckpt",
# )

# model = utils.loadModel(
#     "/data/transformer_project/transforemer_model/train_data/training/results/conditional_mrc_hsp60/test2",
#     "stage3/epoch=6-total_validate_loss=0.0343.ckpt",
# )


for filename, path, model in zip(l, paths, models):

    model = utils.loadModel(
        path,
        model,
    )

    print(filename)

    model = model.cpu()
    # if os.path.exists(
    #     f"/home/feity/cryoem/temp/results/fold_hsp60_{filename[: len('24092002')]}.pkl"
    # ):
    #     print("exists, skip")
    #     continue

    dataset = data.TestDatasetMrc(
        # "/data/transformer_project/transforemer_model/test_data/dxd20240920_hsp60/24092001.pkl",
        "/data/transformer_project/transforemer_model/test_data/dxd20240920/tomo2/"
        + filename,
        norm="hist",
        reshape=800,
        length_for_average=3,
        gap=1,
    )

    fn = filename[: len("24092002")]
    # with open(f"/home/feity/cryoem/temp/results/{fn}_9_gap1_new.pkl", "rb") as f:
    #     df, graphs2 = pickle.load(f)

    df, graph = postprocess.generatedf(
        model, dataset, gap=1, columns=["ribo", "None"], filter_prob=0.2
    )
    df["unlabeled"] = True
    df["label"] = ""
    df["label_id"] = -1
    subdf = postprocess.processClass(df, "ribo", **params["ribo"], use_myscan=False)
    # print(subdf.head())
    postprocess.markFilter(df, "ribo", subdf, **mark_and_filter["ribo"])

    stage1_output = {}
    build_masks = {
        "ribo": np.zeros((500, 800, 800), dtype=int),
        "hsp60": np.zeros((500, 800, 800), dtype=int),
        "mitochondria": np.zeros((500, 800, 800), dtype=int),
        "nucleus": np.zeros((500, 800, 800), dtype=int),
        "endoplasmic_reticulum_or_Golgi": np.zeros((500, 800, 800), dtype=int),
    }
    all_inputs = {}
    label_center = {}
    model = model.cuda(0)
    model.mask_head.TF = 1.0
    for (label, label_id), r in tqdm(df.groupby(["label", "label_id"])):
        if label != "ribo":
            continue
        # if label == "":
        #     continue
        if label_id == -1:
            continue
        # print(label, label_id, len(r))
        # if label == "ribo":
        #     continue
        if "%s_%s" % (label, label_id) in all_inputs:
            continue
        inputs, stage1_output = postprocess.getMaskInputs(
            r, model, dataset, **prepare_mask_input[label], stage1_outputs=stage1_output
        )
        # inputs = postprocess.getMaskInputsStage2(
        #     r,
        #     dataset,
        #     **prepare_mask_input[label],
        #     res = res
        # )
        all_inputs["%s_%s" % (label, label_id)] = inputs
        # print("before get mask")
        smoothed_masks = postprocess.getMasks(
            model, inputs, refine_mask[label]["smooth"], on_z_only=True
        )
        # print("after get mask")

        minz = np.min(list(inputs.keys()))
        maxz = np.max(list(inputs.keys()))
        t = np.max(smoothed_masks)
        # masks = []
        # print("before refine")
        for i in range(len(smoothed_masks)):
            mask = smoothed_masks[i]
            # box = inputs[i + minz]["input_mask"]
            # t1 = (mask * box).sum() / box.sum()
            # t2 = (mask * (1 - box)).sum() / (1 - box).sum()
            thres = t * mask_thres[label]
            refined_mask = postprocess.refineMask(
                mask,
                threshold=thres,
                **refine_mask[label],
            )
            smoothed_masks[i] = refined_mask
        # print("after refine")
        smoothed_masks = smoothed_masks.astype(np.int32)
        # print("cm")
        cm = center_of_mass(smoothed_masks > 0.5)
        cm = np.array(cm)
        # print("after cm")
        cm[0] += minz
        label_center[label_id] = cm

        for i in range(minz, maxz + 1):
            build_masks[label][i][smoothed_masks[i - minz] > 1] = label_id + 1
    mask = {}
    # res["masks"].shape
    for i in range(500):
        mask[i] = csr_matrix(build_masks["ribo"][i])
    with open(f"/home/feity/cryoem/temp/results/ribo_{fn}_gap1.pkl", "wb") as f:
        pickle.dump(
            {
                "centers": label_center,
                "df": df,
                "masks": mask,
                "graph": graph[0],
                # "inputs": all_inputs
            },
            f,
        )
