import json
import multiprocessing
import os
import pickle
import random
import threading
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pycocotools
import pytorch_lightning as L
import torch
import torchvision.datasets
import torchvision.transforms.v2 as transforms
from PIL import Image
from pycocotools.coco import COCO
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from skimage import exposure
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as Gdataset
from torchvision.tv_tensors import BoundingBoxes, Mask

import utils


class MyCOCO(COCO):
    """
    rewrite the COCO class to support loading annotations from a json file or a pickle file.
    the original COCO class only supports loading from a json file.
    Args:
        annotation_file (str): Path to the annotation file (json or pickle).
    """

    def __init__(self, annotation_file=None):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print("loading annotations into memory...")
            tic = time.time()
            if annotation_file.endswith(".json"):
                with open(annotation_file, "r") as f:
                    dataset = json.load(f)
            elif annotation_file.endswith(".pkl"):
                import pickle

                with open(annotation_file, "rb") as f:
                    dataset = pickle.load(f)
            else:
                raise NotImplementedError
            assert (
                type(dataset) == dict
            ), "annotation file format {} not supported".format(type(dataset))
            print("Done (t={:0.2f}s)".format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()


class stage2Dataset(Gdataset):
    """
    A dataset class for stage 2 of the training process.
    This class only loads stage 2 graph for GNN training only.
    In our final training process, this is not used.
    Only used for comparing with training stage 1 and stage 2 together.
    Args:
        dataset_list (list): List of datasets to combine.
        dataset_len (int): the number of samples in a epoch.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, dataset_list, dataset_len, train_val=None, seed=1013):
        super().__init__()
        self.dataset_list = []
        self.dataset_len = dataset_len
        # split the dataset into train and validation sets
        random.seed(seed)
        for i in dataset_list:
            a = list(range(i.x.shape[0]))

            if not hasattr(i, "train_mask"):

                train_mask = torch.ones(len(a), dtype=torch.bool)
                if train_val is not None:
                    val = random.sample(a, int(len(a) * train_val[1]))
                    train_mask[val] = False
                val_mask = ~train_mask
                i.train_mask = train_mask
                i.val_mask = val_mask
            # if hasattr(i, "edge_label"):
            #     edge_train_mask = torch.zeros(i.edge_label.shape[0], dtype=torch.bool)
            #     train = random.sample(
            #         list(range(i.edge_label.shape[0])), int(len(a) * train_val[0])
            #     )
            #     edge_train_mask[train] = True
            #     i.edge_train_mask = edge_train_mask
            #     i.edge_val_mask = ~edge_train_mask
            self.dataset_list.append(i)

    def len(self):
        return self.dataset_len

    def get(self, idx):
        t = idx % len(self.dataset_list)
        return self.dataset_list[t]


class stage2DataModule(L.LightningDataModule):
    """
    pytorch lightning datamodule for stage 2 training.

    """

    def __init__(
        self,
        dataset_list_train,
        dataset_list_val,
        dataset_len,
        ifaug=True,
    ):
        super().__init__()

        self.dataset_train = dataset_list_train
        self.dataset_val = dataset_list_val
        self.dataset_len = dataset_len
        self.ifaug = ifaug
        if ifaug:
            print("use node augmentation")

    def train_dataloader(self):
        # ds = stage2Dataset(self.dataset, self.dataset_len[0], self.train_val, self.seed)
        ds = stage2Dataset(self.dataset_train, self.dataset_len[0])
        return torch.utils.data.DataLoader(
            ds, batch_size=1, collate_fn=collect_graph, num_workers=8
        )

    def val_dataloader(self):
        # ds = stage2Dataset(self.dataset, self.dataset_len[1], self.train_val, self.seed)
        ds = stage2Dataset(self.dataset_val, self.dataset_len[1])
        return torch.utils.data.DataLoader(
            ds, batch_size=1, collate_fn=collect_graph, num_workers=1
        )


class MaskDataset(Dataset):
    """
    Mask Dataset for further training masks after stage 1 and stage 2.
    Although training masks is involved in stage 1 + 2, we still need to
    train masks futher since this is a more difficult task.
    Args:
        input_image_list (list): List of input images.
        input_embed (torch.Tensor): Input embeddings, from stage 2 GNN.
        target_masks (torch.Tensor): Target masks for training.
        sample_mapping (list): Mapping from mask indices to the original image.
        boxes (torch.Tensor, optional): Bounding boxes for the images. This really helps mask training.
    """

    def __init__(
        self, input_image_list, input_embed, target_masks, sample_mapping, boxes
    ):
        self.input_image_list = input_image_list
        self.input_embed = input_embed
        self.target_mask = target_masks
        self.sample_mapping = sample_mapping
        self.boxes = boxes

    def __len__(self):
        return self.target_mask.shape[0]

    def __getitem__(self, index):
        if self.boxes is None:
            return (
                self.input_image_list[self.sample_mapping[index]],
                self.input_embed[index],
                self.target_mask[index],
            )
        else:
            return (
                self.input_image_list[self.sample_mapping[index]],
                self.input_embed[index],
                self.target_mask[index],
                self.boxes[index],
            )


class MaskDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for mask training.
    Handles splitting the dataset into training and validation sets.

    Args:
        input_image_list (list): List of input images.
        input_embed (torch.Tensor): Input embeddings, from stage 2 GNN.
        target_masks (torch.Tensor): Target masks for training.
        sample_mapping (list): Mapping from mask indices to the original image.
        boxes (torch.Tensor, optional): Bounding boxes for the images.
        batch_size (int): Batch size for training and validation loaders.
        train_val (list): Proportion of training and validation split.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        input_image_list,
        input_embed,
        target_masks,
        sample_mapping,
        boxes=None,
        batch_size=2,
        train_val=[0.8, 0.2],
        seed=1013,
    ):
        super().__init__()
        self.dataset = MaskDataset(
            input_image_list, input_embed, target_masks, sample_mapping, boxes
        )
        torch.manual_seed(seed)
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, train_val
        )
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size)


class CocoDetection(torchvision.datasets.vision.VisionDataset):
    """
    main dataset class for COCO detection.
    This class handles loading images and annotations from COCO dataset.
    It alsp handles filtering classes, transforming images and data augmentation.


    Args:
        image_directory_path (str): Path to the directory containing images.
        annotation_file_path (str): Path to the COCO annotation file (json or pickle).
        is_npy (bool): Whether the images are stored as numpy arrays. Default is True.
        require_mask (bool): Whether to require masks in the annotations. Default is False.


        filter_class (list): List of classes to filter. Default is None.
        single_class (bool): Whether to treat all filtered classes as a single class. Default is False.
        map_class (dict): Mapping of class IDs to new class IDs. Default is None.
        The above three parameters are used to filter and manipulate the classes in the dataset.

        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        add_classname (bool): Whether to add class names to the annotations. Default is False.

        maxsize (int): Maximum size of the image after resizing. Default is 800.
        Our model is trained on images with size of 800x800.

        norm (str): Normalization method for images, can be 'none', 'zscore', or 'hist'. Default is 'none'.
        filtermin (int): Minimum number of annotations required for an image to be included in the dataset. Default is 5.
    """

    def __init__(
        self,
        image_directory_path: str,
        annotation_file_path: str,
        is_npy=True,
        require_mask=False,
        filter_class=None,
        single_class=False,
        map_class=None,
        transform=None,
        add_classname=False,
        maxsize=800,
        norm="none",
        filtermin=2,
    ):
        # annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super().__init__(image_directory_path)
        # super().__init__(image_directory_path, annotation_file_path)
        self.coco = MyCOCO(annotation_file_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.is_npy = is_npy
        self.require_mask = require_mask
        self.filter_class = filter_class
        self.single_class = single_class
        if map_class is not None:
            mc = {}
            for i, j in map_class.items():
                mc[int(i)] = j
        else:
            mc = None
        self.map_class = mc

        self.transform = transform
        self.classes = pd.DataFrame(self.coco.dataset["categories"])
        self.classes = self.classes.set_index("id")
        self.add_classname = add_classname
        self.maxsize = maxsize
        self.norm = norm

        self.filtermin = filtermin
        self._filterIds()

        # self.seed = None

        self.lock = threading.Lock()
        # self.zpos = []

    def _load_image(self, id: int):
        if self.is_npy:
            path = self.coco.loadImgs(id)[0]["file_name"]
            # print(self.root, path)
            path = os.path.join(self.root, path)
            return np.load(path, allow_pickle=True)
        else:
            path = self.coco.loadImgs(id)[0]["file_name"]
            return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int):
        if "zpos" in self.coco.loadImgs(id)[0]:
            pos = self.coco.loadImgs(id)[0]["zpos"]
        else:
            pos = 0
        t = self.coco.loadAnns(self.coco.getAnnIds(id))

        if self.filter_class is not None:
            t = list(filter(lambda x: x["category_id"] in self.filter_class, t))

        if len(t) < self.filtermin:
            return [], 0
        return t, pos

    def processAnnotations(self, annotations, image, require_mask=True):
        labels = torch.tensor(
            [sample["category_id"] for sample in annotations], dtype=torch.long
        )

        if self.add_classname:
            names = []
            for i in labels:
                names.append(self.classes.loc[i.item()]["name"])

        if self.single_class:
            assert self.map_class is None
            labels = torch.zeros_like(labels)

        if self.map_class is not None:
            labels = torch.tensor(
                [self.map_class[sample["category_id"]] for sample in annotations],
                dtype=torch.long,
            )

        # labels = [str(sample["category_id"]) for sample in annotations]
        bbboxes = torch.stack(
            [torch.tensor(mask["bbox"]) for mask in annotations], dim=0
        )

        bt = torchvision.ops.box_convert(torch.Tensor(bbboxes), "xywh", "xyxy")
        boxes = BoundingBoxes(bt, format="xyxy", canvas_size=image.shape[1:])

        retdict = {"bboxes": boxes, "class_labels": labels}

        if "item_id" in annotations[0]:
            item_id = [sample["item_id"] for sample in annotations]
            retdict["item_id"] = item_id

        if self.require_mask and require_mask:
            masks = torch.stack(
                [
                    torch.tensor(
                        pycocotools.mask.decode(mask["segmentation"]), dtype=torch.bool
                    )
                    for mask in annotations
                ],
                dim=0,
            )
            retdict["masks"] = Mask(masks)
        #     return {"masks": Mask(masks), "bboxes": boxes, "class_labels": labels}
        # else:
        #     return {"bboxes": boxes, "class_labels": labels}
        if self.add_classname:
            retdict["names"] = names
        return retdict

    def __len__(self):
        return len(self.needids)

    def _filterIds(self):
        needids = []
        zposes = []
        for i in range(len(self.ids)):
            annotation, zpos = self._load_target(self.ids[i])
            if len(annotation) >= self.filtermin:
                needids.append(i)
                zposes.append(zpos)
            # if len(self.coco.loadAnns(self.coco.getAnnIds(i))) > 0:
            #     need.append(i)
        self.needids = needids
        self.zpos = zposes
        # print(needids)

    def _getitem(self, index):
        if not isinstance(index, int):
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead."
            )

        id = self.ids[index]
        image = self._load_image(id)
        if self.norm == "zscore":
            for i in range(3):
                image[i] = (image[i] - image[i].mean()) / image[i].std()
        elif self.norm == "hist":
            for i in range(3):
                image[i] = exposure.equalize_hist(image[i])
        target, _ = self._load_target(id)

        return image, target

    def __getitem__(self, idx, seed=None, require_mask=True):
        # idx1 = idx
        # print("call item")
        if self.needids is not None:
            zpos = self.zpos[idx]
            idx = self.needids[idx]

        image, annotation = self._getitem(idx)

        # print(idx1, idx, len(annotation))
        if len(annotation) == 0:
            raise ValueError
        image = torch.tensor(image)
        target = self.processAnnotations(annotation, image, require_mask=require_mask)

        ori_class_labels = target["class_labels"].clone()
        target["class_labels"] = torch.tensor(
            range(len(target["class_labels"])), dtype=torch.long
        )

        if seed is not None:
            with self.lock:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if self.transform is not None:
                    image, target = self.transform(image, target)
        else:

            if self.transform is not None:
                image, target = self.transform(image, target)

        c, h, w = image.shape
        target["orig_size"] = torch.tensor((h, w))
        if h > self.maxsize or w > self.maxsize:
            hq = h if h < self.maxsize else self.maxsize
            hq = hq / h
            wq = w if w < self.maxsize else self.maxsize
            wq = wq / w
            r = min(hq, wq)
            temp = transforms.Compose(
                [
                    transforms.Resize((int(r * h), int(r * w))),
                    transforms.SanitizeBoundingBoxes(),
                ]
            )
            image, target = temp(image, target)
            h = int(r * h)
            w = int(r * w)

        target["size"] = torch.tensor((h, w))

        mask = torch.zeros((self.maxsize, self.maxsize), dtype=torch.long)
        mask[:h, :w] = 1

        padtransform = transforms.Pad(
            (0, 0, self.maxsize - w, self.maxsize - h), fill=0
        )
        image, target = padtransform(image, target)

        target["image_id"] = torch.tensor((idx))

        target["boxes"] = torch.zeros_like(target["bboxes"], dtype=torch.float32)
        target["boxes"][:, 0] = (target["bboxes"][:, 0] + target["bboxes"][:, 2]) / (
            self.maxsize * 2
        )
        target["boxes"][:, 1] = (target["bboxes"][:, 1] + target["bboxes"][:, 3]) / (
            self.maxsize * 2
        )
        target["boxes"][:, 2] = (
            -target["bboxes"][:, 0] + target["bboxes"][:, 2]
        ) / self.maxsize
        target["boxes"][:, 3] = (
            -target["bboxes"][:, 1] + target["bboxes"][:, 3]
        ) / self.maxsize

        if self.require_mask:
            target["area"] = target["masks"].sum([1, 2])
        else:
            target["area"] = (
                target["boxes"][:, 2]
                * target["boxes"][:, 3]
                * self.maxsize
                * self.maxsize
            )

        target["iscrowd"] = torch.zeros(
            (len(target["class_labels"])), dtype=torch.int64
        )

        target["pos"] = zpos
        # print(target["class_labels"])
        for i in ["names", "item_id"]:
            if i in target:
                target[i] = [target[i][j.item()] for j in target["class_labels"]]
        target["class_labels"] = ori_class_labels[target["class_labels"]]

        ret = {
            "pixel_values": image,
            "pixel_mask": mask,
            "labels": target,
        }
        return ret


class CocoDetection2(CocoDetection):
    """
    A subclass of CocoDetection that allows for batch processing of multiple continuous slices
    Using continuous slices enables training stage 1 and stage 2 together.

    Args:
        CocoDetection (_type_): _description_
    """

    def __init__(self, num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("using batch size ", num)
        self.num = num
        self.seed = 42

    def __len__(self):
        return super().__len__() - self.num + 1

    def __getitem__(self, idx):
        # could cause chaos between different workers, but doesn't matter
        with self.lock:
            self.seed = self.seed + 1
            seed = self.seed
        res = []
        num = self.num
        for i in range(num):
            res.append(super().__getitem__(idx + i, seed))
        return stackBatch(res)


class CocoDataModule(L.LightningDataModule):
    def __init__(
        self, trainsets, valsets, train_batch_size, val_batch_size, stack_batch=True
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.trainset = trainsets
        self.valset = valsets
        if stack_batch:
            self.stack_batch = stackBatch
        else:
            self.stack_batch = identicalMapping

    def train_dataloader(self):

        if not isinstance(self.trainset, dict):
            return torch.utils.data.DataLoader(
                dataset=self.trainset,
                collate_fn=self.stack_batch,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=4,
            )

        train_sets = []
        for i in self.trainset:
            train_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.trainset[i],
                    collate_fn=self.stack_batch,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=4,
                )
            )
        return CombinedLoader(
            train_sets,
            mode="max_size_cycle",  # Ensures cycling through both dataloaders
        )

    def val_dataloader(self):

        if not isinstance(self.valset, dict):
            return torch.utils.data.DataLoader(
                dataset=self.valset,
                collate_fn=self.stack_batch,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=4,
            )

        val_sets = []
        for i in self.valset:
            val_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.valset[i],
                    collate_fn=self.stack_batch,
                    batch_size=self.val_batch_size,
                    shuffle=False,
                    num_workers=4,
                )
            )
        return CombinedLoader(
            val_sets, mode="max_size_cycle"  # Ensures cycling through both dataloaders
        )

    def test_dataloader(self):
        return self.testset


class MrcDataset(Dataset):
    def __init__(
        self,
        annotation_path: str,
        norm="hist",
        map_class=None,
        transform=None,
        reshape=800,
        gap=5,
        length_for_average=10,
        require_mask=False,
        mask_length=1,
        mask_input_channels=3,
        add_classname=False,
        filtermin=0,
        zpos_max=None,
        use_ori=True,
    ):
        print("reading dataset:", annotation_path)
        with open(annotation_path, "rb") as f:
            self.annotation = pickle.load(f)

        self.ori_mrc = utils.readTomogram(self.annotation["mrc_path"])
        self.zpos_max = self.ori_mrc.shape[0]
        self.norm = norm
        self.maxsize = reshape

        if map_class is None:
            self.map_class = self.annotation["mapclass"]
        else:
            self.map_class = map_class

        print("dataset map classes: ", self.map_class)

        self.transform = transform
        self.mrc_shape = self.annotation["mrc_shape"]

        self.gap = gap
        self.filtermin = filtermin
        if zpos_max is not None:
            self.zpos_max = zpos_max

        self.needids = None

        self.length_for_average = length_for_average
        self.add_classname = add_classname

        self.mask_length = mask_length
        self.require_mask = require_mask

        self.mask_input_channels = mask_input_channels

        self._filter_ids()

        self.use_ori = use_ori

        # self.lock = threading.Lock()
        self.lock = multiprocessing.Lock()

        print("dataset length:", len(self))

    def _filter_ids(self):
        needids = []
        for i in range(self.mrc_shape[0]):
            cnt = 0
            for j in self.map_class:
                cnt += len(self.annotation["annotations"][j].get(i, []))
            if cnt > self.filtermin:
                needids.append(i)
        # print(len(needids))
        self.needids = needids
        self.start_pos = np.min(self.needids)
        self.end_pos = np.max(self.needids) + 1

    def __len__(self):
        if not hasattr(self, "needids"):
            self._filter_ids()

        if hasattr(self, "idx"):
            if len(self.idx) > 0:
                return len(self.idx)

        t = np.max(self.needids) - np.min(self.needids)

        if self.gap > 0:
            self.idx = []
            for i in range(np.min(self.needids), np.max(self.needids) + 1, self.gap):
                if i in self.needids:
                    self.idx.append(i)
            num = len(self.idx)
            # num = (t + self.gap) // self.gap
        else:
            num = len(self.needids)
        return num

    def _get_slice(self, index, require_mask=False):
        img = np.zeros(
            (3, self.annotation["mrc_shape"][1], self.annotation["mrc_shape"][2]),
            dtype=np.float32,
        )
        img[1] = self.ori_mrc[index]
        if max(0, index - self.length_for_average) < index:
            img[0] = np.mean(
                self.ori_mrc[max(0, index - self.length_for_average) : index], axis=0
            )
        else:
            img[0] = self.ori_mrc[index]
        if index + 1 < min(index + self.length_for_average + 1, len(self.ori_mrc)):

            img[2] = np.mean(
                self.ori_mrc[
                    index
                    + 1 : min(index + self.length_for_average + 1, len(self.ori_mrc))
                ],
                axis=0,
            )
        else:
            img[2] = self.ori_mrc[index]

        if self.norm == "zscore":
            # print("using zscore")
            for i in range(3):
                img[i] = (img[i] - img[i].mean()) / (img[i].std() + 1e-6)
            # print(img[1].mean())
        elif self.norm == "hist":
            # print("before hist")
            for i in range(3):
                img[i] = exposure.equalize_hist(img[i])
            # print("after hist")

        if (
            self.mask_input_channels > 0
            and require_mask
            and (self.mask_input_channels != 3 or not self.use_ori)
        ):
            m = np.zeros(
                (
                    self.mask_input_channels,
                    self.annotation["mrc_shape"][1],
                    self.annotation["mrc_shape"][2],
                )
            )
            for j in range(
                -(self.mask_input_channels // 2), self.mask_input_channels // 2 + 1
            ):
                d = index + j
                if d < 0 or d >= len(self.ori_mrc):
                    continue
                m[j + self.mask_length // 2] = self.ori_mrc[d]
                if self.norm == "zscore":
                    m[j + self.mask_length // 2] = (
                        m[j + self.mask_length // 2]
                        - m[j + self.mask_length // 2].mean()
                    ) / m[j + self.mask_length // 2].std()
                elif self.norm == "hist":
                    m[j + self.mask_length // 2] = exposure.equalize_hist(
                        m[j + self.mask_length // 2]
                    )
            img = np.concatenate((img, m), axis=0)
            # print(img.shape)

        return img

    def _get_annotations(self, index, require_mask=False):
        labels = []
        names = []
        bboxes = []
        masks = []
        item_id = []
        for j in self.map_class:
            t = self.annotation["annotations"][j].get(index, [])
            # print(t)
            for i in t:
                item_id.append(j + "_" + str(i))
                labels.append(self.map_class[j])
                names.append(j)
                bboxes.append(self.annotation["bboxes"][j][index][i])
                if require_mask:
                    m = np.zeros(
                        (
                            self.mask_length,
                            self.annotation["mrc_shape"][1],
                            self.annotation["mrc_shape"][2],
                        )
                    )
                    for k in range(-(self.mask_length // 2), self.mask_length // 2 + 1):
                        mm = self.annotation["masks"][j].get(index + k, None)
                        # print(k, self.mask_length)
                        if mm is not None:
                            m[k + self.mask_length // 2] = (mm == i).toarray()
                    masks.append(m)
            # print(len(names), item_id)

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        if bboxes.dim() < 2:
            bboxes = bboxes.view(-1, 4)
        bt = torchvision.ops.box_convert(bboxes, "xywh", "xyxy")
        boxes = BoundingBoxes(bt, format="xyxy", canvas_size=self.mrc_shape[1:])
        ret = {"bboxes": boxes, "class_labels": torch.tensor(labels, dtype=torch.long)}
        if require_mask:
            if len(masks) == 0:
                masks = torch.zeros(
                    (0, self.mask_length, self.mrc_shape[1], self.mrc_shape[2]),
                    dtype=torch.bool,
                )
            else:
                masks = torch.stack([torch.tensor(m) for m in masks], dim=0)
                masks = Mask(masks)
            ret["masks"] = masks

        if self.add_classname:
            ret["names"] = names

        if len(item_id) > 0:
            ret["item_id"] = item_id
        # print(ret)
        return ret

    def _getitem(self, index, offset=0, require_mask=False, pos=None):
        if pos is not None:
            idx = pos
        else:
            if self.gap > 0:
                idx = self.idx[index] + offset
            else:
                idx = self.needids[index] + offset

            if idx < 0:
                idx = 0
            if idx >= self.mrc_shape[0]:
                idx = self.mrc_shape[0] - 1

            idx = min(self.needids, key=lambda x: abs(x - idx))

        img = self._get_slice(idx, require_mask)
        # img = torch.tensor(img, dtype=torch.float32)
        target = self._get_annotations(idx, require_mask)
        return img, target, idx

    def __getitem__(self, idx=0, seed=None, offset=0, require_mask=True, pos=None):
        # idx1 = idx
        # print("call item")
        # if self.needids is not None:
        #     zpos = self.zpos[idx]
        #     idx = self.needids[idx]

        require_mask = self.require_mask & require_mask

        image, target, zpos = self._getitem(idx, offset, require_mask, pos)

        # print(idx1, idx, len(annotation))
        if len(target) == 0:
            raise ValueError
        image = torch.tensor(image)

        # print(image.mean())

        ori_class_labels = target["class_labels"].clone()
        target["class_labels"] = torch.tensor(
            range(len(target["class_labels"])), dtype=torch.long
        )
        # print("cl", target["class_labels"])
        if seed is not None:
            with self.lock:
                # print(f"getting {zpos} with seed {seed}")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if self.transform is not None:
                    image, target = self.transform(image, target)
                # print("cl", target["class_labels"])
        else:
            if self.transform is not None:
                image, target = self.transform(image, target)
            # print("cl2", target["class_labels"])
        # print(image.mean())
        c, h, w = image.shape
        target["orig_size"] = torch.tensor((h, w))
        if h > self.maxsize or w > self.maxsize:
            hq = h if h < self.maxsize else self.maxsize
            hq = hq / h
            wq = w if w < self.maxsize else self.maxsize
            wq = wq / w
            r = min(hq, wq)
            temp = transforms.Compose(
                [
                    transforms.Resize((int(r * h), int(r * w))),
                    # transforms.SanitizeBoundingBoxes(),
                ]
            )
            image, target = temp(image, target)
            h = int(r * h)
            w = int(r * w)
            # print("cl3", target["class_labels"])
        # print(image.mean())
        target["size"] = torch.tensor((h, w))

        mask = torch.zeros((self.maxsize, self.maxsize), dtype=torch.long)
        mask[:h, :w] = 1

        padtransform = transforms.Pad(
            (0, 0, self.maxsize - w, self.maxsize - h), fill=0
        )
        image, target = padtransform(image, target)

        # print(image.mean())

        target["image_id"] = torch.tensor((idx))

        target["boxes"] = torch.zeros_like(target["bboxes"], dtype=torch.float32)
        target["boxes"][:, 0] = (target["bboxes"][:, 0] + target["bboxes"][:, 2]) / (
            self.maxsize * 2
        )
        target["boxes"][:, 1] = (target["bboxes"][:, 1] + target["bboxes"][:, 3]) / (
            self.maxsize * 2
        )
        target["boxes"][:, 2] = (
            -target["bboxes"][:, 0] + target["bboxes"][:, 2]
        ) / self.maxsize
        target["boxes"][:, 3] = (
            -target["bboxes"][:, 1] + target["bboxes"][:, 3]
        ) / self.maxsize

        # if require_mask:
        #     target["area"] = target["masks"].sum([1, 2])
        # else:
        #     target["area"] = (
        #         target["boxes"][:, 2]
        #         * target["boxes"][:, 3]
        #         * self.maxsize
        #         * self.maxsize
        #     )

        target["iscrowd"] = torch.zeros(
            (len(target["class_labels"])), dtype=torch.int64
        )

        target["pos"] = zpos
        target["zposmax"] = self.zpos_max
        # print(target["class_labels"])
        for i in ["names", "item_id"]:
            if i in target:
                target[i] = [target[i][j.item()] for j in target["class_labels"]]
        target["class_labels"] = ori_class_labels[target["class_labels"]]
        # if self.require_mask:
        #     for i in ["names", "item_id", "class_labels", "bboxes"]:
        #         if i in target:
        #             target[i] = target[i][:: self.mask_length]
        #         target["masks"] = torch.tensor(target["masks"], dtype=torch.bool).view(
        #             -1, self.mask_length, self.maxsize, self.maxsize
        #         )
        # print(image.mean())
        if require_mask:
            # print(image.shape)
            if self.mask_input_channels == 3 and self.use_ori:
                target["mask_input"] = image
            else:
                target["mask_input"] = image[3:]
                image = image[:3]
        # print(image.mean())
        ret = {
            "pixel_values": image,
            "pixel_mask": mask,
            "labels": target,
        }
        return ret


class MrcDataset2(MrcDataset):
    def __init__(self, num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("using %d slices for one graph" % num)
        self.num = num
        self.seed = 42

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        # could cause chaos between different workers, but doesn't matter
        with self.lock:
            self.seed = self.seed + 1
            seed = self.seed
            # print("create seed", seed)
        res = []
        num = self.num
        if self.gap > 0:
            offset = random.randint(0, self.gap - 1)
            offset -= self.gap // 2
        else:
            offset = 0
        # offset = 0
        if self.gap > 0:
            gap = self.gap
        else:
            gap = 1
        for i in range(num):
            if i == num // 2:
                res.append(super().__getitem__(idx, seed, offset, require_mask=True))
            else:
                res.append(
                    super().__getitem__(
                        idx, seed, offset + (i - num // 2) * gap, require_mask=False
                    )
                )
        return stackBatch(res)


class MrcDataModule(L.LightningDataModule):
    def __init__(
        self,
        trainsets,
        valsets,
        train_batch_size,
        val_batch_size,
        stack_batch=True,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.trainset = trainsets
        self.valset = valsets
        if stack_batch:
            self.stack_batch = stackBatch
        else:
            self.stack_batch = identicalMapping

    def train_dataloader(self):

        if not isinstance(self.trainset, dict):
            return torch.utils.data.DataLoader(
                dataset=self.trainset,
                collate_fn=self.stack_batch,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=8,
            )

        train_sets = []
        for i in self.trainset:
            train_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.trainset[i],
                    collate_fn=self.stack_batch,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=8,
                )
            )
        return CombinedLoader(
            train_sets,
            mode="max_size_cycle",  # Ensures cycling through both dataloaders
        )

    def val_dataloader(self):

        if not isinstance(self.valset, dict):
            return torch.utils.data.DataLoader(
                dataset=self.valset,
                collate_fn=self.stack_batch,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=8,
            )

        val_sets = []
        for i in self.valset:
            val_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.valset[i],
                    collate_fn=self.stack_batch,
                    batch_size=self.val_batch_size,
                    shuffle=False,
                    num_workers=8,
                )
            )
        return CombinedLoader(
            val_sets, mode="max_size_cycle"  # Ensures cycling through both dataloaders
        )

    def test_dataloader(self):
        return self.testset


class CocoDataModule(L.LightningDataModule):
    def __init__(
        self, trainsets, valsets, train_batch_size, val_batch_size, stack_batch=True
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.trainset = trainsets
        self.valset = valsets
        if stack_batch:
            self.stack_batch = stackBatch
        else:
            self.stack_batch = identicalMapping

    def train_dataloader(self):

        if not isinstance(self.trainset, dict):
            return torch.utils.data.DataLoader(
                dataset=self.trainset,
                collate_fn=self.stack_batch,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=4,
            )

        train_sets = []
        for i in self.trainset:
            train_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.trainset[i],
                    collate_fn=self.stack_batch,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=4,
                )
            )
        return CombinedLoader(
            train_sets,
            mode="max_size_cycle",  # Ensures cycling through both dataloaders
        )

    def val_dataloader(self):

        if not isinstance(self.valset, dict):
            return torch.utils.data.DataLoader(
                dataset=self.valset,
                collate_fn=self.stack_batch,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=4,
            )

        val_sets = []
        for i in self.valset:
            val_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.valset[i],
                    collate_fn=self.stack_batch,
                    batch_size=self.val_batch_size,
                    shuffle=False,
                    num_workers=4,
                )
            )
        return CombinedLoader(
            val_sets, mode="max_size_cycle"  # Ensures cycling through both dataloaders
        )

    def test_dataloader(self):
        return self.testset


class TestDatasetMrc(Dataset):
    def __init__(
        self,
        mrc_path,
        norm="hist",
        reshape=800,
        gap=5,
        length_for_average=10,
        zpos_max=None,
    ):
        print("loading test dataset")
        self.mrc_path = mrc_path
        self.ori_mrc = utils.readTomogram(mrc_path)

        self.zpos_max = self.ori_mrc.shape[0]

        self.norm = norm
        self.reshape = reshape
        self.gap = gap
        self.length_for_average = length_for_average
        if zpos_max is not None:
            self.zpos_max = zpos_max

        self.start_pos = 0
        self.end_pos = len(self.ori_mrc)

    def __len__(self):
        return (len(self.ori_mrc) + self.gap - 1) // self.gap

    def __getitem__(self, index=None, pos=None, offset=0):
        if index is None and pos is None:
            raise ValueError("Either index or pos must be provided.")
        img = np.zeros(
            (3, self.ori_mrc.shape[1], self.ori_mrc.shape[2]), dtype=np.float32
        )
        if pos is not None:
            idx = pos
        else:
            idx = index * self.gap
        idx += offset
        # if idx is None:
        #     idx = index * self.gap

        img[1] = self.ori_mrc[idx]
        if max(0, idx - self.length_for_average) < idx:
            img[0] = np.mean(
                self.ori_mrc[max(0, idx - self.length_for_average) : idx], axis=0
            )
        else:
            img[0] = self.ori_mrc[idx]
        if idx + 1 < min(idx + self.length_for_average + 1, len(self.ori_mrc)):

            img[2] = np.mean(
                self.ori_mrc[
                    idx + 1 : min(idx + self.length_for_average + 1, len(self.ori_mrc))
                ],
                axis=0,
            )
        else:
            img[2] = self.ori_mrc[idx]
        img[np.isnan(img)] = 0.0
        if self.norm == "zscore":
            for i in range(3):
                img[i] = (img[i] - img[i].mean()) / img[i].std()
        elif self.norm == "hist":
            for i in range(3):
                if np.max(img[i]) - np.min(img[i]) > 0.01:
                    img[i] = exposure.equalize_hist(img[i])

        c, h, w = img.shape

        img = torch.tensor(img, dtype=torch.float32)

        if h > self.reshape or w > self.reshape:
            hq = h if h < self.reshape else self.reshape
            hq = hq / h
            wq = w if w < self.reshape else self.reshape
            wq = wq / w
            r = min(hq, wq)
            new_h = max(1, int(r * h))
            new_w = max(1, int(r * w))
            img = transforms.Resize((new_h, new_w))(img)
            h, w = new_h, new_w
        elif h < self.reshape and w < self.reshape:
            # Upscale when both sides are smaller so the shorter side reaches `reshape`.
            r = self.reshape / min(h, w)
            new_h = max(1, int(r * h))
            new_w = max(1, int(r * w))
            img = transforms.Resize((new_h, new_w))(img)
            h, w = new_h, new_w

            # If upscaling makes one side exceed `reshape`, crop to keep output shape stable.
            if h > self.reshape:
                img = img[:, : self.reshape, :]
                h = self.reshape
            if w > self.reshape:
                img = img[:, :, : self.reshape]
                w = self.reshape

        mask = torch.zeros((self.reshape, self.reshape), dtype=torch.long)
        mask[:h, :w] = 1

        padtransform = transforms.Pad(
            (0, 0, self.reshape - w, self.reshape - h), fill=0
        )
        img = padtransform(img)

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)

        return {
            "pixel_values": img,
            "pixel_mask": mask,
            "labels": {"pos": idx, "zposmax": self.zpos_max, "class_labels": None},
        }


class TestDataset(Dataset):
    """
    a test dataset
    read all images from a directory, and return them as tensors.

    Args:
        image_path (str): Path to the directory containing images.
        norm (str): Normalization method for images, can be 'zscore', 'hist', or None. Default is 'hist'.
        maxsize (int): resizing the image to this size, default is 800.
    """

    def __init__(self, image_path, norm="hist", maxsize=800):
        # self.image_path = image_path
        # self.transform = transform
        self.image_path = image_path
        self.image_list = os.listdir(image_path)
        self.norm = norm
        self.maxsize = maxsize
        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Resize((800, 800)),
        #         transforms.Normalize([0.5], [0.5]),
        #     ]
        # )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = np.load(
            os.path.join(self.image_path, self.image_list[index]), allow_pickle=True
        )
        zpos = self.image_list[index].split(".")[0]
        zpos = int(zpos.split("_")[-1])

        if self.norm == "zscore":
            for i in range(3):
                image[i] = (image[i] - image[i].mean()) / image[i].std()
        elif self.norm == "hist":
            for i in range(3):
                image[i] = exposure.equalize_hist(image[i])

        image = torch.tensor(image)
        # if self.transform is not None:
        #     image = self.transform(image)
        target = {}
        c, h, w = image.shape
        target["orig_size"] = torch.tensor((h, w))

        if h > self.maxsize or w > self.maxsize:
            hq = h if h < self.maxsize else self.maxsize
            hq = hq / h
            wq = w if w < self.maxsize else self.maxsize
            wq = wq / w
            r = min(hq, wq)
            temp = transforms.Compose([transforms.Resize((int(r * h), int(r * w)))])
            image = temp(image)
            h = int(r * h)
            w = int(r * w)

        target["size"] = torch.tensor((h, w))

        mask = torch.zeros((self.maxsize, self.maxsize), dtype=torch.long)
        mask[:h, :w] = 1

        padtransform = transforms.Pad(
            (0, 0, self.maxsize - w, self.maxsize - h), fill=0
        )
        image = padtransform(image)

        return {
            "pixel_values": image,
            "pixel_mask": mask,
            "labels": {"pos": zpos, "zposmax": 500, "class_labels": None},
        }


# helper function to get the collate function for the dataset
def get_collate_fn(image_processor):
    def collate_fn(batch):
        # DETR authors employ various image sizes during training, making it not possible
        # to directly batch together images. Hence they pad the images to the biggest
        # resolution in a given batch, and create a corresponding binary pixel_mask
        # which indicates which pixels are real/which are padding
        pixel_values = [item[0] for item in batch]
        encoding = image_processor.pad(
            pixel_values,
            return_tensors="pt",  # , pad_size={"height": 800, "width": 800}
        )
        labels = [item[1] for item in batch]
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }

    return collate_fn


def stackBatch(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    labels = [item["labels"] for item in batch]
    ret = {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
        # "marks": marks,
    }
    return ret


def identicalMapping(batch):
    return batch


def collect_graph(batch):
    return batch[0]


def getDefaultTransform():

    allt = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResize(600, 1000),
            # transforms.Lambda(lambda x:torch.clamp(x, min=-4.0, max=4.0)),
            transforms.RandomIoUCrop(min_scale=0.8),
            transforms.SanitizeBoundingBoxes(min_size=5),
        ]
    )
    return allt


def getSimpleTransform():

    allt = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True),
            # transforms.RandomResize(600, 1000),
            # transforms.Lambda(lambda x:torch.clamp(x, min=-4.0, max=4.0)),
            # transforms.RandomIoUCrop(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.SanitizeBoundingBoxes(min_size=5),
        ]
    )
    return allt


def getConstantTransform():
    allt = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True),
            # transforms.RandomResize(600, 1000),
            # transforms.Lambda(lambda x:torch.clamp(x, min=-4.0, max=4.0)),
            # transforms.RandomIoUCrop(),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.SanitizeBoundingBoxes(min_size=5),
        ]
    )
    return allt


# helper function to load datasets
def get_stage12_dataset(configs):
    if configs["data"]["transform"] == "default":
        t = getDefaultTransform()
    elif configs["data"]["transform"] == "simple":
        t = getSimpleTransform()
    else:
        t = getConstantTransform()

    if "norm" not in configs["data"]:
        configs["data"]["norm"] = "none"

    if "num" not in configs["data"]:
        configs["data"]["num"] = 5

    if "train_batch_size" not in configs["training"]:
        configs["training"]["train_batch_size"] = 1

    if "val_batch_size" not in configs["training"]:
        configs["training"]["val_batch_size"] = 1

    # dataset = modules.CocoDetection(
    #     configs["image_path"],
    #     configs["annotation_path"],
    #     is_npy=configs["is_npy"],
    #     transform=t,
    #     require_mask=configs["is_segmentation"],
    # )  # , transform=transforms)
    # train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    if isinstance(configs["data"]["annotation_path_train"], list):
        train_sets = []
        for i in configs["data"]["annotation_path_train"]:
            map_class = None
            if "map_class" in configs["data"]:
                map_class = configs["data"]["map_class"]
            train_sets.append(
                CocoDetection2(
                    configs["data"]["num"],
                    configs["data"]["image_path"],
                    i,
                    is_npy=configs["data"]["is_npy"],
                    transform=t,
                    require_mask=configs["data"]["require_mask"],
                    filter_class=configs["data"]["filter_class"],
                    single_class=configs["data"]["single_class"],
                    norm=configs["data"]["norm"],
                    map_class=map_class,
                )
            )
        train_sets = torch.utils.data.ConcatDataset(train_sets)
    else:
        map_class = None
        if "map_class" in configs["data"]:
            map_class = configs["data"]["map_class"]
        train_sets = CocoDetection2(
            configs["data"]["num"],
            configs["data"]["image_path"],
            configs["data"]["annotation_path_train"],
            is_npy=configs["data"]["is_npy"],
            transform=t,
            require_mask=configs["data"]["require_mask"],
            filter_class=configs["data"]["filter_class"],
            single_class=configs["data"]["single_class"],
            norm=configs["data"]["norm"],
            map_class=map_class,
        )

    if isinstance(configs["data"]["annotation_path_val"], list):
        val_sets = []
        for i in configs["data"]["annotation_path_val"]:
            map_class = None
            if "map_class" in configs["data"]:
                map_class = configs["data"]["map_class"]
            val_sets.append(
                CocoDetection2(
                    configs["data"]["num"],
                    configs["data"]["image_path"],
                    i,
                    is_npy=configs["data"]["is_npy"],
                    transform=getConstantTransform(),
                    require_mask=configs["data"]["require_mask"],
                    filter_class=configs["data"]["filter_class"],
                    single_class=configs["data"]["single_class"],
                    norm=configs["data"]["norm"],
                    map_class=map_class,
                )
            )
        val_sets = torch.utils.data.ConcatDataset(val_sets)
    else:
        val_sets = CocoDetection2(
            configs["data"]["num"],
            configs["data"]["image_path"],
            configs["data"]["annotation_path_val"],
            is_npy=configs["data"]["is_npy"],
            transform=getConstantTransform(),
            require_mask=configs["data"]["require_mask"],
            filter_class=configs["data"]["filter_class"],
            single_class=configs["data"]["single_class"],
            norm=configs["data"]["norm"],
            map_class=map_class,
        )

    ds = CocoDataModule(
        train_sets,
        val_sets,
        configs["training"]["train_batch_size"],
        configs["training"]["val_batch_size"],
        False,
    )
    # print("here build dataset stage 1")
    # print(ds)
    return ds


def get_stage12_dataset_mrc(configs):
    if configs["data"]["transform"] == "default":
        t = getDefaultTransform()
    elif configs["data"]["transform"] == "simple":
        t = getSimpleTransform()
    else:
        t = getConstantTransform()

    if "norm" not in configs["data"]:
        configs["data"]["norm"] = "none"

    if "num" not in configs["data"]:
        configs["data"]["num"] = 5

    if "mask_in_channel" in configs["model"]:
        configs["data"]["mask_input_channels"] = configs["model"]["mask_in_channel"]

    if "mask_out_channel" in configs["model"]:
        configs["data"]["mask_length"] = configs["model"]["mask_out_channel"]

    if "train_batch_size" not in configs["training"]:
        configs["training"]["train_batch_size"] = 1

    if "val_batch_size" not in configs["training"]:
        configs["training"]["val_batch_size"] = 1

    if "gap" not in configs["data"]:
        configs["data"]["gap"] = 5

    if "length_for_average" not in configs["data"]:
        configs["data"]["length_for_average"] = 10

    if "filtermin" not in configs["data"]:
        configs["data"]["filtermin"] = 3

    if "mask_input_channels" not in configs["data"]:
        configs["data"]["mask_input_channels"] = 3

    if "mask_length" not in configs["data"]:
        configs["data"]["mask_length"] = 1

    # dataset = modules.CocoDetection(
    #     configs["image_path"],
    #     configs["annotation_path"],
    #     is_npy=configs["is_npy"],
    #     transform=t,
    #     require_mask=configs["is_segmentation"],
    # )  # , transform=transforms)
    # train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    if isinstance(configs["data"]["annotation_path_train"], list):
        train_sets = []
        for i in configs["data"]["annotation_path_train"]:
            map_class = None
            if "map_class" in configs["data"]:
                map_class = configs["data"]["map_class"]
            train_sets.append(
                MrcDataset2(
                    configs["data"]["num"],
                    i,
                    transform=t,
                    require_mask=configs["data"]["require_mask"],
                    gap=configs["data"]["gap"],
                    norm=configs["data"]["norm"],
                    map_class=map_class,
                    mask_input_channels=configs["data"]["mask_input_channels"],
                    mask_length=configs["data"]["mask_length"],
                    length_for_average=configs["data"]["length_for_average"],
                    filtermin=configs["data"]["filtermin"],
                )
            )
        train_sets = torch.utils.data.ConcatDataset(train_sets)
    else:
        map_class = None
        if "map_class" in configs["data"]:
            map_class = configs["data"]["map_class"]
        train_sets = MrcDataset2(
            configs["data"]["num"],
            configs["data"]["annotation_path_train"],
            transform=t,
            gap=configs["data"]["gap"],
            require_mask=configs["data"]["require_mask"],
            norm=configs["data"]["norm"],
            map_class=map_class,
            mask_input_channels=configs["data"]["mask_input_channels"],
            mask_length=configs["data"]["mask_length"],
            length_for_average=configs["data"]["length_for_average"],
            filtermin=configs["data"]["filtermin"],
        )

    if isinstance(configs["data"]["annotation_path_val"], list):
        val_sets = []
        for i in configs["data"]["annotation_path_val"]:
            map_class = None
            if "map_class" in configs["data"]:
                map_class = configs["data"]["map_class"]
            val_sets.append(
                MrcDataset2(
                    configs["data"]["num"],
                    i,
                    transform=t,
                    require_mask=configs["data"]["require_mask"],
                    norm=configs["data"]["norm"],
                    gap=configs["data"]["gap"],
                    map_class=map_class,
                    mask_input_channels=configs["data"]["mask_input_channels"],
                    mask_length=configs["data"]["mask_length"],
                    length_for_average=configs["data"]["length_for_average"],
                    filtermin=configs["data"]["filtermin"],
                )
            )
        val_sets = torch.utils.data.ConcatDataset(val_sets)
    else:
        map_class = None
        if "map_class" in configs["data"]:
            map_class = configs["data"]["map_class"]
        val_sets = MrcDataset2(
            configs["data"]["num"],
            configs["data"]["annotation_path_val"],
            transform=t,
            require_mask=configs["data"]["require_mask"],
            norm=configs["data"]["norm"],
            gap=configs["data"]["gap"],
            map_class=map_class,
            mask_input_channels=configs["data"]["mask_input_channels"],
            mask_length=configs["data"]["mask_length"],
            length_for_average=configs["data"]["length_for_average"],
            filtermin=configs["data"]["filtermin"],
        )

    ds = MrcDataModule(
        train_sets,
        val_sets,
        configs["training"]["train_batch_size"],
        configs["training"]["val_batch_size"],
        False,
    )
    # print("here build dataset stage 1")
    # print(ds)
    return ds


def get_stage1_dataset(configs):
    if configs["data"]["transform"] == "default":
        t = getDefaultTransform()
    elif configs["data"]["transform"] == "simple":
        t = getSimpleTransform()
    else:
        t = getConstantTransform()

    if "norm" not in configs["data"]:
        configs["data"]["norm"] = "none"

    if isinstance(configs["data"]["filter_class"], dict):
        train_sets = {}
        for i in configs["data"]["filter_class"]:
            map_class = None
            if "map_class" in configs["data"]:
                map_class = configs["data"]["map_class"]
            train_sets[i] = CocoDetection(
                configs["data"]["image_path"],
                configs["data"]["annotation_path_train"],
                is_npy=configs["data"]["is_npy"],
                transform=t,
                require_mask=configs["data"]["require_mask"][i],
                filter_class=configs["data"]["filter_class"][i],
                single_class=configs["data"]["single_class"][i],
                norm=configs["data"]["norm"],
                map_class=map_class,
                mark=i,
            )

        val_sets = {}
        for i in configs["data"]["filter_class"]:
            map_class = None
            if "map_class" in configs["data"]:
                map_class = configs["data"]["map_class"]
            val_sets[i] = CocoDetection(
                configs["data"]["image_path"],
                configs["data"]["annotation_path_val"],
                is_npy=configs["data"]["is_npy"],
                transform=getConstantTransform(),
                require_mask=configs["data"]["require_mask"][i],
                filter_class=configs["data"]["filter_class"][i],
                single_class=configs["data"]["single_class"][i],
                norm=configs["data"]["norm"],
                map_class=map_class,
                mark=i,
            )
    else:
        map_class = None
        if "map_class" in configs["data"]:
            map_class = configs["data"]["map_class"]
        train_sets = CocoDetection(
            configs["data"]["image_path"],
            configs["data"]["annotation_path_train"],
            is_npy=configs["data"]["is_npy"],
            transform=t,
            require_mask=configs["data"]["require_mask"],
            filter_class=configs["data"]["filter_class"],
            single_class=configs["data"]["single_class"],
            norm=configs["data"]["norm"],
            map_class=map_class,
        )
        val_sets = CocoDetection(
            configs["data"]["image_path"],
            configs["data"]["annotation_path_val"],
            is_npy=configs["data"]["is_npy"],
            transform=getConstantTransform(),
            require_mask=configs["data"]["require_mask"],
            filter_class=configs["data"]["filter_class"],
            single_class=configs["data"]["single_class"],
            norm=configs["data"]["norm"],
            map_class=map_class,
        )

    ds = CocoDataModule(
        train_sets,
        val_sets,
        configs["training"]["train_batch_size"],
        configs["training"]["val_batch_size"],
    )
    # print("here build dataset stage 1")
    # print(ds)
    return ds


def get_stage2_dataset(configs):
    data_list_train = []
    for i in configs["data"]["train"]:
        data_list_train.append(torch.load(i))

    data_list_val = []
    for i in configs["data"]["val"]:
        data_list_val.append(torch.load(i))

    if "aug" not in configs["data"]:
        configs["data"]["aug"] = True

    ds = stage2DataModule(
        data_list_train,
        data_list_val,
        configs["data"]["dataset_len"],
        ifaug=configs["data"]["aug"],
    )

    return ds


def get_stageMask_dataset(configs):
    pixels = []
    embeds = []
    masks = []
    boxes = []
    sample_mapping = {}
    num1 = 0
    num2 = 0
    for i in configs["data"]["datasets"]:
        data = torch.load(i)
        pixels.extend(data["pixel_values"])
        embeds.append(data["embed"])
        masks.append(data["masks"])
        boxes.append(data["boxes"])
        for j in data["sample_mapping"]:
            sample_mapping[j + num1] = data["sample_mapping"][j] + num2
        num1 += len(data["sample_mapping"])
        num2 += len(data["pixel_values"])
        # sample_mapping.update(data["sample_mapping"])
    pixels = torch.stack(pixels, dim=0)
    embeds = torch.cat(embeds, dim=0)
    masks = torch.cat(masks, dim=0)
    boxes = torch.cat(boxes, dim=0)
    ds = MaskDataModule(
        pixels,
        embeds,
        masks,
        sample_mapping,
        boxes=boxes,
        batch_size=configs["training"]["train_batch_size"],
    )
    return ds


class Particle3DDataset(Dataset):
    """
    Dataset for extracting 3D subvolumes centered at particle coordinates for binary classification.
    Each sample is a (1, 65, 65, 65) volume and a label (0/1).

    Args:
        df (pd.DataFrame): DataFrame with columns ['z', 'y', 'x', 'label', 'tomogram']
            - 'z', 'y', 'x': center coordinates (int)
            - 'label': 0 (neg) or 1 (pos)
            - 'tomogram': path to tomogram file (all rows can share the same path, or be per-row)
        volume_cache (dict, optional): If provided, maps tomogram path to loaded 3D numpy array.
        crop_size (int): Size of the cubic patch to extract (default 65).
        norm (str): Normalization method ('zscore', 'hist', or None).
    """

    def __init__(
        self,
        df,
        tomo_paths,
        crop_size=65,
        norm=None,
        r=0,
        if_augmentation=True,
        output_center=False,
        norm_sample=False,
    ):
        self.df = df.reset_index(drop=True)
        self.crop_size = crop_size
        self.norm = norm
        self.r = r
        self.if_augmentation = if_augmentation
        self.output_center = output_center
        self.norm_sample = norm_sample
        self.loaded_volumes = {}
        for name in tomo_paths:
            path = tomo_paths[name]
            if path.endswith(".mrc"):
                vol = utils.readTomogram(path)
                if self.norm == "zscore":
                    vol = np.stack(
                        [(sl - sl.mean()) / (sl.std() + 1e-6) for sl in vol], axis=0
                    )
                elif self.norm == "hist":
                    from skimage import exposure

                    vol = np.stack([exposure.equalize_hist(sl) for sl in vol], axis=0)
            else:
                vol = np.load(tomo_paths[name])
            self.loaded_volumes[name] = vol

    def __len__(self):
        return len(self.df)

    def _get_volume(self, path):
        return self.loaded_volumes[path]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        z0, y0, x0 = int(row["z"]), int(row["y"]), int(row["x"])
        z, y, x = z0, y0, x0
        label = int(row["label"])
        tomo_path = row["tomogram"]
        vol = self._get_volume(tomo_path)
        D, H, W = vol.shape
        sz = self.crop_size // 2
        # Data augmentation
        if self.if_augmentation:
            # Random center shift
            if self.r > 0:
                dz = np.random.uniform(-self.r, self.r)
                dy = np.random.uniform(-self.r, self.r)
                dx = np.random.uniform(-self.r, self.r)
                z = int(np.clip(z + dz, 0, D - 1))
                y = int(np.clip(y + dy, 0, H - 1))
                x = int(np.clip(x + dx, 0, W - 1))
        else:
            # No augmentation: no random shift
            pass
        # Clamp to valid range
        z1, z2 = max(z - sz, 0), min(z + sz + 1, D)
        y1, y2 = max(y - sz, 0), min(y + sz + 1, H)
        x1, x2 = max(x - sz, 0), min(x + sz + 1, W)
        crop = np.zeros(
            (self.crop_size, self.crop_size, self.crop_size), dtype=vol.dtype
        )
        cz1, cy1, cx1 = sz - (z - z1), sz - (y - y1), sz - (x - x1)
        cz2, cy2, cx2 = cz1 + (z2 - z1), cy1 + (y2 - y1), cx1 + (x2 - x1)
        crop[cz1:cz2, cy1:cy2, cx1:cx2] = vol[z1:z2, y1:y2, x1:x2]

        # Track true particle center location inside the crop volume.
        center_z = int(cz1 + (z0 - z1))
        center_y = int(cy1 + (y0 - y1))
        center_x = int(cx1 + (x0 - x1))
        center_z = int(np.clip(center_z, 0, self.crop_size - 1))
        center_y = int(np.clip(center_y, 0, self.crop_size - 1))
        center_x = int(np.clip(center_x, 0, self.crop_size - 1))

        flip_axes = []
        k = 0
        if self.if_augmentation:
            # Flip along axes
            for axis in range(3):
                if np.random.rand() < 0.5:
                    crop = np.flip(crop, axis=axis)
                    flip_axes.append(axis)
            # Random 90-degree rotation about z axis
            k = np.random.randint(0, 4)
            crop = np.rot90(crop, k=k, axes=(1, 2))

            # Apply same geometric transforms to center coordinates.
            n = self.crop_size
            if 0 in flip_axes:
                center_z = n - 1 - center_z
            if 1 in flip_axes:
                center_y = n - 1 - center_y
            if 2 in flip_axes:
                center_x = n - 1 - center_x

            if k == 1:
                center_y, center_x = n - 1 - center_x, center_y
            elif k == 2:
                center_y, center_x = n - 1 - center_y, n - 1 - center_x
            elif k == 3:
                center_y, center_x = center_x, n - 1 - center_y

        crop = crop.astype(np.float32)
        crop = np.expand_dims(crop, 0)  # (1, D, H, W)
        if self.norm_sample:
            if self.norm == "zscore":
                crop = (crop - crop.mean()) / (crop.std() + 1e-6)
            elif self.norm == "hist":
                from skimage import exposure

                crop = exposure.equalize_hist(crop[0])[None, ...]  # (1, D, H, W)

        if self.output_center:
            center = torch.tensor([center_z, center_y, center_x], dtype=torch.float32)
            return (
                torch.from_numpy(crop),
                torch.tensor(label, dtype=torch.float32),
                center,
            )
        return torch.from_numpy(crop), torch.tensor(label, dtype=torch.float32)


class Particle3DOrientationDataset(Dataset):
    """
    Dataset for extracting 3D subvolumes centered at particle coordinates and outputting orientation angles as sin/cos pairs.
    Each sample is a (1, crop_size, crop_size, crop_size) volume and a vector of 6 values: [sin(rot), cos(rot), sin(tilt), cos(tilt), sin(psi), cos(psi)].

    Args:
        df (pd.DataFrame): DataFrame with columns ['z', 'y', 'x', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 'tomogram']
        tomo_paths (dict): Mapping from tomogram name to file path
        crop_size (int): Size of the cubic patch to extract (default 65)
        norm (str): Normalization method ('zscore', 'hist', or None)
        r (float): Max random center shift for augmentation
    """

    def __init__(
        self, df, tomo_paths, crop_size=65, norm=None, r=0, if_augmentation=True
    ):
        self.df = df.reset_index(drop=True)
        self.crop_size = crop_size
        self.norm = norm
        self.r = r
        self.if_augmentation = if_augmentation
        self.loaded_volumes = {}
        for name in tomo_paths:
            path = tomo_paths[name]
            if path.endswith(".mrc"):
                vol = utils.readTomogram(path)
                if self.norm == "zscore":
                    vol = np.stack(
                        [(sl - sl.mean()) / (sl.std() + 1e-6) for sl in vol], axis=0
                    )
                elif self.norm == "hist":
                    from skimage import exposure

                    vol = np.stack([exposure.equalize_hist(sl) for sl in vol], axis=0)
            else:
                vol = np.load(path)
            self.loaded_volumes[name] = vol

    def __len__(self):
        return len(self.df)

    def _get_volume(self, path):
        return self.loaded_volumes[path]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        z, y, x = int(row["z"]), int(row["y"]), int(row["x"])
        angles = np.array(
            [
                float(row["rlnAngleRot"]),
                float(row["rlnAngleTilt"]),
                float(row["rlnAnglePsi"]),
            ],
            dtype=np.float32,
        )
        tomo_path = row["tomogram"]
        vol = self._get_volume(tomo_path)
        D, H, W = vol.shape
        sz = self.crop_size // 2
        # Data augmentation
        flip_axes = []
        k = 0
        if self.if_augmentation:
            # Random center shift
            if self.r > 0:
                dz = np.random.uniform(-self.r, self.r)
                dy = np.random.uniform(-self.r, self.r)
                dx = np.random.uniform(-self.r, self.r)
                z = int(np.clip(z + dz, 0, D - 1))
                y = int(np.clip(y + dy, 0, H - 1))
                x = int(np.clip(x + dx, 0, W - 1))
            # Flip along axes
            for axis in range(3):
                if np.random.rand() < 0.5:
                    vol = np.flip(vol, axis=axis)
                    flip_axes.append(axis)
            # Random 90-degree rotation about z axis
            k = np.random.randint(0, 4)
            vol = np.rot90(vol, k=k, axes=(1, 2))
        # Clamp to valid range
        z1, z2 = max(z - sz, 0), min(z + sz + 1, D)
        y1, y2 = max(y - sz, 0), min(y + sz + 1, H)
        x1, x2 = max(x - sz, 0), min(x + sz + 1, W)
        crop = np.zeros(
            (self.crop_size, self.crop_size, self.crop_size), dtype=vol.dtype
        )
        cz1, cy1, cx1 = sz - (z - z1), sz - (y - y1), sz - (x - x1)
        cz2, cy2, cx2 = cz1 + (z2 - z1), cy1 + (y2 - y1), cx1 + (x2 - x1)
        crop[cz1:cz2, cy1:cy2, cx1:cx2] = vol[z1:z2, y1:y2, x1:x2]
        # Adjust angles for augmentation
        if self.if_augmentation:
            angles = self._adjust_angles(angles, flip_axes, k)
        # Convert angles to radians for sin/cos
        angles_rad = np.deg2rad(angles)
        sincos = np.concatenate(
            [np.sin(angles_rad), np.cos(angles_rad)], axis=0
        )  # (6,)
        crop = crop.astype(np.float32)
        crop = np.expand_dims(crop, 0)  # (1, D, H, W)
        return torch.from_numpy(crop), torch.from_numpy(sincos)

    def _adjust_angles(self, angles, flip_axes, k):
        """
        Adjust Euler angles (rot, tilt, psi) for flips and 90-degree rotations using rotation matrices.
        Angles are in degrees.
        Returns new angles (rot, tilt, psi) in degrees.
        """
        import scipy.spatial.transform

        # Convert angles to radians
        rot, tilt, psi = angles
        # Relion convention: ZYZ extrinsic (rot, tilt, psi)
        r = scipy.spatial.transform.Rotation.from_euler(
            "ZYZ", [rot, tilt, psi], degrees=True
        )
        # Compose augmentation transforms as rotation matrices
        # 1. Flips (mirror)
        for axis in flip_axes:
            if axis == 0:  # flip z
                # Flipping z: mirror in z, equivalent to 180 deg about x or y, but for ZYZ, can be handled as below
                # For 3D, flipping z is equivalent to a 180 deg rotation about x or y, but for Euler, we can use a matrix
                M = np.diag([1, 1, -1])
                r = scipy.spatial.transform.Rotation.from_matrix(M) * r
            elif axis == 1:  # flip y
                M = np.diag([1, -1, 1])
                r = scipy.spatial.transform.Rotation.from_matrix(M) * r
            elif axis == 2:  # flip x
                M = np.diag([-1, 1, 1])
                r = scipy.spatial.transform.Rotation.from_matrix(M) * r
        # 2. 90-degree rotation about z axis (axes 1,2)
        if k != 0:
            rz = scipy.spatial.transform.Rotation.from_euler("z", 90 * k, degrees=True)
            r = rz * r
        # Convert back to Euler angles (ZYZ, extrinsic)
        new_angles = r.as_euler("ZYZ", degrees=True)
        # Ensure angles are in [0, 360)
        new_angles = np.mod(new_angles, 360)
        return new_angles.astype(np.float32)


def collate_with_aux(batch):
    # 4-tuple to activate: main cls + center localization + center-weighted aux classifier
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0).long()
    center = torch.stack([b[2] for b in batch], dim=0).float()
    aux_label = y.clone()
    return x, y, center, aux_label
