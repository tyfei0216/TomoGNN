import json
import os
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

    def __init__(self, dataset_list, dataset_len, train_val=[0.8, 0.2], seed=1013):
        super().__init__()
        self.dataset_list = []
        self.dataset_len = dataset_len
        # split the dataset into train and validation sets
        random.seed(seed)
        for i in dataset_list:
            a = list(range(i.x.shape[0]))
            train = random.sample(a, int(len(a) * train_val[0]))
            train_mask = torch.zeros(len(a), dtype=torch.bool)
            train_mask[train] = True
            val_mask = ~train_mask
            i.train_mask = train_mask
            i.val_mask = val_mask
            if hasattr(i, "edge_label"):
                edge_train_mask = torch.zeros(i.edge_label.shape[0], dtype=torch.bool)
                train = random.sample(
                    list(range(i.edge_label.shape[0])), int(len(a) * train_val[0])
                )
                edge_train_mask[train] = True
                i.edge_train_mask = edge_train_mask
                i.edge_val_mask = ~edge_train_mask
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
        resize=512,
        norm="none",
        filtermin=5,
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
        self.resize = resize
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

    def processAnnotations(self, annotations, image):
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

        if self.require_mask:
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
            image = (image - image.mean()) / image.std()
            # for i in range(3):
            #     image[i] = (image[i] - image[i].mean()) / image[i].std()
        elif self.norm == "hist":
            image = exposure.equalize_hist(image)

            c, w, h = image.shape
            if c == 0:
                print(id)
            # for i in range(3):
            #     image[i] = exposure.equalize_hist(image[i])
        target, _ = self._load_target(id)

        return image, target

    def __getitem__(self, idx, seed=None):
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
        target = self.processAnnotations(annotation, image)

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
        if h > self.resize or w > self.resize:
            hq = h if h < self.resize else self.resize
            hq = hq / h
            wq = w if w < self.resize else self.resize
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

        mask = torch.zeros((self.resize, self.resize), dtype=torch.long)
        mask[:h, :w] = 1

        padtransform = transforms.Pad((0, 0, self.resize - w, self.resize - h), fill=0)
        image, target = padtransform(image, target)

        target["image_id"] = torch.tensor((idx))

        target["boxes"] = torch.zeros_like(target["bboxes"], dtype=torch.float32)
        target["boxes"][:, 0] = (target["bboxes"][:, 0] + target["bboxes"][:, 2]) / (
            self.resize * 2
        )
        target["boxes"][:, 1] = (target["bboxes"][:, 1] + target["bboxes"][:, 3]) / (
            self.resize * 2
        )
        target["boxes"][:, 2] = (
            -target["bboxes"][:, 0] + target["bboxes"][:, 2]
        ) / self.resize
        target["boxes"][:, 3] = (
            -target["bboxes"][:, 1] + target["bboxes"][:, 3]
        ) / self.resize

        if self.require_mask:
            target["area"] = target["masks"].sum([1, 2])
        else:
            target["area"] = (
                target["boxes"][:, 2]
                * target["boxes"][:, 3]
                * self.resize
                * self.resize
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
                num_workers=16,
            )

        train_sets = []
        for i in self.trainset:
            train_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.trainset[i],
                    collate_fn=self.stack_batch,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=16,
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
        self.resize = maxsize
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

        if h > self.resize or w > self.resize:
            hq = h if h < self.resize else self.resize
            hq = hq / h
            wq = w if w < self.resize else self.resize
            wq = wq / w
            r = min(hq, wq)
            temp = transforms.Compose([transforms.Resize((int(r * h), int(r * w)))])
            image = temp(image)
            h = int(r * h)
            w = int(r * w)

        target["size"] = torch.tensor((h, w))

        mask = torch.zeros((self.resize, self.resize), dtype=torch.long)
        mask[:h, :w] = 1

        padtransform = transforms.Pad((0, 0, self.resize - w, self.resize - h), fill=0)
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
            transforms.RandomResize(600, 1000),
            # transforms.Lambda(lambda x:torch.clamp(x, min=-4.0, max=4.0)),
            transforms.RandomIoUCrop(min_scale=0.8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
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
