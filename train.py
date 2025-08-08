import argparse
import json
import os

import pytorch_lightning as L
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import data
import utils

torch.set_float32_matmul_precision("high")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-d", "--devices", type=int, nargs="+", default=[0])
    parser.add_argument("-s", "--strategy", type=str, default="auto")
    parser.add_argument("-n", "--name", type=str, default="detr")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args


def run():
    args = parseArgs()
    path = args.path

    with open(os.path.join(path, "config.json"), "r") as f:
        configs = json.load(f)
    json_formatted_str = json.dumps(configs, indent=2)
    print("fetch config from ", os.path.join(path, "config.json"))
    print("--------")
    print("config: ")
    print(json_formatted_str)
    print("--------")
    print("using devices ", args.devices)
    print("using strategy ", args.strategy)
    print("--------")

    L.seed_everything(configs["seed"])

    print("loading dataset")
    # CHECKPOINT = "facebook/detr-resnet-50"
    # image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    if "mrc" in configs["data"]:
        ds = data.get_stage12_dataset_mrc(configs)
    elif configs["model"]["stage"] == "stage 1 + 2" or configs["model"][
        "stage"
    ].startswith("stage 1 + 2 + 3"):
        ds = data.get_stage12_dataset(configs)
    elif configs["model"]["stage"] == "stage 1":
        ds = data.get_stage12_dataset(configs)
    elif configs["model"]["stage"] == "stage 2":
        ds = data.get_stage2_dataset(configs)
    elif configs["model"]["stage"] == "stage mask":
        ds = data.get_stageMask_dataset(configs)

    print("finish loading data")

    print("building model")

    # categories = dataset.coco.cats
    # id2label = {k: v["name"] for k, v in categories.items()}

    # if not configs["use_pretrained"]:
    #     configs["lr_backbone"] = None

    model = utils.getModel(configs)

    # if "checkpoint" in configs["model"] and configs["model"]["checkpoint"] is not None:
    #     t = torch.load(configs["model"]["checkpoint"], map_location="cpu")
    #     model.load_state_dict(t["state_dict"], strict=False)

    print("finish build model")
    monitor = (
        "total_validate_loss"
        if configs["model"]["stage"] != "stage 2"
        else "total_validate_loss"
    )

    if configs["model"]["stage"] == "stage mask":
        monitor = "total_validate_mask_auroc"

    mode = "min" if monitor == "total_validate_loss" else "max"

    filename = (
        "{epoch}-{total_validate_loss:.4f}"
        if configs["model"]["stage"] != "stage 2"
        else "{epoch}-{total_validate_loss:.4f}-{total_validate_auroc:.4f}"
    )
    if configs["model"]["stage"] == "stage mask":
        filename = "{epoch}-{total_validate_mask_auroc:.4f}"

    if configs["model"]["stage"] == "stage 1 + 2":
        filename = "{epoch}-{total_validate_auroc:.4f}-{total_validate_loss:.4f}"

    # p = os.path.join(args.path, "a.txt")
    # f = open(p, "w")
    # a = "abc"
    # for i in range(20):
    #     a = a + a
    # f.write(a)
    # f.close()
    print(args.path)
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,  # Replace with your validation metric
        mode=mode,  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
        save_top_k=3,  # Save top k checkpoints based on the monitored metric
        save_last=True,  # Save the last checkpoint at the end of training
        dirpath=args.path,  # Directory where the checkpoints will be saved
        filename=filename,  # Checkpoint file naming pattern
    )

    checkpoint_callback2 = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=-1,
        save_last=False,  # Save the last checkpoint at the end of training
        dirpath=args.path,  # Directory where the checkpoints will be saved
        filename=filename,  # Checkpoint file naming pattern
    )

    if "gradient_clip_val" not in configs["training"]:
        configs["training"]["gradient_clip_val"] = None

    logger_path = (
        configs["training"]["logger_path"]
        if "logger_path" in configs["training"]
        else "tb_logs"
    )

    name = (
        configs["training"]["name"]
        if "name" in configs["training"]
        else os.path.basename(args.path)
    )

    logger = TensorBoardLogger(logger_path, name=name)
    trainer = Trainer(
        logger=logger,
        devices=args.devices,
        accelerator="gpu",
        max_epochs=configs["training"]["epochs"],
        # val_check_interval=1000,
        gradient_clip_val=configs["training"]["gradient_clip_val"],
        accumulate_grad_batches=configs["training"]["accumulate_grad_batches"],
        log_every_n_steps=configs["training"]["log_every_n_steps"],
        callbacks=[checkpoint_callback, checkpoint_callback2],
        strategy=args.strategy,
    )

    print("start training")
    if args.checkpoint is not None:
        trainer.fit(model, ds, ckpt_path=args.checkpoint)
    else:
        trainer.fit(model, ds)
    print("finish training")


if __name__ == "__main__":
    run()
