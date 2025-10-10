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
    if not os.path.exists(os.path.join(args.path, "stage1")):
        os.mkdir(os.path.join(args.path, "stage1"))
        os.mkdir(os.path.join(args.path, "stage2"))
        # os.mkdir(os.path.join(args.path, "stage3"))

    L.seed_everything(configs["seed"])

    print("building model")

    configs["model"]["stage"] = "stage 1 mask"
    configs["model"]["mask_in_channels"] = 1
    configs["training"]["scheduler_step"] = configs["training"]["epochs"][0] // 3
    configs["training"]["lr_backbone"] = 4e-5
    configs["training"]["lr_detr"] = 1e-4
    configs["training"]["lr"] = 1e-4
    configs["training"]["weight_decay"] = 0.0

    model = utils.getModel(configs)

    print("finish build model")

    logger_path = (
        configs["training"]["logger_path"]
        if "logger_path" in configs["training"]
        else "tb_logs"
    )
    monitor = "total_validate_loss"

    filename = "{epoch}-{total_validate_loss:.4f}"

    if "gradient_clip_val" not in configs["training"]:
        configs["training"]["gradient_clip_val"] = None

    print("training for stage 1:")

    if configs["training"]["epochs"][0] > 0:
        print("loading dataset")
        # CHECKPOINT = "facebook/detr-resnet-50"
        # image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
        configs["data"]["transform"] = "default"
        configs["data"]["num"] = 1
        configs["data"]["norm"] = "hist"
        configs["data"]["require_mask"] = True

        if "mrc" in configs["data"]:
            ds = data.get_stage12_dataset_mrc(configs)

        print("finish loading data")

        # if configs["model"]["stage"] == "stage mask":
        #     monitor = "total_validate_mask_auroc"

        # mode = "min" if monitor == "total_validate_loss" else "max"

        print(args.path)
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,  # Replace with your validation metric
            mode="min",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
            save_top_k=3,  # Save top k checkpoints based on the monitored metric
            save_last=True,  # Save the last checkpoint at the end of training
            dirpath=os.path.join(
                args.path, "stage1"
            ),  # Directory where the checkpoints will be saved
            filename=filename,  # Checkpoint file naming pattern
        )

        checkpoint_callback2 = ModelCheckpoint(
            every_n_epochs=10,
            save_top_k=-1,
            save_last=False,  # Save the last checkpoint at the end of training
            dirpath=os.path.join(
                args.path, "stage1"
            ),  # Directory where the checkpoints will be saved
            filename=filename,  # Checkpoint file naming pattern
        )

        name = configs["training"]["name"] + "_stage1"

        logger = TensorBoardLogger(logger_path, name=name)
        trainer = Trainer(
            logger=logger,
            devices=args.devices,
            accelerator="gpu",
            max_epochs=configs["training"]["epochs"][0],
            # val_check_interval=1000,
            gradient_clip_val=configs["training"]["gradient_clip_val"],
            accumulate_grad_batches=8,
            log_every_n_steps=5,
            callbacks=[checkpoint_callback, checkpoint_callback2],
            strategy=args.strategy,
        )

        print("start training stage 1")
        # if args.checkpoint is not None:
        #     trainer.fit(model, ds, ckpt_path=args.checkpoint)
        # else:
        trainer.fit(model, ds)
        print("finish training stage 1")
    else:
        print("skip training stage 1")

    model = utils.pickAndLoadBest(model, os.path.join(args.path, "stage1"))

    if configs["training"]["epochs"][1] > 0:

        print("loading dataset")
        # CHECKPOINT = "facebook/detr-resnet-50"
        # image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
        configs["data"]["transform"] = "default"
        configs["data"]["num"] = 15
        configs["data"]["norm"] = "hist"
        configs["data"]["require_mask"] = False

        if "mrc" in configs["data"]:
            ds = data.get_stage12_dataset_mrc(configs)

        print("finish loading data")

        model.stage = "stage 1 + 2"
        model.scheduler_step = configs["training"]["epochs"][1]
        model.lr_backbone = 0.0
        model.lr_detr = 0.0
        name = configs["training"]["name"] + "_stage2"
        logger = TensorBoardLogger(logger_path, name=name)

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,  # Replace with your validation metric
            mode="min",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
            save_top_k=3,  # Save top k checkpoints based on the monitored metric
            save_last=True,  # Save the last checkpoint at the end of training
            dirpath=os.path.join(
                args.path, "stage2"
            ),  # Directory where the checkpoints will be saved
            filename=filename,  # Checkpoint file naming pattern
        )

        checkpoint_callback2 = ModelCheckpoint(
            every_n_epochs=10,
            save_top_k=-1,
            save_last=False,  # Save the last checkpoint at the end of training
            dirpath=os.path.join(
                args.path, "stage2"
            ),  # Directory where the checkpoints will be saved
            filename=filename,  # Checkpoint file naming pattern
        )

        trainer = Trainer(
            logger=logger,
            devices=args.devices,
            accelerator="gpu",
            max_epochs=configs["training"]["epochs"][1],
            # val_check_interval=1000,
            gradient_clip_val=configs["training"]["gradient_clip_val"],
            accumulate_grad_batches=8,
            log_every_n_steps=5,
            callbacks=[checkpoint_callback, checkpoint_callback2],
            strategy=args.strategy,
        )

        print("start training stage 2")
        # if args.checkpoint is not None:
        #     trainer.fit(model, ds, ckpt_path=args.checkpoint)
        # else:
        trainer.fit(model, ds)
        print("finish training stage 2")
    else:
        print("skip training stage 2")

    # model = utils.pickAndLoadBest(model, os.path.join(args.path, "stage2"))

    # print("loading dataset")
    # # CHECKPOINT = "facebook/detr-resnet-50"
    # # image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    # configs["data"]["transform"] = "default"
    # configs["data"]["num"] = 1
    # configs["data"]["norm"] = "hist"
    # configs["data"]["require_mask"] = True

    # if "mrc" in configs["data"]:
    #     ds = data.get_stage12_dataset_mrc(configs)

    # print("finish loading data")

    # model.stage = "stage 1 mask"
    # model.scheduler_step = configs["training"]["epochs"][2]
    # model.lr_backbone = 0.0
    # model.lr_detr = 0.0
    # name = configs["training"]["name"] + "_stage3"
    # logger = TensorBoardLogger(logger_path, name=name)

    # checkpoint_callback = ModelCheckpoint(
    #     monitor=monitor,  # Replace with your validation metric
    #     mode="min",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
    #     save_top_k=3,  # Save top k checkpoints based on the monitored metric
    #     save_last=True,  # Save the last checkpoint at the end of training
    #     dirpath=os.path.join(
    #         args.path, "stage3"
    #     ),  # Directory where the checkpoints will be saved
    #     filename=filename,  # Checkpoint file naming pattern
    # )

    # checkpoint_callback2 = ModelCheckpoint(
    #     every_n_epochs=10,
    #     save_top_k=-1,
    #     save_last=False,  # Save the last checkpoint at the end of training
    #     dirpath=os.path.join(
    #         args.path, "stage3"
    #     ),  # Directory where the checkpoints will be saved
    #     filename=filename,  # Checkpoint file naming pattern
    # )

    # trainer = Trainer(
    #     logger=logger,
    #     devices=args.devices,
    #     accelerator="gpu",
    #     max_epochs=configs["training"]["epochs"][2],
    #     # val_check_interval=1000,
    #     gradient_clip_val=configs["training"]["gradient_clip_val"],
    #     accumulate_grad_batches=8,
    #     log_every_n_steps=5,
    #     callbacks=[checkpoint_callback, checkpoint_callback2],
    #     strategy=args.strategy,
    # )

    # print("start training stage 3")
    # # if args.checkpoint is not None:
    # #     trainer.fit(model, ds, ckpt_path=args.checkpoint)
    # # else:
    # trainer.fit(model, ds)
    # print("finish training stage 3")


if __name__ == "__main__":
    run()
