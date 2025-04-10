import os
import argparse
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from lightning.fabric.loggers import TensorBoardLogger

from configs.config import cfg
from datasets import call_load_dataset
from model import Model
from utils.eval_utils import AverageMeter, get_prompts, validate
from utils.tools import copy_model, create_csv
from model import Model
from sam_lora import LoRA_Sam


def multi_main(cfg):
    prompts = ["box", "point"]
    for prompt in prompts:
        cfg.prompt = prompt
        torch.cuda.empty_cache()
        main(cfg)


def main(cfg: Box, ckpt: str = None) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        model.setup()
        LoRA_Sam(model.model, 4)

    load_datasets = call_load_dataset(cfg)
    _, _, train_each_data = load_datasets(cfg, model.model.image_encoder.img_size)
    fabric.print(f"Train Data: {len(train_each_data) * cfg.batch_size}")
    train_each_data = fabric._setup_dataloader(train_each_data)

    if ckpt is not None:
        full_checkpoint = fabric.load(ckpt)
        model.load_state_dict(full_checkpoint["model"])
    validate(fabric, cfg, model, train_each_data, name=cfg.name, iters=0, save_image_path='./Pred_results/',is_val=False)
    for data_name in ['cod', 'camo', 'cham']:
        _, val_data, _ = load_datasets(cfg, model.model.image_encoder.img_size, eval_dataset=data_name)
        fabric.print(f"Val {data_name} Data: {len(val_data) * cfg.batch_size}")
        val_data = fabric._setup_dataloader(val_data)
        validate(fabric, cfg, model, val_data, name=cfg.name, iters=0, save_image_path='./Pred_results/',eval_dataset=data_name)

    del model, val_data


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    cfg.gpu_ids = '7'
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", 
                        default='/home/panly/CodeField/wesam/output/benchmark/CODAll/save/CODAll-box-last-ckpt.pth', 
                        type=str)
    args = parser.parse_args()

    main(cfg, args.ckpt)
    # multi_main(cfg, args.ckpt)
    torch.cuda.empty_cache()
