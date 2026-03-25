# Import libraries

import os
# import math
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from peft import PeftModel

# Utils
from data_utils.ddp_dataset import PromptSegmentDataset, collate_fn
from segment_model.cgreasoner import build_llm_seg 
from loss import structure_loss, dice_score, BceDiceLoss

IGNORE_INDEX = 0

# --------- Logging ----------
def setup_logging(is_main: bool, log_file: str):
    level = logging.INFO if is_main else logging.ERROR
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file if is_main else None,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def ddp_bar(*args, **kwargs):
    disable = not is_main_process()
    return tqdm(*args, disable=disable, **kwargs)


def all_reduce_mean(t: torch.Tensor):
    if not dist.is_initialized():
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t


def all_reduce_sum(t: torch.Tensor):
    if not dist.is_initialized():
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def count_train_parameters(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        print(f"Number of trainable parameters: {n/1e6:.2f}M")
    return n

def perplexity(logits, labels):
    print(logits.shape, labels.shape)
    labels_len = labels.shape[-1]
    logits_len = logits.shape[-1]
    logits = logits[:, logits_len - labels_len:]
    print("After:", logits.shape, labels.shape)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=IGNORE_INDEX
    )
    return torch.exp(loss)

# ------------- Evaluation -----------------
@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    local_sum = torch.tensor(0.0, device=device)
    local_cnt = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in ddp_bar(val_loader, desc="Evaluating"):
            input_ids        = batch['input_ids'].to(device, non_blocking=True)
            image_tensor     = batch['image_tensor'].to(device, non_blocking=True)
            mask_tensor      = batch['mask_tensor'].to(device, non_blocking=True)
            image_sam_tensor = batch['image_sam'].to(device, non_blocking=True)
            attention_mask   = batch['attention_masks'].to(device, non_blocking=True)
            answers_ids      = batch['answers_ids'].to(device, non_blocking=True)
            labels           = batch['label'].to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs_mask, logits = model(
                    input_ids=input_ids,
                    image_tensor_for_vlm=image_tensor,
                    image_tensor_for_image_enc=image_sam_tensor,
                    attention_mask=attention_mask,
                    answers=answers_ids
                )

            d = dice_score(outputs_mask, mask_tensor).detach()
            local_sum += d
            local_cnt += 1

    global_sum = all_reduce_sum(local_sum.clone())
    global_cnt = all_reduce_sum(local_cnt.clone())
    mean_dice = (global_sum / torch.clamp_min(global_cnt, 1.0)).item()
    return mean_dice


# ------------- Train Loop -----------------------
def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    device,
    save_dir,
    llm_weight = 1.0,
    seg_weight = 2.0,
    cls_weight = 0.5
):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )
    criterion_bce_dice = BceDiceLoss()

    for epoch in range(epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(epoch)

        model.train()
        ep_loss = 0.0
        total_llm_loss = 0.0
        total_segment_loss = 0.0
        total_cls_loss = 0.0

        pbar = ddp_bar(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)

            input_ids        = batch['input_ids'].to(device, non_blocking=True)
            image_tensor     = batch['image_tensor'].to(device, non_blocking=True)
            mask_tensor      = batch['mask_tensor'].to(device, non_blocking=True)
            image_sam_tensor = batch['image_sam'].to(device, non_blocking=True)
            attention_mask   = batch['attention_masks'].to(device, non_blocking=True)
            answers_ids      = batch['answers_ids'].to(device, non_blocking=True)
            labels           = batch['label'].to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs_mask, output_cls, logit_loss = model(
                    input_ids=input_ids,
                    image_tensor_for_vlm=image_tensor,
                    image_tensor_for_image_enc=image_sam_tensor,
                    attention_mask=attention_mask,
                    answers=answers_ids
                )

                outputs_mask = F.interpolate(        
                    outputs_mask, size=(1024, 1024),
                    mode='bilinear', align_corners=False
                )

                # Debug
                if is_main_process() and step < 3:
                    print(f"\n[DEBUG] Epoch {epoch+1} Step {step+1}")
                    print(" mask_tensor unique:", mask_tensor.unique()[:10])
                    print(" mask_tensor dtype/shape:", mask_tensor.dtype, mask_tensor.shape)
                    if torch.isnan(outputs_mask).any():
                        print(" outputs_mask has NaN values!")
                    if torch.isinf(outputs_mask).any():
                        print(" outputs_mask has Inf values!")
                    else:
                        print(" outputs_mask min/max:",
                              outputs_mask.min().item(),
                              outputs_mask.max().item())

                
                outputs_mask = torch.clamp(outputs_mask, min=-10.0, max=10.0)
                outputs_mask = torch.nan_to_num(outputs_mask, nan=0.0, posinf=10.0, neginf=-10.0)
                output_cls   = torch.nan_to_num(output_cls, nan=0.0, posinf=10.0, neginf=-10.0)
                logit_loss   = torch.nan_to_num(logit_loss, nan=0.0, posinf=10.0, neginf=-10.0)

                
                cls_loss = nn.CrossEntropyLoss()(output_cls, labels)
                segment_loss = structure_loss(outputs_mask, mask_tensor)

                if is_main_process() and step < 3:
                    print(f"  logit_loss: {logit_loss.item():.4f}, "
                          f"cls_loss: {cls_loss.item():.4f}, "
                          f"segment_loss: {segment_loss.item():.4f}")

                if epoch < 5:
                    loss = segment_loss + 0.5 * cls_loss + logit_loss
                else:
                    loss = segment_loss + logit_loss

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process():
                    print(f" NaN/Inf detected at step {step+1}, "
                          f"loss={loss.item()}, "
                          f"seg={segment_loss.item()}, "
                          f"cls={cls_loss.item()}, "
                          f"llm={logit_loss.item()}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_loss += loss.item()
            total_llm_loss += float(logit_loss.detach())
            total_segment_loss += float(segment_loss.detach())
            total_cls_loss += float(cls_loss.detach())

            steps_done = step + 1
            avg_loss = ep_loss / steps_done
            avg_llm_loss = total_llm_loss / steps_done
            avg_segment_loss = total_segment_loss / steps_done
            avg_cls_loss = total_cls_loss / steps_done

            if is_main_process():
                print(
                    f"Step {steps_done} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LLM: {avg_llm_loss:.4f} | "
                    f"Seg: {avg_segment_loss:.4f} | "
                    f"Cls: {avg_cls_loss:.4f}"
                )
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    llm=f"{avg_llm_loss:.4f}",
                    seg=f"{avg_segment_loss:.4f}",
                    cls=f"{avg_cls_loss:.4f}"
                )

        scheduler.step()
        ep_loss_t = torch.tensor(ep_loss / max(1, len(train_loader)), device=device)
        ep_loss_mean = all_reduce_mean(ep_loss_t.clone()).item()

        if is_main_process():
            print(f"Epoch [{epoch+2}/{epochs}] train loss: {ep_loss_mean:.4f}")
            logging.info(f"Epoch [{epoch+2}/{epochs}] train loss: {ep_loss_mean:.6f}")

        # mean_dice = evaluate(model, val_loader, device=device)

        if is_main_process():
            # print(f"Epoch [{epoch+2}/{epochs}] val mean Dice: {mean_dice:.4f}")
            # logging.info(f"Epoch [{epoch+2}/{epochs}] val mean Dice: {mean_dice:.6f}")
            os.makedirs(save_dir, exist_ok=True)
            model.module.save_model(os.path.join(save_dir, f"llm_seg_{epoch+1+1}"))


# ---------------- Main ----------------
@dataclass
class Args:
    model_path: str
    data_path: str
    annotation_path: str
    save_dir: str
    batch_size: int = 4
    epochs: int = 20
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-5
    eps: float = 1e-6


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--annotation_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--eps", type=float, default=1e-6)
    return Args(**vars(p.parse_args()))


def main():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    setup_logging(is_main_process(), "logs/training.log")
    if is_main_process():
        print(f"World size: {dist.get_world_size()}, using device: {device}")
        
    
    

    model, tokenizer, image_processor, config = build_llm_seg(
        model_path=args.model_path,
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device=device
    )

# # Updated code to load weights

    ckpt_dir = "/path of the latest checkpoint to re-run "

    # 1. Load classification head
    cls_path = os.path.join(ckpt_dir, "cls.pth")
    if os.path.exists(cls_path):
        model.cls.load_state_dict(torch.load(cls_path, map_location=device))
        print("Loaded classifier head")

    # 2. Load mask decoder
    mask_decoder_path = os.path.join(ckpt_dir, "mask_decoder.pth")
    if os.path.exists(mask_decoder_path):
        model.mask_decoder.load_state_dict(torch.load(mask_decoder_path, map_location=device))
        print("Loaded mask decoder")

    # 3. Load image encoder
    img_enc_path = os.path.join(ckpt_dir, "image_encoder.pth")
    if os.path.exists(img_enc_path):
        model.image_encoder.load_state_dict(torch.load(img_enc_path, map_location=device))
        print(" Loaded image encodcder")

    # 4. Load LoRA adapter
    lora_path = os.path.join(ckpt_dir, "lora_adapter")
    if os.path.exists(lora_path):
        model.base_model = PeftModel.from_pretrained(model.base_model, lora_path)
        print("Loaded LoRA into base_model")


    count_train_parameters(model)
    model = model.to(device)

    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            print(f"WARNING: parameter {name} is still on CPU")

    model = model.to(torch.bfloat16)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    train_dataset = PromptSegmentDataset(
        data_path=args.data_path,
        annotation_path=args.annotation_path,
        image_processor=image_processor,
        tokenizer=tokenizer,
        data_config=config
    )
    val_dataset = PromptSegmentDataset(
        data_path=args.data_path,
        annotation_path=args.annotation_path,
        image_processor=image_processor,
        tokenizer=tokenizer,
        data_config=config
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        # persistent_workers=(args.num_workers > 0)
        persistent_workers=False # Updated code line
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        # persistent_workers=(args.num_workers > 0)
        persistent_workers=False # Updated code line
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.eps
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        save_dir=args.save_dir
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
