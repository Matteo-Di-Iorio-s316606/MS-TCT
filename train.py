import time
import argparse
import csv
from datetime import datetime
import json
from torch.autograd import Variable
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from utils import *
from apmeter import APMeter
import os


# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epoch', type=str, default='50')
parser.add_argument('-model', type=str, default='')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-batch_size', type=str, default='False')
parser.add_argument('-num_clips', type=str, default='False')
parser.add_argument('-skip', type=str, default='False')
parser.add_argument('-num_layer', type=str, default='False')
parser.add_argument('-unisize', type=str, default='False')
parser.add_argument('-alpha_l', type=float, default='1.0')
parser.add_argument('-beta_l', type=float, default='1.0')

# New: experiment + checkpointing/logging
parser.add_argument('--exp_name', type=str, default='mstct_charades', help='experiment name')
parser.add_argument('--out_dir', type=str, default='./runs', help='base output directory')
parser.add_argument('--resume', type=str2bool, default='True', help='resume from last checkpoint if available')
parser.add_argument('--resume_run_dir', type=str, default='', help='resume from a specific run directory (folder containing checkpoints/last.pt)')
parser.add_argument('--ckpt_every', type=int, default=1, help='save numbered checkpoint every N epochs (0 disables)')
parser.add_argument('--save_best_only', type=str2bool, default='False', help='if True, only keep best + last')
parser.add_argument('--print_ap_every', type=int, default=1, help='print full AP vector every N epochs (1 = every epoch)')

args = parser.parse_args()


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED:', SEED)


batch_size = int(args.batch_size)
GPU_ID = int(args.gpu)


# -----------------------------
# Dataset config
# -----------------------------
if args.dataset == 'charades':
    from charades_dataloader import Charades as Dataset

    if str(args.unisize) == "True":
        print("uni-size padd all T to", args.num_clips)
        from charades_dataloader import collate_fn_unisize
        collate_fn_f = collate_fn_unisize(args.num_clips)
        collate_fn = collate_fn_f.charades_collate_fn_unisize
    else:
        from charades_dataloader import mt_collate_fn as collate_fn

    train_split = './data/charades.json'
    test_split = train_split
    rgb_root = args.rgb_root
    flow_root = args.flow_root  # optional
    classes = 157
elif args.dataset == 'cap':
    from cap_dataloader import CAP as Dataset
    from cap_dataloader import mt_collate_fn as collate_fn

    train_split = "./data/cap_val.txt"
    test_split = train_split
    rgb_root = args.rgb_root
    flow_root = args.flow_root
    classes = 202

# -----------------------------
# Run dirs / logging / checkpoints
# -----------------------------
def make_run_dirs(args_):
    """
    Creates (fixed directory, NO timestamp):
      out_dir/exp_name/
        checkpoints/
        logs/
        best/
        save_logit/
        config.json
    """

    # Fixed run directory: NO timestamp
    run_dir = os.path.join(args_.out_dir, args_.exp_name)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    best_dir = os.path.join(run_dir, "best")
    logits_dir = os.path.join(run_dir, "save_logit")

    for d in [run_dir, ckpt_dir, log_dir, best_dir, logits_dir]:
        os.makedirs(d, exist_ok=True)

    # Save / overwrite config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args_), f, indent=2)

    return run_dir, ckpt_dir, log_dir, best_dir, logits_dir

def log_to_csv(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_map, extra=None):
    state = {
        "epoch": int(epoch),
        "best_val_map": float(best_val_map),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "extra": extra or {},
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val_map = float(ckpt.get("best_val_map", 0.0))
    return start_epoch, best_val_map, ckpt


# -----------------------------
# Data loading
# -----------------------------
def load_data(train_split_, val_split_, root):
    print('load data', root)

    if len(train_split_) > 0:
        dataset = Dataset(train_split_, 'training', root, batch_size, classes, int(args.num_clips), int(args.skip))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn
        )
        dataloader.root = root
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split_, 'testing', root, batch_size, classes, int(args.num_clips), int(args.skip))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_dataloader.root = root
    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


# -----------------------------
# Core training / eval
# -----------------------------
def run_network(model, data, gpu, epoch=0, baseline=False):
    inputs, mask, labels, other, hm = data
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))
    hm = Variable(hm.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)

    # CAP/Charades: inputs arriva come (B, T, D). Conv1d vuole (B, D, T).
    if inputs.dim() == 3:
        inputs = inputs.permute(0, 2, 1).contiguous()

    outputs_final, out_hm = model(inputs)

    probs_f = torch.sigmoid(outputs_final) * mask.unsqueeze(2)

    # classification loss
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, reduction="sum")
    loss_f = torch.sum(loss_f) / torch.sum(mask)

    # heatmap loss (optional)
    if float(args.beta_l) > 0:
        loss_h = focal_loss(out_hm, hm)
    else:
        loss_h = torch.tensor(0.0, device=outputs_final.device)

    loss = args.alpha_l * loss_f + args.beta_l * loss_h

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.0
    apm = APMeter()

    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data

        loss.backward()
        optimizer.step()

    train_map = 100 * apm.value().mean()
    print('epoch', epoch, 'train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / max(num_iter, 1.0)
    return train_map, epoch_loss


def val_step(model, gpu, dataloader, epoch, print_ap=False):
    model.train(False)
    apm = APMeter()
    sampled_apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.0
    full_probs = {}

    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)

        # sampled subset condition (note: with num_clips<25 this never triggers)
        if sum(data[1].numpy()[0]) > 25:
            p1, l1 = sampled_25(
                probs.data.cpu().numpy()[0],
                data[2].numpy()[0],
                data[1].numpy()[0]
            )
            sampled_apm.add(p1, l1)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        probs_1 = mask_probs(probs.data.cpu().numpy()[0], data[1].numpy()[0]).squeeze()
        
        # ---- robust key extraction ----
        key = other
        # unwrap liste annidate fino a trovare qualcosa di hashabile
        while isinstance(key, list) and len(key) > 0:
            key = key[0]

        # se è ancora lista o None, fallback
        if isinstance(key, list) or key is None:
            key = f"sample_{int(num_iter)}"

        full_probs[str(key)] = probs_1.T
    epoch_loss = tot_loss / max(num_iter, 1.0)

    # FULL VAL MAP (robust)
    full_ap = apm.value()
    if not torch.is_tensor(full_ap):
        full_ap = torch.tensor(full_ap)
    full_ap = full_ap.float()

    full_nz = torch.count_nonzero(full_ap).item()
    if full_nz == 0:
        val_map = torch.tensor(0.0)
    else:
        val_map = (100.0 * full_ap).sum() / full_nz

    # SAMPLED VAL MAP (robust)
    sampled_ap = sampled_apm.value()
    if not torch.is_tensor(sampled_ap):
        sampled_ap = torch.tensor(sampled_ap)
    sampled_ap = sampled_ap.float()

    sampled_nz = torch.count_nonzero(sampled_ap).item()
    if sampled_nz == 0:
        sample_val_map = torch.tensor(0.0)
    else:
        sample_val_map = (100.0 * sampled_ap).sum() / sampled_nz

    print("\n========== VALIDATION DEBUG ==========")
    print('epoch', epoch, 'Full-val-map:', val_map.item())
    print('epoch', epoch, 'sampled-val-map:', sample_val_map.item())

    if print_ap:
        print("\nFull AP matrix (per class):")
        print(100.0 * full_ap)
        print("\nSampled AP matrix (per class):")
        print(100.0 * sampled_ap)

    print("\nFull AP nonzero classes:", full_nz, "/", full_ap.numel())
    print("Full AP mean:", (100.0 * full_ap).mean().item())
    if full_ap.numel() > 0:
        topv, topi = torch.topk(full_ap, k=min(5, full_ap.numel()))
        print("Top-5 Full AP classes:", [(int(i), float(v * 100)) for i, v in zip(topi, topv)])
    print("=====================================\n")

    apm.reset()
    sampled_apm.reset()

    return full_probs, epoch_loss, val_map


def run(models, criterion, num_epochs=50,
        run_dir=None, ckpt_dir=None, log_dir=None, best_dir=None, logits_dir=None,
        resume=True, ckpt_every=1, save_best_only=False, print_ap_every=1):

    since = time.time()
    Best_val_map = 0.0

    assert run_dir and ckpt_dir and log_dir and best_dir and logits_dir, "run dirs must be provided"

    csv_path = os.path.join(log_dir, "train_log.csv")
    txt_log_path = os.path.join(log_dir, "train_log.txt")

    def tprint(*items):
        msg = " ".join(str(x) for x in items)
        print(msg)
        with open(txt_log_path, "a") as f:
            f.write(msg + "\n")

    last_ckpt_path = os.path.join(ckpt_dir, "last.pt")

    # Resume logic: use the first model entry
    model0, gpu0, dataloaders0, optimizer0, sched0, _ = models[0]
    start_epoch = 0

    print("[DEBUG] looking for last checkpoint at:", last_ckpt_path)
    print("[DEBUG] exists?:", os.path.exists(last_ckpt_path))

    if resume and os.path.exists(last_ckpt_path):
        tprint(f"[RESUME] Found checkpoint: {last_ckpt_path}")
        start_epoch, Best_val_map, _ = load_checkpoint(last_ckpt_path, model0, optimizer0, sched0, map_location="cpu")
        tprint(f"[RESUME] start_epoch={start_epoch}, best_val_map={Best_val_map:.4f}")
    else:
        tprint("[RESUME] No checkpoint found, starting from scratch.")

    for epoch in range(start_epoch, num_epochs):
        since_epoch = time.time()
        tprint(f"\nEpoch {epoch}/{num_epochs - 1}")
        tprint("-" * 10)

        for model, gpu, dataloader, optimizer, sched, model_file in models:
            train_map, train_loss = train_step(model, gpu, optimizer, dataloader['train'], epoch)

            do_print_ap = (print_ap_every > 0 and (epoch % print_ap_every == 0))
            prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], epoch, print_ap=do_print_ap)

            if sched is not None:
                sched.step(val_loss)

            total_time = time.time() - since
            epoch_time = time.time() - since_epoch
            tprint(f"epoch {epoch} Total_Time {total_time:.2f} Epoch_time {epoch_time:.2f}")

            log_to_csv(csv_path, {
                "epoch": epoch,
                "train_map": float(train_map) if torch.is_tensor(train_map) else float(train_map),
                "train_loss": float(train_loss) if torch.is_tensor(train_loss) else float(train_loss),
                "val_loss": float(val_loss) if torch.is_tensor(val_loss) else float(val_loss),
                "val_map": float(val_map) if torch.is_tensor(val_map) else float(val_map),
                "best_val_map": float(Best_val_map),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_sec": float(epoch_time),
            })

            # Save last checkpoint every epoch
            save_checkpoint(last_ckpt_path, model, optimizer, sched, epoch, Best_val_map)

            # Save numbered checkpoint occasionally
            if (ckpt_every > 0) and (epoch % ckpt_every == 0) and (not save_best_only):
                numbered = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
                save_checkpoint(numbered, model, optimizer, sched, epoch, Best_val_map)
                tprint(f"[CKPT] Saved epoch checkpoint -> {numbered}")

            # Save best
            if float(val_map) > float(Best_val_map):
                Best_val_map = float(val_map)
                tprint(f"[BEST] epoch {epoch} Best Val Map Update {Best_val_map:.4f}")

                best_model_path = os.path.join(best_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                tprint(f"[BEST] Saved best model weights -> {best_model_path}")

                best_ckpt_path = os.path.join(best_dir, "best_checkpoint.pt")
                save_checkpoint(best_ckpt_path, model, optimizer, sched, epoch, Best_val_map)
                tprint(f"[BEST] Saved best checkpoint -> {best_ckpt_path}")

                best_logit_path = os.path.join(logits_dir, f"{epoch}.pkl")
                pickle.dump(prob_val, open(best_logit_path, 'wb'), pickle.HIGHEST_PROTOCOL)
                tprint(f"[BEST] logit_saved at: {best_logit_path}")


def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]
        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    if args.mode == 'flow':
        print('flow mode', flow_root)
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'rgb':
        print('RGB mode', rgb_root)
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)
    else:
        raise ValueError("args.mode must be 'rgb' or 'flow'")

    # Create run dirs (or reuse an existing one if --resume_run_dir provided)
    run_dir, ckpt_dir, log_dir, best_dir, logits_dir = make_run_dirs(args)
    print("[RUN] outputs will be saved to:", run_dir)

    if args.train:
        if args.model == "MS_TCT":
            print("MS_TCT")
            from MSTCT.MSTCT_Model import MSTCT

            num_clips = int(args.num_clips)
            num_classes = classes
            inter_channels = [256, 384, 576, 864]
            num_block = 3
            head = 8
            mlp_ratio = 8
            in_feat_dim = 1024
            final_embedding_dim = 512

            rgb_model = MSTCT(inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes)
            print("loaded", args.load_model)
        else:
            raise ValueError("Unknown model: " + str(args.model))

        # Move to GPU
        rgb_model.cuda(GPU_ID)

        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        optimizer = optim.Adam(rgb_model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)

        run(
            [(rgb_model, GPU_ID, dataloaders, optimizer, lr_sched, args.comp_info)],
            criterion,
            num_epochs=int(args.epoch),
            run_dir=run_dir,
            ckpt_dir=ckpt_dir,
            log_dir=log_dir,
            best_dir=best_dir,
            logits_dir=logits_dir,
            resume=bool(args.resume),
            ckpt_every=int(args.ckpt_every),
            save_best_only=bool(args.save_best_only),
            print_ap_every=int(args.print_ap_every),
        )
    else:
        # EVAL-ONLY
        if args.model == "MS_TCT":
            from MSTCT.MSTCT_Model import MSTCT

            num_classes = classes
            inter_channels = [256, 384, 576, 864]
            num_block = 3
            head = 8
            mlp_ratio = 8
            in_feat_dim = 1024
            final_embedding_dim = 512

            rgb_model = MSTCT(inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes)
        else:
            raise ValueError("Unknown model: " + str(args.model))

        rgb_model.cuda(GPU_ID)
        rgb_model.eval()

        # carica checkpoint (OBBLIGATORIO in eval-only)
        # qui decidi tu: best_checkpoint.pt o last.pt ecc.
        ckpt_path = os.path.join(ckpt_dir, "last.pt")
        print("[EVAL] Loading checkpoint:", ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["model_state"]

        # 1) rimuovi i layer che cambiano dimensione (157 -> 202)
        drop_prefixes = [
            "Classfication_Module.linear_pred.",  # classifier
            "Classfication_Module.hm.",           # heatmap head
        ]

        sd_filtered = {}
        dropped = []
        for k, v in sd.items():
            if any(k.startswith(p) for p in drop_prefixes):
                dropped.append(k)
                continue
            sd_filtered[k] = v

        print(f"[EVAL] filtered state_dict: kept={len(sd_filtered)} dropped={len(dropped)}")
        if dropped:
            print("[EVAL] dropped keys sample:", dropped[:10])

        missing, unexpected = rgb_model.load_state_dict(sd_filtered, strict=False)
        print("[EVAL] load_state_dict(strict=False) ok")
        print("[EVAL] missing keys:", len(missing))
        print("[EVAL] unexpected keys:", len(unexpected))
       
        # solo validation
        do_print_ap = True
        prob_val, val_loss, val_map = val_step(rgb_model, GPU_ID, dataloaders['val'], epoch=0, print_ap=do_print_ap)
        print("[EVAL] val_loss:", float(val_loss), "val_map:", float(val_map))