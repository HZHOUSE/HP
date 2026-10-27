#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Seven from-scratch PyTorch baselines; access train/ and val/ only.

python src/train_compare.py --data data/rightpad_v2_split \
  --out runs/compare7_lr1e2 --epochs 50 --batch-size 32 \
  --lr 0.01 --optimizer sgd --sched step --lr-decay 2 --lr-step 10 \
  --img-size 160 --workers 8 --seed 42 --save-pred

No splitting, pretrained downloads, test loading or early stopping.
Dependencies: torch, torchvision, Pillow (via torchvision), standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as image_functional

MODELS = ("vgg16", "googlenet", "resnet18", "resnet34", "resnet50", "shufflenet", "densenet")
ALIASES = {"shufflenet_v2_x1_0": "shufflenet", "shufflenet_v2": "shufflenet",
           "densenet121": "densenet", "desentnet": "densenet"}
ROOT = Path(__file__).resolve().parents[1]
METRICS = ("acc", "macro_p", "macro_r", "macro_f1", "weighted_p", "weighted_r", "weighted_f1", "loss")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def seed_worker(_worker_id):
    random.seed(torch.initial_seed() % (2 ** 32))


class EpochImageFolder(datasets.ImageFolder):
    """Same sample/epoch flip for every model, independent of dropout RNG."""

    def __init__(self, root, transform, seed, training=False):
        super().__init__(str(root), transform=None)
        self.final_transform, self.seed, self.training = transform, seed, training
        self.epoch = 0

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.training:
            generator = torch.Generator().manual_seed(
                (self.seed + self.epoch * len(self) + index) % (2 ** 63 - 1))
            if torch.rand((), generator=generator).item() < 0.5:
                image = image_functional.hflip(image)
        return self.final_transform(image), target


def build_datasets(data, img_size, seed):
    # Whitelist two subdirectories. Never scan the supplied root recursively.
    for split in ("train", "val"):
        if not (data / split).is_dir():
            raise ValueError(f"Required directory missing: {data / split}")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = {split: EpochImageFolder(data / split, transform, seed, training=split == "train")
          for split in ("train", "val")}
    if ds["train"].class_to_idx != ds["val"].class_to_idx:
        raise ValueError("train/ and val/ must have exactly the same class folders.")
    if len(ds["train"].classes) < 2:
        raise ValueError("At least two non-empty classes are required.")
    return ds


def audit_train_val(ds, data):
    """Exact-content overlap audit on the authorized splits; does not alter data."""
    hashes, fingerprints, manifest = {}, {}, []
    for split in ("train", "val"):
        split_hashes, digest = set(), hashlib.sha256()
        split_root = (data / split).resolve()
        for filename, target in ds[split].samples:
            path = Path(filename)
            try:
                relative = path.resolve().relative_to(split_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"Image points outside its split: {path}") from exc
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            split_hashes.add(file_hash)
            record = dict(split=split, file=relative, target=target,
                          class_name=ds[split].classes[target], sha256=file_hash)
            manifest.append(record)
            digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            digest.update(b"\n")
        hashes[split], fingerprints[split] = split_hashes, digest.hexdigest()
    overlap = hashes["train"] & hashes["val"]
    if overlap:
        raise ValueError(f"{len(overlap)} identical contents shared by train and val. "
                         "Training aborted; supplied split remains unchanged.")
    return manifest, fingerprints


def build_loaders(ds, batch_size, workers, seed, device):
    # Recreated per model: same shuffling/worker seeds. Restart workers per epoch
    # so the epoch value used for augmentation reaches their dataset copies.
    kwargs = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=device.type == "cuda", persistent_workers=False)
    return {split: DataLoader(ds[split], shuffle=split == "train", drop_last=False,
                             generator=torch.Generator().manual_seed(seed + offset),
                             worker_init_fn=seed_worker, **kwargs)
            for offset, split in enumerate(("train", "val"))}


def build_model(name, n_classes):
    constructors = dict(vgg16=models.vgg16, resnet18=models.resnet18,
                        resnet34=models.resnet34, resnet50=models.resnet50,
                        shufflenet=models.shufflenet_v2_x1_0, densenet=models.densenet121)
    if name == "googlenet":
        return models.googlenet(weights=None, num_classes=n_classes, aux_logits=False,
                                transform_input=False, init_weights=True)
    return constructors[name](weights=None, num_classes=n_classes)


def metrics_from_cm(cm):
    """Rows=true, cols=predicted. Undefined class metrics are zero.

    TP=diagonal, FP=column_sum-TP, FN=row_sum-TP.
    P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN).
    Macro: equal class weights. Weighted: weights=true class support / total.
    """
    counts = cm.detach().cpu().to(torch.float64)
    tp, support, predicted = counts.diag(), counts.sum(1), counts.sum(0)
    fp, fn = predicted - tp, support - tp
    precision, recall = tp / (tp + fp).clamp_min(1), tp / (tp + fn).clamp_min(1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    weights = support / support.sum().clamp_min(1)
    return dict(acc=float(tp.sum() / support.sum().clamp_min(1)),
                macro_p=float(precision.mean()), macro_r=float(recall.mean()), macro_f1=float(f1.mean()),
                weighted_p=float((precision * weights).sum()),
                weighted_r=float((recall * weights).sum()), weighted_f1=float((f1 * weights).sum()),
                per_class=dict(precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(),
                               support=support.to(torch.int64).tolist(), correct=tp.to(torch.int64).tolist()))


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes, collect=False, limit_batches=0):
    model.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64, device=device)
    loss_sum, count, predictions = 0.0, 0, []
    for batch_index, (images, targets) in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss; inspect training log and data.")
        predicted = logits.argmax(1)
        cm += torch.bincount(targets * n_classes + predicted,
                             minlength=n_classes ** 2).reshape(n_classes, n_classes)
        loss_sum += loss.item() * len(targets)
        count += len(targets)
        if collect:
            confidence = logits.softmax(1).max(1).values
            predictions.extend(zip(targets.cpu().tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()))
    if count == 0:
        raise ValueError("Validation loader yielded no samples.")
    result = metrics_from_cm(cm)
    result.update(loss=loss_sum / count, cm=cm.cpu(), predictions=predictions)
    return result


def train_one_epoch(model, loader, criterion, optimizer, device, limit_batches=0, clip=0):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for batch_index, (images, targets) in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; no per-model parameter adjustment is applied.")
        loss.backward()
        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        loss_sum += loss.item() * len(targets)
        correct += int((logits.argmax(1) == targets).sum().item())
        count += len(targets)
    if count == 0:
        raise ValueError("Training loader yielded no samples.")
    return loss_sum / count, correct / count


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty CSV output: {path}")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar_metrics(result):
    return {key: result[key] for key in METRICS}


def save_validation_outputs(out, name, selection, result, dataset, save_pred):
    rows = []
    for index, class_name in enumerate(dataset.classes):
        row = dict(class_index=index, class_name=class_name)
        row.update({key: values[index] for key, values in result["per_class"].items()})
        rows.append(row)
    write_csv(out / f"per_class_{name}_{selection}_val.csv", rows)
    with (out / f"confusion_{name}_{selection}_val.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + list(dataset.classes))
        for class_name, values in zip(dataset.classes, result["cm"].tolist()):
            writer.writerow([class_name] + values)
    if save_pred:
        rows = []
        for (filename, target), (true, predicted, confidence) in zip(dataset.samples, result["predictions"]):
            if target != true:
                raise RuntimeError("Validation prediction order mismatch.")
            rows.append(dict(file=Path(filename).relative_to(dataset.root).as_posix(),
                             true=dataset.classes[true], pred=dataset.classes[predicted],
                             confidence=confidence, correct=int(true == predicted)))
        if len(rows) != len(result["predictions"]):
            raise RuntimeError("Validation prediction count mismatch.")
        write_csv(out / f"pred_{name}_{selection}_val.csv", rows)


def run_one(name, args, ds, device):
    set_seed(args.seed)
    loaders = build_loaders(ds, args.batch_size, args.workers, args.seed, device)
    model = build_model(name, len(ds["train"].classes)).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.sched == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=1 / args.lr_decay)
    elif args.sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    start = time.perf_counter()
    best_score, best_epoch, best_state, logs = -1.0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        ds["train"].epoch = epoch
        used_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer,
                                               device, args.limit_batches, args.clip)
        val = evaluate(model, loaders["val"], criterion, device, len(ds["train"].classes),
                       limit_batches=args.limit_batches)
        improved = val["macro_f1"] > best_score
        if improved:
            best_score, best_epoch = val["macro_f1"], epoch
            # CPU checkpoint avoids storing a second VGG model on GPU.
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if scheduler is not None:
            scheduler.step()
        log = dict(epoch=epoch, lr=used_lr, next_lr=optimizer.param_groups[0]["lr"],
                   train_loss=train_loss, train_acc=train_acc)
        log.update({f"val_{key}": value for key, value in scalar_metrics(val).items()})
        log["seconds"] = time.perf_counter() - epoch_start
        logs.append(log)
        write_csv(args.out / f"log_{name}.csv", logs)
        print(f"[{name}] {epoch:02d}/{args.epochs} lr={used_lr:.8g} train_loss={train_loss:.4f} "
              f"val_ACC={val['acc']:.4f} P={val['macro_p']:.4f} R={val['macro_r']:.4f} "
              f"F1={val['macro_f1']:.4f} weighted_F1={val['weighted_f1']:.4f} "
              f"{log['seconds']:.1f}s{' *best' if improved else ''}", flush=True)
    train_seconds = time.perf_counter() - start
    last_val = evaluate(model, loaders["val"], criterion, device, len(ds["train"].classes),
                        collect=args.save_pred, limit_batches=args.limit_batches)
    save_validation_outputs(args.out, name, "last", last_val, ds["val"], args.save_pred)
    if args.save_best:
        (args.out / "ckpt").mkdir(exist_ok=True)
        torch.save(dict(model=name, epoch=args.epochs, classes=ds["train"].classes,
                        pretrained=False, state_dict=model.state_dict()), args.out / "ckpt" / f"{name}_last.pt")
    model.load_state_dict(best_state)
    best_val = evaluate(model, loaders["val"], criterion, device, len(ds["train"].classes),
                        collect=args.save_pred, limit_batches=args.limit_batches)
    save_validation_outputs(args.out, name, "best", best_val, ds["val"], args.save_pred)
    if args.save_best:
        torch.save(dict(model=name, epoch=best_epoch, classes=ds["train"].classes,
                        pretrained=False, state_dict=best_state), args.out / "ckpt" / f"{name}_best.pt")
    row = dict(model=name, params_M=params_m, epochs=args.epochs, ran_epochs=len(logs),
               best_epoch=best_epoch, selection="highest_val_macro_f1_first_tie", pretrained=False,
               val_acc=best_val["acc"], val_P=best_val["macro_p"], val_R=best_val["macro_r"],
               val_F1=best_val["macro_f1"], val_P_w=best_val["weighted_p"],
               val_R_w=best_val["weighted_r"], val_F1_w=best_val["weighted_f1"], val_loss=best_val["loss"])
    row.update({f"last_val_{key}": value for key, value in scalar_metrics(last_val).items()})
    row["train_min"] = train_seconds / 60
    print(f"[{name}] done: best_epoch={best_epoch} val_ACC={row['val_acc']:.4f} "
          f"P={row['val_P']:.4f} R={row['val_R']:.4f} F1={row['val_F1']:.4f}; "
          f"last_ACC={last_val['acc']:.4f} last_F1={last_val['macro_f1']:.4f}; "
          f"{row['train_min']:.1f} min", flush=True)
    del model, optimizer, best_state, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Seven from-scratch baselines, train/val only.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/rightpad_v2_split")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/compare7_lr1e2")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--sched", choices=("step", "cosine", "none"), default="step")
    parser.add_argument("--lr-decay", type=float, default=2.0)
    parser.add_argument("--lr-step", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip", type=float, default=0, help="Uniform gradient clipping; default disabled.")
    parser.add_argument("--save-pred", action="store_true", help="Save validation predictions only.")
    parser.add_argument("--save-best", action="store_true", help="Save best and last weights.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--check-only", action="store_true", help="Audit train/val, no training.")
    parser.add_argument("--smoke", action="store_true", help="1 epoch/1 batch per split; diagnostic only.")
    parser.add_argument("--limit-batches", type=int, default=0, help="Diagnostic batch limit; 0=full split.")
    args = parser.parse_args(argv)
    args.names = [ALIASES.get(name.strip().lower(), name.strip().lower())
                  for name in args.models.split(",") if name.strip()]
    if not args.names or len(set(args.names)) != len(args.names):
        parser.error("Provide a non-empty list of distinct models.")
    if set(args.names) - set(MODELS):
        parser.error(f"Supported models: {', '.join(MODELS)}")
    if min(args.epochs, args.batch_size, args.img_size, args.lr_step) <= 0:
        parser.error("epochs, batch-size, img-size and lr-step must be positive.")
    if args.lr <= 0 or args.lr_decay <= 1 or args.wd < 0 or args.clip < 0:
        parser.error("Require lr>0, lr-decay>1, wd>=0 and clip>=0.")
    if args.workers < 0 or args.limit_batches < 0 or args.seed < 0:
        parser.error("workers, limit-batches and seed must be non-negative.")
    if args.smoke:
        args.epochs, args.limit_batches = 1, 1
    args.data, args.out = args.data.resolve(), args.out.resolve()
    if args.out == args.data or args.data in args.out.parents:
        parser.error("Output must be outside the supplied data directory.")
    if (args.out / "config.json").exists() or (args.out / "results.csv").exists():
        parser.error("Output already contains a run; choose a new --out.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable.")
    return args


def main(argv=None):
    args = parse_args(argv)
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    ds = build_datasets(args.data, args.img_size, args.seed)
    manifest, fingerprints = audit_train_val(ds, args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                  device=str(device), pretrained=False, evaluated_splits=["val"],
                  accessed_splits=["train", "val"], test_accessed=False, precision="float32",
                  loss="cross_entropy", momentum=0.9 if args.optimizer == "sgd" else None,
                  augmentation="resize + seeded_train_hflip(p=0.5) + fixed_normalization",
                  class_to_idx=ds["train"].class_to_idx,
                  counts={split: len(dataset) for split, dataset in ds.items()},
                  fingerprints=fingerprints, exact_train_val_overlap=0,
                  selection="highest_val_macro_f1_first_tie", diagnostic=bool(args.limit_batches))
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out / "train_val_manifest.csv", manifest)
    print(f"device={device}; train={len(ds['train'])}; val={len(ds['val'])}; "
          f"classes={len(ds['train'].classes)}; weights=None; test_accessed=False", flush=True)
    print(f"models={','.join(args.names)}; epochs={args.epochs}; batch={args.batch_size}; "
          f"lr={args.lr}; optimizer={args.optimizer}; sched={args.sched}; "
          f"wd={args.wd}; clip={args.clip}", flush=True)
    if args.limit_batches:
        print("DIAGNOSTIC RUN: metrics use limited batches, not the full validation set.", flush=True)
    if args.check_only:
        print(f"Train/val audit passed. No training. Outputs: {args.out}", flush=True)
        return
    results = []
    for name in args.names:
        results.append(run_one(name, args, ds, device))
        write_csv(args.out / "results.csv", results)
    print(f"Completed {len(results)} models. Validation results: {args.out / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
