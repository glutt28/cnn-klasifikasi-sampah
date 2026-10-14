"""Fine-tuning EfficientNet untuk Studio 2 - Klasifikasi Sampah.

Pemakaian:
    python train.py --data ./train --val ./my_val
    python evaluate.py --submission ./submission --data ./my_val
    
"""
import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "submission"))
import model as sub  # noqa: E402  (submission/model.py)

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
K = len(sub.CLASSES)


# ----------------------------------------------------------------------------- data
def list_images(data_dir):
    items = []
    for ci, cls in enumerate(sub.CLASSES):
        d = Path(data_dir) / cls
        if not d.is_dir():
            raise SystemExit(f"[GAGAL] folder kelas tidak ditemukan: {d}")
        items += [(p, ci) for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
    return items


def make_split(data_dir, val_dir, val_frac, seed):
    all_items = list_images(data_dir)
    val_dir = Path(val_dir)
    if val_dir.is_dir() and any(val_dir.iterdir()):
        val_keys = {(c, p.name) for p, c in list_images(val_dir)}
        print(f"[split] memakai ulang {val_dir} ({len(val_keys)} gambar)")
    else:
        rng = random.Random(seed)
        val_keys = set()
        for ci, cls in enumerate(sub.CLASSES):
            files = [p for p, c in all_items if c == ci]
            rng.shuffle(files)
            for p in files[: round(len(files) * val_frac)]:
                val_keys.add((ci, p.name))
                dst = val_dir / cls / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
        print(f"[split] membuat {val_dir} ({len(val_keys)} gambar, frac={val_frac})")
    train = [(p, c) for p, c in all_items if (c, p.name) not in val_keys]
    val = [(p, c) for p, c in all_items if (c, p.name) in val_keys]
    return train, val


class ImageDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(sub.IMG_SIZE, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(sub.MEAN, sub.STD),
        T.RandomErasing(p=0.25),
    ])


# ------------------------------------------------------------------------- metrics
def macro_f1(y_true, y_pred):
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    f1s = []
    for c in range(K):
        tp = cm[c, c]
        fp, fn = cm[:, c].sum() - tp, cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)), float(np.trace(cm) / max(1, cm.sum())), f1s


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(device)).argmax(1).cpu().tolist()
        ys += y.tolist()
    return macro_f1(ys, ps)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./train")
    ap.add_argument("--val", default="./my_val")
    ap.add_argument("--out", default=str(ROOT / "submission" / "model.pth"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4, help="LR backbone; head memakai 3x lipat")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="uji alur saja, tanpa unduh bobot")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    amp = device == "cuda"
    print(f"device={device}  backbone={sub.BACKBONE}  img={sub.IMG_SIZE}")

    train_items, val_items = make_split(a.data, a.val, a.val_frac, a.seed)
    print(f"train={len(train_items)}  val={len(val_items)}")
    tl = DataLoader(ImageDS(train_items, train_transform()), a.batch_size, shuffle=True,
                    num_workers=a.workers, drop_last=True, pin_memory=amp)
    vl = DataLoader(ImageDS(val_items, sub.get_transform()), a.batch_size * 2,
                    num_workers=a.workers, pin_memory=amp)

    model = sub.build_model(pretrained=not a.no_pretrained, drop_rate=0.3, drop_path_rate=0.1).to(device)

    # LR berbeda: head baru belajar lebih cepat, backbone cukup disesuaikan pelan-pelan.
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head = [p for p in model.parameters() if id(p) in head_ids]
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.lr * 3}],
                            weight_decay=a.weight_decay)

    steps_per_epoch = len(tl)
    total, warm = a.epochs * steps_per_epoch, steps_per_epoch  # warmup 1 epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    # Bobot kelas ringan (akar invers frekuensi) karena jumlah per kelas tidak seimbang (395..760).
    counts = np.bincount([c for _, c in train_items], minlength=K)
    w = torch.tensor((counts.mean() / counts) ** 0.5, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)  # smoothing meredam efek label noise
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, out = -1.0, Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        f1, acc, per = evaluate(model, vl, device)
        mark = ""
        if f1 > best:
            best, mark = f1, "  <- simpan"
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out)
        print(f"ep {ep:02d}/{a.epochs}  loss={run / steps_per_epoch:.4f}  val_macroF1={f1:.4f}  "
              f"acc={acc:.4f}  ({time.time() - t0:.0f}s){mark}")

    # Verifikasi seperti yang dilakukan evaluate.py: arsitektur dari get_model() + strict=True.
    chk = sub.get_model()
    chk.load_state_dict(torch.load(out, map_location="cpu", weights_only=True), strict=True)
    print(f"\nBest val Macro-F1 = {best:.4f}  ->  {out}  ({out.stat().st_size / 1e6:.1f} MB, strict load OK)")
    print("Cek akhir: python evaluate.py --submission ./submission --data ./my_val")


if __name__ == "__main__":
    main()
