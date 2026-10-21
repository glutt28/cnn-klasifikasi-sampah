# Studio 2 — Klasifikasi Sampah · Kelompok <4>

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
|1|Michael Mathew|123140101|Augmentasi Dataset|
|2|Varasina Farmadani|123140107|Hyperparameter Tuning|
|3||||

## Ringkasan Pendekatan

**Arsitektur.** UniRepLKNet-S pretrained ImageNet-1K. Classification head diinisialisasi ulang untuk 5 kelas: `botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, `styrofoam`. Seluruh parameter di-fine-tune (tanpa freezing), dengan `drop_path_rate=0.1`.

**Strategi pelatihan.**

| Komponen | Pengaturan |
|---|---|
| Optimizer | AdamW, weight decay 0.03 |
| Learning rate | Backbone 1e-5, head 3e-4 |
| Scheduler | CosineAnnealingLR (`T_max` = jumlah epoch) |
| Loss | CrossEntropyLoss berbobot (class weighting seimbang), label smoothing 0.02 |
| Presisi | Mixed precision FP16 (jika CUDA tersedia) |
| Gradient clipping | Norma maksimum 1.0 |
| Pemilihan checkpoint | Validation Macro-F1 terbaik |
| Seed | 42 |

**Konfigurasi per putaran.**

| | Preview 1 | Preview 2 |
|---|---|---|
| Epoch | 20 | 40 |
| Batch size | 4 | 4 |
| DataLoader workers | 4 | 8 |
| Early stopping | patience 6 | patience 6 |
| Augmentasi | dasar (lihat di bawah) | diperkuat (lihat di bawah) |

**Augmentasi training.**

| Transformasi | Preview 1 | Preview 2 |
|---|---|---|
| RandomResizedCrop (224, bicubic) | scale 0.80–1.0, rasio 0.85–1.15 | scale 0.70–1.0, rasio 0.80–1.20 |
| RandomHorizontalFlip | p=0.5 | p=0.5 |
| RandomVerticalFlip | – | p=0.3 |
| RandomRotation | – | ±15° |
| ColorJitter | ±12% (hue ±2%) | ±25% (hue ±5%) |
| RandomPerspective | – | distortion 0.1, p=0.2 |
| RandomErasing | p=0.08, scale 0.02–0.08 | p=0.15, scale 0.02–0.15 |

**Preprocessing.** Input RGB 224 × 224, dinormalisasi dengan mean `(0.485, 0.456, 0.406)` dan std `(0.229, 0.224, 0.225)` (ImageNet). Pada validasi dan inferensi tidak ada augmentasi acak: Resize 256 (bicubic) → CenterCrop 224 → ToTensor → Normalize.

## Model weights

Karena ukuran model lebih dari 100 MB, file model.pth disimpan di GitHub Release.

Direct download: [model.pth](https://github.com/glutt28/cnn-klasifikasi-sampah/releases/download/Preview2/model.pth)

**Data tambahan.** Hanya bobot pretrained ImageNet-1K sebagai inisialisasi. Checkpoint hasil fine-tuning (`model.pth`, > 100 MB) disimpan di GitHub Releases:

| Putaran | Unduh |
|---|---|
| Preview 1 | [model.pth](https://github.com/glutt28/cnn-klasifikasi-sampah/releases/download/Preview1/model.pth) atau https://github.com/glutt28/cnn-klasifikasi-sampah/releases/download/Preview1/model.pth |
| Preview 2 | [model.pth](https://github.com/glutt28/cnn-klasifikasi-sampah/releases/download/Preview2/model.pth) atau https://github.com/glutt28/cnn-klasifikasi-sampah/releases/download/Preview2/model.pth |

## Hasil

| Putaran | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|----------------------------------|------------------------------|
| Preview 1 (14 Okt)|0.9566| |
| Preview 2 (21 Okt)|0.9581 | |
| Final (25 Okt)| | |

Catatan eksperimen:
Strategi di atas dilakukan dengan konfigurasi learning rate, optimizer, scheduler, loss, augmentasi, dan early stopping berada di train.py, sedangkan ukuran input serta transformasi validasi dan inferensi ditentukan oleh submission/model.py. 

## Cara Reproduksi

```bash
pip install -r requirements.txt
python train.py ...            # perintah melatih model
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik
  requirements.txt  # (opsional) paket tambahan, mis. timm
evaluate.py         # salinan skrip evaluasi resmi
...                 # kode/notebook pelatihan kelompok
```
