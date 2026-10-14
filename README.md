# Studio 2 — Klasifikasi Sampah · Kelompok <4>

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
|1|Michael Mathew|123140101|Mencoba membangun arsitektur EfficientNet (tidak dipilih)|
|2|Varasina Farmadani|123140107|Membangun arsitektur UniRepLKNet-S (dipilih)|
|3||||

## Ringkasan Pendekatan

- *Arsitektur / backbone*: UniRepLKNet-S pretrained ImageNet-1K dengan classification head yang diinisialisasi ulang untuk klasifikasi 5 kelas: botol_plastik, gelas_plastik, kertas_kardus, saset_kemasan, dan styrofoam.

- *Strategi pelatihan*: Fine-tuning seluruh parameter model tanpa freezing, menggunakan optimizer AdamW dengan learning rate berbeda untuk backbone (1e-5) dan classification head (3e-4), serta weight decay 0.03. Learning rate dijadwalkan menggunakan CosineAnnealingLR dengan T_max=20. Pelatihan maksimal 20 epoch, batch size 4, dan 4 DataLoader workers, dengan early stopping berdasarkan validation Macro-F1 (patience=6). Loss menggunakan weighted CrossEntropyLoss dengan class weighting seimbang dan label smoothing 0.02. Augmentasi meliputi RandomResizedCrop, RandomHorizontalFlip, ColorJitter, dan RandomErasing. Pelatihan menggunakan mixed precision FP16 jika CUDA tersedia dan gradient clipping dengan maximum norm 1.0. Checkpoint terbaik dipilih berdasarkan validation Macro-F1.

- *Data tambahan*: Model menggunakan bobot pretrained ImageNet-1K sebagai inisialisasi. Checkpoint hasil fine-tuning model.pth (lebih dari 100 MB) tersedia di [GitHub Releases — Preview1](https://github.com/glutt28/cnn-klasifikasi-sampah/releases/tag/Preview1).

- *Ukuran input & preprocessing*: Input RGB berukuran 224 × 224 piksel. Pada training, gambar diproses dengan RandomResizedCrop berukuran 224, skala crop 0.80–1.0, rasio aspek 0.85–1.15, dan interpolasi bicubic. Selanjutnya diterapkan augmentasi horizontal flip dan ColorJitter, konversi ke tensor, normalisasi menggunakan mean (0.485, 0.456, 0.406) dan standard deviation (0.229, 0.224, 0.225) dari ImageNet, lalu RandomErasing. Pada validasi dan inferensi, digunakan Resize 256 dengan interpolasi bicubic, CenterCrop 224, konversi ke tensor, dan normalisasi ImageNet tanpa augmentasi acak.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | |0.9566| |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

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
