# Kontrak Submission dan Milestone

## 1. Struktur Berkas Wajib

```
├── README.md              # Nama kelompok, nama lengkap dan NIM anggota
├── evaluate.py            # Skrip evaluasi resmi (tidak diubah)
└── submission/
    ├── model.py           # Berisi get_model() dan get_transform()
    ├── model.pth          # Bobot model (state_dict)
    └── requirements.txt   # (Opsional) Paket di luar torch, torchvision, timm, pillow, numpy
```

## 2. Kontrak `submission/model.py`

### `get_model()`
- Mengembalikan objek `torch.nn.Module`.
- Hanya membangun arsitektur, tanpa memuat bobot (`torch.load`) dan tanpa unduhan daring.
- Output berupa logits berdimensi `[batch, 5]`.

### `get_transform()`
- Mengembalikan objek *callable*.
- Input: `PIL.Image` (RGB).
- Output: `torch.Tensor` berdimensi `[3, H, W]`.
- Dimensi tensor keluaran harus konsisten (ukuran tetap).

### Indeks Kelas
- 0: `botol_plastik`
- 1: `gelas_plastik`
- 2: `kertas_kardus`
- 3: `saset_kemasan`
- 4: `styrofoam`

## 3. Kontrak `submission/model.pth`

- Disimpan menggunakan `torch.save(model.state_dict(), ...)`.
- Wajib kompatibel dengan pemanggilan `model.load_state_dict(..., strict=True)`.
- Batas ukuran berkas maksimal 500 MB (jika > 100 MB, gunakan Git LFS atau GitHub Release dan cantumkan tautan pada `README.md`).

## 4. Batasan Teknis dan Aturan

- Repositori GitHub disetel publik.
- Anggota kelompok maksimal 3 orang.
- Dilarang mengunggah dataset ke repositori publik.
- Evaluasi dijalankan pada commit terakhir di branch utama sebelum tenggat waktu.
- Waktu inferensi evaluasi maksimal 15 menit.
- Metrik penentu nilai: Macro-F1 pada putaran Final.

## 5. Milestone Tenggat Commit

| Putaran | Tenggat (Branch Utama) | Keterangan |
|---|---|---|
| **Preview 1** | 14 Oktober 2026, 23:59 WIB | Umpan balik (tidak dinilai), repo terdaftar |
| **Preview 2** | 21 Oktober 2026, 23:59 WIB | Umpan balik (tidak dinilai) |
| **Final** | 25 Oktober 2026, 23:59 WIB | Penentu nilai Studio 2 |
