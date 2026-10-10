**Nilai berbasis performa.** Nilai Studio 2 dihitung dari skor **Macro-F1** model kelompok pada validation set tersembunyi di putaran **Final (25 Oktober 2026)**, dikonversi dengan ambang nilai yang **diumumkan kemudian** (lihat [Penilaian](https://informatika-itera.github.io/deep-learning-IF25-40401/studio-2/#penilaian)). Skor Preview 1 dan Preview 2 hanya umpan balik dan tidak memengaruhi nilai.


## Tugas

Bangun model klasifikasi citra yang menerima satu foto sampah dan memprediksi salah satu dari lima kelas di bawah. Anda bebas memilih pendekatan: CNN _from scratch_ seperti di Pertemuan 4, arsitektur modern dari Pertemuan 5, atau _fine-tuning_ backbone pretrained ImageNet sesuai materi Studio 2. Data tambahan dari luar juga diperbolehkan.

Kelompok tidak pernah melihat validation set. Asisten dosen yang menjalankan model (`model.pth`) kelompok Anda pada validation set menggunakan skrip `evaluate.py` yang sama persis dengan yang ada di starter kit, lalu mencatat skornya di Google Sheets skor kelompok.

## Dataset

Train set berisi **±2.900 gambar** dalam 5 kelas, dikumpulkan dari web lalu dibersihkan (deduplikasi, penyaringan otomatis, dan pemeriksaan manual oleh asisten). Validation set ( **±740 gambar**) berasal dari pengumpulan yang sama, dipisahkan secara acak dan disimpan oleh tim pengajar. Label masih dapat mengandung sedikit _noise_ — menangani hal itu adalah bagian dari tantangan.

Kelas 0 · ±540 gambar

### `botol_plastik`

Botol PET bekas: air mineral, teh kemasan, minuman bersoda — utuh, penyok, dengan atau tanpa tutup. Ciri: memiliki leher botol atau tutup ulir.

Kelas 1 · ±570 gambar

### `gelas_plastik`

Gelas air mineral bertutup segel, gelas plastik es teh / boba, cup minuman plastik. Ciri: tidak memiliki leher botol.

Kelas 2 · ±395 gambar

### `kertas_kardus`

Kertas bekas, kertas diremas, kardus, kertas bungkus nasi, dan gelas kertas.

Kelas 3 · ±760 gambar

### `saset_kemasan`

Saset dan kemasan plastik fleksibel: bungkus mi instan, saset kopi/sampo, bungkus makanan ringan.

Kelas 4 · ±700 gambar

### `styrofoam`

Kotak makanan styrofoam, gelas styrofoam, dan gabus pengemas.

Unduhan

### Train set + starter kit

Folder Google Drive berisi dataset: `train/`, `train_sources.csv`, dan starter kit.

[Unduh Dataset (Google Drive) ↗](https://drive.google.com/drive/folders/1_XcZoDFRznHB7fI603VHzdFxreg3ma60?usp=sharing)

**Lisensi & etika:** gambar dikumpulkan dari internet untuk keperluan pendidikan non-komersial di kelas ini. Sumber setiap gambar tercatat di `train_sources.csv`. **Jangan mengunggah ulang dataset** ke repositori publik, Kaggle, atau Hugging Face — cukup unggah kode dan bobot model.

## Jadwal & Preview Skor

Setiap kelompok mendapat tepat **tiga kali** evaluasi pada validation set. Pada setiap tenggat, asisten dosen mengambil **commit terakhir di branch utama yang dibuat sebelum tenggat**, menjalankan `evaluate.py`, lalu memperbarui Macro-F1 serta akurasi setiap kelompok di **Google Sheets skor kelompok** (tautan dibagikan asisten di grup kelas). Tidak ada evaluasi tambahan di luar tiga putaran ini. Informasi perubahan jadwal disampaikan asisten melalui grup kelas / ketua kelas.

| Putaran | Tenggat commit | Fungsi |
| --- | --- | --- |
| **Preview 1** | Rabu, 14 Oktober 2026 · 23:59 WIB | Umpan balik — tidak dinilai. Kelompok & tautan repo harus sudah terdaftar. |
| **Preview 2** | Rabu, 21 Oktober 2026 · 23:59 WIB | Umpan balik — tidak dinilai. |
| **Final** | Minggu, 25 Oktober 2026 · 23:59 WIB | Skor ini yang menentukan nilai Studio 2. |

## Aturan

1. **Kelompok** beranggotakan maksimal **3 orang**. Daftarkan nama kelompok, anggota (nama + NIM), dan tautan repo GitHub kepada asisten dosen sebelum tenggat Preview 1.
2. **Repo GitHub wajib publik** sejak pendaftaran hingga nilai diumumkan. Seluruh kode pelatihan dan submission berada di repo ini.
3. **Lacak progres lewat commit.** Commit secara berkala selama pengerjaan — riwayat commit dipakai untuk melacak progres kelompok. Jangan mengunggah seluruh pekerjaan dalam satu commit di menit terakhir.
4. **README wajib jelas** dan memuat: nama kelompok, **nama lengkap dan NIM seluruh anggota**, ringkasan pendekatan, sumber data tambahan (jika ada), dan cara reproduksi. Gunakan [`README_TEMPLATE.md`](https://informatika-itera.github.io/deep-learning-IF25-40401/materials/studio-2/starter-kit/README_TEMPLATE.md).
5. **Pendekatan bebas:** CNN from scratch, backbone pretrained (torchvision / timm), maupun data tambahan dari luar diperbolehkan. Cantumkan sumber data tambahan di README.
6. **Format submission wajib** mengikuti kontrak di bagian [Format Submission](https://informatika-itera.github.io/deep-learning-IF25-40401/studio-2/#submission). Submission yang gagal dijalankan oleh `evaluate.py` tidak mendapat skor pada putaran tersebut.
7. **Batas teknis:** berkas `model.pth` maksimal 500 MB dan evaluasi seluruh validation set harus selesai dalam 15 menit. Bila bobot > 100 MB (batas GitHub), lampirkan sebagai aset _GitHub Release_ atau Git LFS dan tulis tautannya di README.
8. **Integritas:** berbagi kode, bobot model, atau prediksi antarkelompok tidak diperbolehkan. Pelanggaran ditindaklanjuti sesuai aturan akademik ITERA.

## Format Submission

Repo kelompok wajib memiliki folder `submission/` dengan struktur berikut:

Struktur repo

```
repo-kelompok/
├── README.md              # nama kelompok, anggota + NIM, pendekatan, hasil
├── submission/
│   ├── model.py           # get_model() dan get_transform()
│   ├── model.pth          # state_dict model terbaik
│   └── requirements.txt   # (opsional) paket tambahan, mis. timm
├── evaluate.py            # salinan skrip evaluasi resmi
└── ...                    # notebook / kode pelatihan
```

submission/model.py — kontrak

```
def get_model():
    """Bangun arsitektur saja — JANGAN mengunduh bobot atau memuat model.pth di sini."""
    ...
    return model            # output: logits [batch, 5]

def get_transform():
    """Transform validasi: PIL.Image (RGB) -> Tensor [3, H, W]."""
    ...

# Urutan kelas (alfabetis, sama dengan torchvision ImageFolder):
# 0 botol_plastik · 1 gelas_plastik · 2 kertas_kardus · 3 saset_kemasan · 4 styrofoam

# Simpan bobot dengan:
torch.save(model.state_dict(), "submission/model.pth")
```

Sebelum tenggat, **uji submission Anda sendiri** pada split validasi yang Anda potong dari train set. Jika perintah ini berhasil, asisten juga dapat menjalankannya:

Terminal

```
python evaluate.py --submission ./submission --data ./my_val
```

## Starter Kit

- [`evaluate.py`](https://informatika-itera.github.io/deep-learning-IF25-40401/materials/studio-2/starter-kit/evaluate.py) — skrip evaluasi resmi (identik dengan milik asisten): Macro-F1, akurasi, F1 per kelas, confusion matrix.
- [`submission/model.py`](https://informatika-itera.github.io/deep-learning-IF25-40401/materials/studio-2/starter-kit/submission/model.py) — contoh submission ResNet-18.
- [`train_baseline.py`](https://informatika-itera.github.io/deep-learning-IF25-40401/materials/studio-2/starter-kit/train_baseline.py) — baseline fine-tuning ResNet-18 yang otomatis membuat split `my_val/` dan menyimpan `submission/model.pth`.
- [`README_TEMPLATE.md`](https://informatika-itera.github.io/deep-learning-IF25-40401/materials/studio-2/starter-kit/README_TEMPLATE.md) — templat README repo kelompok.

Baseline ResNet-18 pretrained (8 epoch) mencapai Macro-F1 sekitar **0,92** pada validation set tersembunyi. Targetkan melampaui angka ini.

## Evaluasi & Penilaian

**Macro-F1** adalah rata-rata F1 dari kelima kelas, sehingga setiap kelas berbobot sama:

```
F1_k     = 2 · Precision_k · Recall_k / (Precision_k + Recall_k)
Macro-F1 = (F1_botol + F1_gelas + F1_kertas + F1_saset + F1_styrofoam) / 5
```

Nilai Studio 2 (skala 0–100) ditentukan oleh Macro-F1 putaran Final dengan **ambang absolut** — bukan peringkat. Semua kelompok bisa mendapat nilai penuh.

**Tabel ambang nilai diumumkan kemudian.**

Submission yang tidak dapat dijalankan pada putaran Final, repo yang tidak publik, atau README tanpa NIM anggota tidak dapat dievaluasi.

## FAQ

- **Boleh pakai Kaggle / Google Colab?** Boleh. Setelah selesai berlatih, unduh `model.pth` dan commit ke repo.
- **Boleh memakai timm atau library lain?** Boleh. Tulis paketnya di `submission/requirements.txt`; asisten memasangnya di environment terpisah. Paket dasar yang selalu tersedia: torch, torchvision, timm, pillow, numpy.
- **Kenapa skor preview berbeda dengan validasi lokal?** Split lokal Anda kecil dan bisa jadi lebih mudah atau lebih sulit. Gunakan validasi lokal untuk memilih model, dan skor preview untuk mengecek generalisasi.
- **Commit kami terlambat beberapa menit dari tenggat.** Asisten akan mengevaluasi commit terakhir _sebelum_ tenggat. Jika tidak ada, kelompok tidak mendapat skor pada putaran tersebut.
- **Bolehkah mengganti anggota kelompok?** Hubungi asisten dosen sebelum Preview 1.