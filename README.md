<img src="assets/icon.svg" width="64" height="64" alt="">

# Supervised Document Image Classification

Proyek klasifikasi gambar dokumen untuk kompetisi `document-image-classification-sd234504`. Tujuannya memprediksi satu dari 16 tipe dokumen berdasarkan tampilan halaman hasil scan. Implementasi model menggunakan PyTorch dan neural network, sesuai ketentuan tugas.

## Dataset

Kompetisi menggunakan subset RVL-CDIP yang terdiri dari 32.003 gambar training dan 7.994 gambar test/evaluasi. File yang diunduh diekstrak ke `data/`:

```text
data/
├── train/train/<class_name>/*.tif
├── test/test/*.tif
├── train.csv
├── test.csv
├── sample_submission.csv
└── document-image-classification-sd234504.zip
```

Dataset dan checkpoint tidak dilacak Git. Folder `data/` dan `checkpoint/` sudah tercantum di `.gitignore`.

### Mengunduh data

Autentikasi Kaggle CLI satu kali pada mesin lokal:

```bash
kaggle auth login
```

Kemudian dari root repository:

```bash
mkdir -p data
kaggle competitions download -c document-image-classification-sd234504 -p data
unzip data/document-image-classification-sd234504.zip -d data
```

Pastikan akun Kaggle memiliki akses ke kompetisi. Jangan menyimpan token atau kredensial di repository.

## Kelas

Label kompetisi adalah integer 0–15 dengan mapping berikut:

| Label | Kategori | Label | Kategori |
| ---: | --- | ---: | --- |
| 0 | advertisement | 8 | memo |
| 1 | budget | 9 | news_article |
| 2 | email | 10 | presentation |
| 3 | file_folder | 11 | questionnaire |
| 4 | form | 12 | resume |
| 5 | handwritten | 13 | scientific_publication |
| 6 | invoice | 14 | scientific_report |
| 7 | letter | 15 | specification |

## Alur eksperimen

1. Inspeksi CSV, struktur folder gambar, distribusi kelas, dan contoh gambar.
2. Bagi data training menjadi train/validation secara stratified dengan seed yang dicatat. Test tidak digunakan untuk tuning.
3. Latih model computer vision berbasis neural network di PyTorch; catat konfigurasi, transformasi, seed, dan checkpoint.
4. Bandingkan eksperimen dengan categorical accuracy pada validation. Accuracy adalah jumlah prediksi yang benar dibagi jumlah sampel.
5. Pilih model berdasarkan hasil validation, lalu prediksi label test.
6. Buat submission dengan kolom persis `id,label`, ID dari `test.csv`, dan label integer 0–15. Validasi jumlah baris 7.994, ID tidak hilang/duplikat, dan label tidak kosong.

Lihat [AGENTS.md](AGENTS.md) untuk alur kerja agent dan [rules.md](rules.md) untuk ketentuan tugas. Update README dengan hasil eksperimen dan interpretasinya setelah training benar-benar dilakukan. Belum ada skor model yang dilaporkan di repository ini.

## Struktur repository

```text
.
├── README.md
├── AGENTS.md
├── rules.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── data/                   # CSV, arsip, dan gambar; diabaikan Git
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_submission.ipynb
├── scripts/
│   ├── train.py
│   └── predict.py
├── src/document_classifier/
│   ├── data.py              # CSV validation and PyTorch Dataset
│   ├── transforms.py        # image resizing and augmentation
│   ├── model.py             # CNN architecture
│   └── submission.py        # submission creation/validation
├── checkpoint/             # bobot hasil training; diabaikan Git
├── submissions/
└── reports/figures/
```

Arsitektur dan komponen data ditempatkan dalam package `src/document_classifier/`, sementara `scripts/` berisi entry point. Artefak bobot PyTorch disimpan di `checkpoint/`. Folder `submissions/` untuk CSV hasil inferensi dan `reports/figures/` untuk visualisasi analisis.

## Menjalankan kode

Siapkan environment dan dependensi:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Notebook berisi langkah awal EDA, validasi split stratified, dan pemeriksaan submission. Training dan prediksi dijalankan dari root repository:

```bash
python scripts/train.py --epochs 3 --batch-size 32 --image-size 224 --workers 2
python scripts/predict.py --batch-size 32 --workers 2
```

Training memilih checkpoint berdasarkan validation accuracy tertinggi dan menyimpan metrik ke `reports/`. Prediksi memakai checkpoint `checkpoint/best.pt` dan menulis `submissions/submission.csv`.

## Hasil baseline

Satu run baseline dilakukan dengan split stratified 80/20 dan seed 42. Model memakai ResNet-18 dengan bobot ImageNet; layer4 dan classifier di-fine-tune selama tiga epoch. Resolusi input 224 × 224, batch size 32, optimizer AdamW dengan learning rate `1e-4`. Run berjalan di CPU.

| Epoch | Train accuracy | Validation accuracy | Validation loss |
| ---: | ---: | ---: | ---: |
| 1 | 65,62% | 74,10% | 0,8611 |
| 2 | 78,16% | 77,74% | 0,7542 |
| 3 | 83,65% | **78,30%** | **0,7308** |

Checkpoint terbaik adalah epoch 3. Validation accuracy naik 4,20 poin persentase dari epoch 1. Selisih train dan validation accuracy pada epoch 3 sekitar 5,35 poin, jadi masih ada ruang untuk memperbaiki generalisasi lewat tuning augmentasi, resolusi, fine-tuning, dan analisis error.

Satu gambar training (`train_08795`) tidak bisa didekode dan dilewati; accuracy train dihitung dari 25.601 gambar yang berhasil dibaca, validation memakai seluruh 6.401 sampel. Detail run ada di `reports/training_history.csv` dan `reports/validation_result.json`, sedangkan ID corrupt dicatat di `reports/corrupt_image_ids.txt`.

Model memprediksi seluruh 7.994 baris test dan menghasilkan `submissions/submission.csv`. File tersebut dikirim ke Kaggle sebagai submission ref `57695733` dengan deskripsi `ResNet18 baseline`. Statusnya `COMPLETE` dan public score yang dilaporkan Kaggle adalah **0,77164**; private score belum tersedia. Skor Kaggle ini berbeda dari validation accuracy 78,30% karena dihitung pada data leaderboard yang terpisah.

## Submission

Contoh format CSV:

```csv
id,label
test_00001,0
test_00002,12
```

Perintah submit yang digunakan dari root repository:

```bash
kaggle competitions submit -c document-image-classification-sd234504 \
  -f submissions/submission.csv \
  -m "ResNet18 baseline"
```

Batas kompetisi adalah lima submission per hari. Lihat [workflow.md](workflow.md) untuk langkah validasi, pengecekan status, dan pencatatan submission berikutnya.
