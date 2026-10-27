# Workflow training di Kaggle GPU

Dokumen ini menjelaskan alur untuk memindahkan baseline dari mesin lokal ke Kaggle Notebook/Kernel, menjalankannya dengan GPU jika tersedia, mengambil artefak hasil run, lalu mengevaluasi dan membandingkannya secara lokal. Workflow ini belum mengunggah atau menjalankan kernel.

## Status dan batasan

- Kaggle CLI sudah terpasang dan autentikasi serta akses kompetisi pernah berhasil dikonfirmasi.
- Kompetisi yang dipakai: `document-image-classification-sd234504`.
- Data kompetisi dipasang sebagai input kernel melalui `competition_sources`; jangan menyalin arsip 3,9 GB ke folder upload kernel.
- GPU yang tersedia untuk akun/run belum dijamin. `NvidiaTeslaT4` adalah contoh accelerator ID pada CLI, bukan jaminan slot tersedia.
- Anggarkan kuota GPU mingguan 30 jam. Kaggle menyebut kuota dapat berubah mengikuti demand/resource; angka di halaman akun, Notebook editor, dan session manager adalah acuan aktual.
- Submission ke leaderboard adalah langkah terpisah. Jalankan hanya setelah ada permintaan eksplisit, validasi CSV, dan cek batas harian.

Kaggle menyediakan tampilan pemantauan GPU di Notebook editor, halaman daftar notebook/profil, dan session manager. Cek sisa kuota sebelum meminta accelerator dan setelah selesai. [Panduan penggunaan GPU Kaggle](https://www.kaggle.com/docs/efficient-gpu-usage)

## 1. Periksa akses dan data

Jalankan dari root repository:

```bash
kaggle --version
kaggle competitions files -c document-image-classification-sd234504
```

Di kernel, pertama-tama inspeksi isi `/kaggle/input` dan CSV. Lokasi kompetisi biasanya `/kaggle/input/document-image-classification-sd234504`, tetapi pastikan dengan `Path.iterdir()` sebelum memulai training. Periksa bahwa `train.csv` dan `test.csv` terbaca dan filename relatif cocok dengan gambar yang terpasang.

## 2. Siapkan folder kernel

Buat staging folder khusus. Upload hanya source dan metadata, bukan `data/`, `.venv/`, `checkpoint/` lokal, atau kredensial:

```text
kaggle_kernel/
├── kernel-metadata.json
├── run.py
├── scripts/
└── src/
    └── document_classifier/
```

Inisialisasi metadata dengan CLI, lalu edit file hasilnya:

```bash
mkdir -p kaggle_kernel
kaggle kernels init -p kaggle_kernel
```

Contoh metadata untuk kernel privat. Ganti `<username>` dengan slug akun Kaggle. `code_file` harus menunjuk file runner yang memang ada di folder staging.

```json
{
  "id": "<username>/document-image-classification-sd234504-baseline",
  "title": "Document Image Classification Baseline",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": ["document-image-classification-sd234504"],
  "kernel_sources": [],
  "model_sources": []
}
```

Jika metadata hasil `kaggle kernels init` berbeda, pertahankan struktur yang dibuat CLI dan sesuaikan field yang diperlukan. `--accelerator` pada perintah push dapat digunakan untuk meminta accelerator. Permintaan bisa ditolak, berubah, atau tertunda sesuai akses dan kapasitas Kaggle.

### Runner dan jalur file

Runner perlu membuat source package dapat diimpor, mengarahkan data ke direktori kompetisi di `/kaggle/input/`, dan menyimpan hasil di `/kaggle/working/`. Panggil training dengan konfigurasi run yang tercatat, misalnya:

```text
--data-dir /kaggle/input/document-image-classification-sd234504
--checkpoint-dir /kaggle/working/checkpoint
--report-dir /kaggle/working/reports
```

Setelah training, jalankan prediksi memakai checkpoint terbaik dan tulis CSV ke `/kaggle/working/submission.csv`. Pastikan runner memasukkan direktori `kaggle_kernel/` dan `kaggle_kernel/src/` ke `sys.path` atau menyiapkan package dengan cara lain yang kompatibel dengan kernel script.

Baseline saat ini menggunakan bobot ResNet-18 ImageNet. Mengunduh bobot pretrained memerlukan internet. Jika internet kernel tidak diizinkan, gunakan sumber bobot Kaggle yang dilampirkan lewat `model_sources`, atau dokumentasikan run scratch sebagai eksperimen terpisah. Jangan mengubah pretrained menjadi scratch diam-diam karena hasilnya tidak lagi sebanding dengan baseline lokal.

## 3. Validasi sebelum push

Selesaikan pemeriksaan berikut secara lokal terlebih dahulu agar kuota tidak terpakai untuk memperbaiki kesalahan setup:

1. Pastikan runner dapat menemukan package, metadata, dan competition source.
2. Pastikan nama kolom dan jumlah baris sesuai (`train.csv`, `test.csv`).
3. Tampilkan versi `torch`, `torchvision`, dan status `torch.cuda.is_available()` di log.
4. Gunakan split validation stratified dan seed `42` yang sama seperti baseline lokal agar accuracy dapat dibandingkan.
5. Simpan checkpoint terbaik berdasarkan validation accuracy, history, `validation_result.json`, daftar gambar corrupt, dan `submission.csv` di `/kaggle/working/`.
6. Pastikan inference menghasilkan seluruh 7.994 ID dengan kolom tepat `id,label`, label integer 0–15, tanpa ID kosong atau duplikat.

Rencana kuota untuk run pertama:

- Gunakan satu run terpadu untuk training, validasi, dan inferensi test; jangan buat GPU session khusus EDA atau tes satu batch.
- Coba satu model dan konfigurasi baseline saja. Gunakan hasil CPU lokal dan validasi lokal untuk memperbaiki kode terlebih dahulu.
- Hindari beberapa run bersamaan. Satu run yang selesai memberi baseline yang dapat dibandingkan; iterasi berikutnya dilakukan hanya bila ada perubahan terukur.
- Tentukan batas epoch sebelum push dan simpan checkpoint terbaik tiap epoch. Tambahkan early stopping berbasis validation accuracy untuk run yang dapat berhenti saat metrik tidak membaik.
- Pastikan rute bobot pretrained sudah diputuskan sebelum sesi GPU dimulai (internet yang diizinkan atau bobot dari Kaggle Model source), sehingga waktu GPU tidak habis menunggu unduhan yang gagal.

## 4. Push dan pantau satu run

Perintah berikut mengunggah versi kernel dan langsung meminta run dengan accelerator T4. `--timeout 7200` adalah batas maksimum dua jam untuk run ini; sesuaikan dengan estimasi runtime dan sisa kuota, jangan mengatur batas jauh di atas kebutuhan:

```bash
kaggle kernels push -p kaggle_kernel --accelerator NvidiaTeslaT4 --timeout 7200
```

Timeout membatasi lama run, tetapi tidak menjamin run akan selesai sukses sebelum batas tersebut. Pilih timeout yang memberi ruang cukup untuk training, validasi, dan menyimpan output. `kaggle kernels push` menjalankan versi yang diunggah; pakai `--no-run` bila hanya menyimpan perubahan metadata atau teks.

Simpan kernel ref yang tercantum di output, dengan format `<username>/<kernel-slug>`. Pantau run dan baca log:

```bash
kaggle kernels status <username>/<kernel-slug>
kaggle kernels logs <username>/<kernel-slug>
```

Untuk mengikuti log saat proses berjalan:

```bash
kaggle kernels logs -f <username>/<kernel-slug>
```

Jangan push ulang selama run masih berjalan. Jika gagal, baca log, perbaiki penyebabnya, lalu push versi baru setelah memeriksa kuota. Hindari retry otomatis.

### Memastikan GPU berhenti

- Untuk run `push` non-interaktif, tunggu status menjadi `complete` atau `error`; eksekusi batch berakhir dan tidak meninggalkan sesi GPU interaktif yang perlu dibiarkan terbuka.
- Jangan membuka Notebook editor dengan accelerator hanya untuk membaca output. Ambil output dengan CLI setelah batch selesai.
- Jika pernah membuka sesi Notebook interaktif, hentikan dengan **Stop session** dari Notebook editor/session manager setelah menyimpan perubahan. Mematikan tab browser saja belum cukup untuk menghentikan sesi.
- Versi Kaggle CLI yang tersedia saat panduan ini diperbarui tidak menyediakan subcommand `kernels stop`/`cancel`; `kaggle kernels delete` menghapus kernel dan bukan pengganti penghentian sesi.
- Setelah run, cek lagi indikator GPU di UI Kaggle untuk memastikan kuota tidak terus bertambah.

## 5. Ambil dan periksa hasil

Setelah status menunjukkan run selesai sukses, unduh seluruh output ke folder sementara agar lokasi file hasilnya bisa diperiksa:

```bash
mkdir -p kaggle_kernel_outputs
kaggle kernels output <username>/<kernel-slug> -p kaggle_kernel_outputs
```

Salin artefak yang diperlukan ke lokasi proyek setelah memeriksa strukturnya:

- bobot terbaik → `checkpoint/`
- history dan validation result → `reports/`
- CSV hasil prediksi → `submissions/`

Jangan timpa artefak lokal tanpa mencatat kernel ref, versi, konfigurasi, dan tanggal run. Verifikasi ulang isi checkpoint, metrik validation, jumlah baris/ID/label CSV, dan bahwa file CSV cocok dengan `test.csv`. Test kompetisi tidak memiliki label lokal, jadi jangan menyebut prediksi test sebagai test accuracy atau leaderboard score.

## 6. Catat eksperimen dan iterasi

Untuk setiap run, catat kernel ref/versi, accelerator yang benar-benar diberikan, durasi GPU/kuota sebelum dan sesudah run, seed, split, model/pretrained weights, resolusi, batch size, epoch, learning rate, corrupt sample count, best epoch, validation accuracy/loss, serta status output. Bandingkan perubahan pada split dan metrik yang sama. Update README dengan hasil dan interpretasinya setelah hasil run benar-benar diunduh dan diverifikasi.

## 7. Kirim submission ke kompetisi

Jalankan dari root repository setelah memastikan CSV berasal dari checkpoint/model yang ingin dinilai. Contoh untuk baseline ResNet-18:

```bash
kaggle competitions submit \
  -c document-image-classification-sd234504 \
  -f submissions/submission.csv \
  -m "ResNet18 baseline (ImageNet pretrained)"
```

Sebelum submit, cocokkan `id` dan urutannya dengan `data/test.csv`, pastikan ada 7.994 baris, kolom tepat `id,label`, dan semua label integer 0–15. Cek batas lima submission per hari dan daftar submission yang sudah ada:

```bash
kaggle competitions submissions -c document-image-classification-sd234504
```

Setelah perintah submit selesai, simpan `Submission ref` dan deskripsi/model pada log eksperimen. Periksa lagi status serta public score melalui `kaggle competitions submissions`. Jangan menghitung public score sebagai validation accuracy; keduanya berasal dari split yang berbeda. Simpan CSV yang benar-benar dikirim di `submissions/` agar hasil dapat direproduksi. Jangan commit data atau checkpoint besar.

Baseline ResNet-18 telah dikirim dari `submissions/submission.csv` dengan deskripsi `ResNet18 baseline`. Kaggle mengembalikan submission ref `57695733`, status `COMPLETE`, public score `0.77164`; private score belum tersedia. Perintah aktual:

```bash
kaggle competitions submit -c document-image-classification-sd234504 \
  -f submissions/submission.csv \
  -m "ResNet18 baseline"
```

## Referensi

- [Kaggle CLI: kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [Kaggle CLI: kernel metadata format](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
- [Kaggle: efficient GPU usage and quota monitoring](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks)
