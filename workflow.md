# Workflow training di Kaggle GPU

Dokumen ini menjelaskan alur untuk memindahkan baseline dari mesin lokal ke Kaggle Notebook/Kernel, menjalankannya dengan GPU jika tersedia, mengambil artefak hasil run, lalu mengevaluasi dan membandingkannya secara lokal. Workflow ini belum mengunggah atau menjalankan kernel.

## Status dan batasan

- Kaggle CLI sudah terpasang dan autentikasi serta akses kompetisi pernah berhasil dikonfirmasi.
- Akun perlu memverifikasi nomor telepon untuk mengakses GPU. Periksa **Phone Verification** di pengaturan akun Kaggle sebelum menjalankan eksperimen GPU. Jika verifikasi bermasalah, gunakan [bantuan verifikasi telepon Kaggle](https://www.kaggle.com/contact#/account/activate/phone). [Penjelasan staf Kaggle](https://www.kaggle.com/discussions/product-feedback/202584)
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

Status `RUNNING` menunjukkan status run yang dilaporkan Kaggle, tetapi belum membuktikan training sudah dimulai atau GPU sudah dialokasikan. Runner akan menampilkan `CUDA available: True` dan nama `GPU` saat kode mulai berjalan; jika CUDA tidak tersedia, runner berhenti. Jika log kosong, tunggu atau periksa kembali status dan jangan menyatakan GPU sudah terpakai sebelum ada bukti di log. Log dapat dibaca melalui CLI tanpa membuat sesi Notebook interaktif tambahan.

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

Setelah perintah submit selesai, simpan `Submission ref` dan deskripsi/model pada `reports/submission_history.csv`. Catat waktu UTC, nama dan konfigurasi model, bobot pretrained, validation accuracy, public score, private score (kosong bila belum tersedia), status, message, dan path CSV. Periksa status serta skor melalui `kaggle competitions submissions`. Public leaderboard score biasanya hanya mencakup bagian test yang dipakai leaderboard publik; jangan menyebutnya skor seluruh test. Private score hanya tersedia jika Kaggle mengungkapkannya. Jangan bandingkan public score dengan validation accuracy seolah-olah berasal dari split yang sama. Simpan CSV yang benar-benar dikirim di `submissions/` agar hasil dapat direproduksi. Jangan commit data atau checkpoint besar.

Untuk memilih perubahan model berikutnya, gunakan validation accuracy sebagai metrik rutin antar eksperimen dengan split/seed yang sama. Gunakan public score sebagai pemeriksaan tambahan pada submission terpilih, bukan sebagai pengganti validation. Catat setiap submission sebagai baris baru, jangan menimpa hasil sebelumnya; gunakan ref dan konfigurasi untuk menghubungkan skor dengan model/CSV yang tepat.

### Loop eksperimen saat ini

Target pra-submission adalah validation accuracy **di atas `0.80` dan mengalahkan model terbaik sebelumnya** pada split stratified 80/20 dengan seed 42. Target evaluasi public score setelah submission adalah `0.79164` (naik dua poin persentase dari `0.77164`); public score kandidat diketahui setelah dikirim. Setiap run harus menggunakan folder output unik dan mencatat konfigurasi serta hasil, termasuk run yang belum mencapai target. Hentikan loop saat sudah ada lima submission dalam batas harian, termasuk submission yang sudah ada sebelumnya.

Bangun dan jalankan kernel pertama dengan kode proyek yang dikemas dalam satu script; periksa kapasitas GPU terlebih dahulu:

```bash
python scripts/build_kaggle_kernel.py --username <username>
kaggle kernels push -p kaggle_kernel --accelerator NvidiaTeslaT4 --timeout 7200
```

Untuk iterasi selanjutnya, siapkan ResNet-50 dengan padding agar rasio halaman tetap terjaga. Konfigurasi diringkas di `reports/experiments/resnet50_aspect_v1.json`:

```bash
python scripts/build_kaggle_kernel.py \
  --username muhammadkafanalkafi \
  --slug document-classification-resnet50-v1 \
  --architecture resnet50 \
  --unfreeze-from layer3 \
  --preserve-aspect-ratio \
  --epochs 4 \
  --patience 2 \
  --learning-rate 0.00005
kaggle kernels push -p kaggle_kernel --accelerator NvidiaTeslaT4 --timeout 7200
```

Eksperimen pertama `resnet18_layer3_v1` pada versi kernel 1 berakhir dengan `CANCEL_ACKNOWLEDGED` dan log kosong. Versi kernel 2 mengonfirmasi Tesla T4 tetapi berhenti sebelum training karena CSV tidak ditemukan. Versi kernel 3 selesai dengan validation accuracy `0.82862` dan submission ref `57698008`; public score `0.82057`. Setiap percobaan disimpan terpisah dalam `reports/experiments/`. Setelah kernel baru selesai, unduh output ke direktori khusus dan periksa laporan serta history. Contoh langkah untuk run v3 yang sudah selesai:

```bash
mkdir -p kaggle_kernel_outputs/resnet18_layer3_v3
kaggle kernels output muhammadkafanalkafi/document-classification-resnet18-improved/3 \
  -p kaggle_kernel_outputs/resnet18_layer3_v3
python scripts/submit_candidate.py --experiment-dir kaggle_kernel_outputs/resnet18_layer3_v3
python scripts/submit_candidate.py --sync-only
```

Skrip mengecek ulang riwayat submission di Kaggle, memeriksa hitungan hari UTC dan allowance resmi dari `kaggle competitions submission-limits`, memastikan validation accuracy checkpoint sesuai laporan, di atas 80%, dan mengalahkan skor model terdahulu, serta memvalidasi seluruh 7.994 ID dan label. CSV submission disalin ke nama unik berdasarkan hash; manifest mencatat hash checkpoint, laporan, dan CSV; ref serta score yang tersedia dicatat di `reports/submission_history.csv`. Jika riwayat Kaggle tidak dapat diperiksa atau API mengembalikan hasil yang ambigu, hentikan dan selidiki sebelum mencoba lagi. Pastikan CSV dan checkpoint memang berasal dari run yang sama dengan meninjau output kernel. Gunakan `--sync-only` kembali untuk memperbarui score setelah Kaggle selesai menilai.

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

### Kandidat ensemble probabilitas

Gunakan checkpoint yang sudah dievaluasi pada split seed 42 yang sama. Contoh evaluasi tetap 30% model 224px dan 70% model 256px:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_ensemble.py \
  --checkpoint kaggle_kernel_outputs/resnet18_layer2_v1/checkpoint/best.pt --weight 0.3 \
  --checkpoint kaggle_kernel_outputs/resnet18_layer2_256_v1/checkpoint/best.pt --weight 0.7 \
  --minimum-validation 0.8454928917356663 \
  --output-dir kaggle_kernel_outputs/ensemble_224_256_v1
```

Skrip membandingkan validation dan membuat prediksi test hanya jika ensemble melewati ambang yang ditentukan. Sebelum submit, pemeriksaan berikut memvalidasi hash kedua checkpoint, hash CSV, urutan ID, skor, dan kuota. `--dry-run` tidak mengirim submission; hapus opsi itu hanya jika kandidat layak dikirim dan kuota tersedia.

```bash
.venv/bin/python scripts/submit_candidate.py --ensemble --dry-run \
  --experiment-dir kaggle_kernel_outputs/ensemble_224_256_v1
```

Pelindung menghitung submission pada hari UTC dan memeriksa `numAllowedNow` dari `kaggle competitions submission-limits --json`. Keduanya diperiksa kembali sebelum upload; jika API tidak dapat memberikan allowance yang jelas, pengiriman dihentikan. Jangan mengirim berulang untuk mengisi slot.

Untuk inferensi CPU yang panjang, tambahkan `--cache-dir kaggle_kernel_outputs/ensemble_cache`. Hasil probabilitas setiap checkpoint dan split disimpan setelah satu model selesai; jalankan ulang perintah yang sama jika proses terputus. Cache diidentifikasi oleh hash checkpoint, urutan metadata, kode transformasi/model, versi library, perangkat, dan lokasi data, serta tidak di-commit. Dataset pada lokasi tersebut harus tetap sama selama cache digunakan.
