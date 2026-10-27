# Panduan Agent dan Indeks Proyek

Baca dokumen ini sebelum mengerjakan repository. Instruksi proyek yang berlaku dirangkum di [`rules.md`](rules.md), bersumber dari tab **TUGAS 2** pada dokumen tugas Supervised Machine Learning 2026.

## Indeks

- [`rules.md`](rules.md) — aturan kompetisi, batas submission, persyaratan PyTorch/neural network, aturan data/checkpoint, kriteria penilaian, dan panduan implementasi.
- [`workflow.md`](workflow.md) — langkah staging, konfigurasi, push, pemantauan, dan pengambilan output training Kaggle GPU.
- [`README.md`](README.md) — dokumentasi utama proyek; lengkapi dengan cara memperoleh data, cara menjalankan, dan interpretasi hasil.
- [`assets/icon.svg`](assets/icon.svg) — aset ikon proyek.

## Ringkasan tugas

Bangun model supervised computer vision untuk mengklasifikasikan halaman dokumen hasil scan ke 16 kelas. Dataset yang diberikan berisi 32.003 gambar train dan 7.994 gambar evaluasi/test. Metrik kompetisi adalah categorical accuracy: jumlah prediksi benar dibagi jumlah sampel.

| ID | Kelas | ID | Kelas |
| ---: | --- | ---: | --- |
| 0 | advertisement | 8 | memo |
| 1 | budget | 9 | news_article |
| 2 | email | 10 | presentation |
| 3 | file_folder | 11 | questionnaire |
| 4 | form | 12 | resume |
| 5 | handwritten | 13 | scientific_publication |
| 6 | invoice | 14 | scientific_report |
| 7 | letter | 15 | specification |

Label harus tetap berupa integer 0–15 dan mengikuti mapping ini pada training, evaluasi, dan submission.

## Alur kerja agent

Ikuti urutan berikut agar eksperimen dapat direproduksi dan hasilnya dapat ditinjau:

1. Baca `AGENTS.md` dan `rules.md`, lalu inspeksi struktur data dan kolom CSV sebelum mengubah kode. Jangan menebak nama kolom atau struktur gambar.
2. Pastikan `data/` tersedia. Unduh dengan Kaggle CLI: `kaggle competitions download -c document-image-classification-sd234504 -p data`, lalu ekstrak ZIP ke `data/` bila belum diekstrak. Jangan menaruh kredensial Kaggle di repository.
3. Buat eksplorasi data ringkas: jumlah baris, distribusi kelas, contoh gambar, dimensi/kanal, dan hubungan ID gambar dengan CSV. Laporkan potensi imbalance atau data bermasalah.
4. Tetapkan validasi yang reproducible dan stratified. Pisahkan train/validation sebelum augmentasi; jangan memakai test untuk pemilihan model atau tuning.
5. Implementasikan pipeline PyTorch yang jelas: dataset/dataloader, transformasi train dan validation terpisah, model neural network, loss, optimizer, checkpoint, serta seed/config yang tercatat. Setiap function/class Python wajib memiliki docstring.
6. Latih dan evaluasi pada validation dengan accuracy sebagai metrik utama. Simpan log eksperimen dan checkpoint di `checkpoint/`; bandingkan eksperimen pada split dan prosedur yang konsisten.
7. Setelah model dipilih, latih/fine-tune sesuai keputusan eksperimen, prediksi seluruh test, dan buat CSV submission dengan kolom tepat `id,label`. Pastikan jumlah baris, keunikan/urutan ID sesuai `test.csv`, label integer 0–15, dan tidak ada nilai kosong.
8. Perbarui README dengan cara menjalankan, konfigurasi eksperimen, hasil validation, interpretasi error, dan lokasi artefak. Jangan mengklaim pengujian atau skor yang belum benar-benar dijalankan.
9. Jangan kirim submission ke Kaggle kecuali diminta secara eksplisit. Jika diminta, cek batas 5 submission per hari dan konfirmasi hasil yang dikembalikan Kaggle.

Gunakan Kaggle CLI untuk aktivitas Kaggle dalam repository ini. Jangan mengandalkan konfigurasi Kaggle MCP proyek karena MCP tidak dikonfigurasi di `.codex/config.toml`.

## Cara bekerja di repository ini

1. Pahami file dan struktur yang sudah ada sebelum mengubahnya. Jangan mengasumsikan direktori yang dicontohkan dalam dokumen tugas sudah tersedia; tambahkan struktur bila memang dibutuhkan oleh implementasi.
2. Ikuti semua ketentuan di `rules.md`. Bila instruksi tugas bertentangan dengan preferensi implementasi umum, ketentuan tugas menjadi acuan.
3. Pertahankan eksperimen dan alur analisis agar mudah ditelusuri. Perbarui `README.md` bila perubahan memengaruhi instalasi, data, penggunaan, atau interpretasi.
4. Pastikan setiap function dan class di file `.py` memiliki docstring.
5. Jangan commit dataset besar atau checkpoint. Atur `.gitignore` untuk `data/` dan `checkpoint/` sesuai aturan.
6. Batas Kaggle adalah 5 submission per hari. Jangan melakukan submission eksternal kecuali diminta secara eksplisit.
7. Jangan menyatakan pengujian atau verifikasi telah dilakukan jika belum dijalankan.

## Cakupan instruksi

Instruksi ini berlaku untuk seluruh repository. Jika kelak ada file `AGENTS.md` yang lebih dekat di subdirektori, baca juga instruksi tersebut untuk cakupan subdirektori itu.
