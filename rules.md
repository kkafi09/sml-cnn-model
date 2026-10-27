# Aturan Proyek — Tugas 2 Supervised Machine Learning

Sumber: dokumen **Tugas Supervised Machine Learning 2026**, tab **TUGAS 2**.

## Ketentuan kompetisi

- Pengerjaan dilakukan secara individu.
- Maksimal 5 submission Kaggle per hari.
- Deadline pengumpulan: 4 November 2026 pukul 22.00 WIB.
- Kompetisi Kaggle: https://www.kaggle.com/t/ec93bc56cc9842959bb6728723d23d29
- Pengumpulan: https://forms.gle/EBzuD8qaGuKgFoVt5

## Ketentuan teknis wajib

- Gunakan PyTorch.
- Model harus berbasis neural network. MLP, CNN, Transformer, atau arsitektur neural network lain diperbolehkan.
- Arsitektur boleh dibuat sendiri atau menggunakan serta memodifikasi arsitektur yang sudah ada.
- Training boleh dilakukan dari awal atau dengan fine-tuning model pre-trained.
- Ikuti workflow pengerjaan sesuai contoh praktikum.
- Setiap function dan class dalam file Python (`.py`) harus memiliki docstring.

## Data, checkpoint, dan repository

- Jangan commit dataset besar, khususnya file berukuran lebih dari 100 MB.
- Tambahkan `data/` dan `checkpoint/` ke `.gitignore`; jangan mengabaikan file kecil yang memang dibutuhkan untuk menjelaskan struktur repository tanpa alasan.
- Dokumentasikan cara memperoleh atau mengunduh data di `README.md`.
- Jaga struktur repository, syntax, notebook, dan workflow analisis tetap jelas dan sesuai contoh praktikum.
- Sertakan petunjuk penggunaan repository di `README.md`.
- Sertakan laporan interpretasi hasil di `README.md`.

## Kriteria penilaian

| Komponen | Bobot |
| --- | ---: |
| Public leaderboard score | 10% |
| Private leaderboard score | 20% |
| Kesesuaian struktur, syntax, dan notebook pada GitHub | 70% |

Komponen repository mencakup struktur direktori, workflow analisis, docstring pada function/class, petunjuk penggunaan repository di README, dan laporan interpretasi di README.

## Panduan implementasi untuk agent

- Pertahankan solusi dalam ekosistem PyTorch dan neural network; jangan mengganti model utama dengan model non-neural-network.
- Saat menambah atau mengubah file `.py`, tulis docstring untuk setiap function dan class.
- Jangan menambahkan data atau checkpoint hasil training ke Git. Perbarui `.gitignore` jika artefak baru memerlukan pengecualian.
- Buat perubahan yang menjaga alur eksperimen mudah diikuti, dan dokumentasikan penggunaan, sumber data, serta interpretasi hasil pada README saat relevan.
- Hormati batas 5 submission per hari. Jangan mengirim submission ke Kaggle secara otomatis tanpa permintaan eksplisit.
