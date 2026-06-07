# Face Recognition Using HOG and SVM

Repository ini berisi program **pengenalan wajah (face recognition)** menggunakan metode **HOG (Histogram of Oriented Gradients)** sebagai ekstraksi fitur dan **SVM (Support Vector Machine)** sebagai model klasifikasi.

Program ini dibuat menggunakan Python dan OpenCV. Sistem dapat melakukan proses training dari dataset wajah, menyimpan model hasil training, lalu melakukan pengenalan wajah secara real-time menggunakan webcam.

---

## Deskripsi Projek

Projek ini bertujuan untuk mengenali identitas wajah seseorang berdasarkan dataset gambar wajah yang sudah disiapkan sebelumnya.

Secara umum, sistem bekerja dengan cara:

1. Membaca dataset gambar wajah dari folder `dataset/`
2. Melakukan preprocessing gambar
3. Melakukan augmentasi data
4. Mengekstrak fitur wajah menggunakan HOG
5. Melatih model klasifikasi menggunakan SVM
6. Menyimpan model hasil training ke file `.pkl`
7. Menggunakan webcam untuk mendeteksi dan mengenali wajah secara real-time

---

## Metode yang Digunakan

### 1. Haar Cascade

Haar Cascade digunakan untuk mendeteksi lokasi wajah pada gambar atau frame webcam.

Pada program ini, Haar Cascade digunakan saat proses testing real-time menggunakan webcam. Ketika wajah terdeteksi, area wajah tersebut akan dipotong, lalu dikirim ke model untuk dikenali.

---

### 2. HOG (Histogram of Oriented Gradients)

HOG digunakan untuk mengekstraksi fitur dari gambar wajah.

Metode ini bekerja dengan mengambil informasi bentuk, tepi, dan arah gradien pada gambar. Fitur hasil ekstraksi HOG kemudian digunakan sebagai input untuk model SVM.
