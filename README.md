# Face Recognition Using HOG and SVM

Repository ini berisi program **pengenalan wajah (face recognition)** menggunakan metode **HOG (Histogram of Oriented Gradients)** sebagai ekstraksi fitur dan **SVM (Support Vector Machine)** sebagai model klasifikasi.

Program ini dibuat menggunakan Python dan OpenCV. Sistem dapat melakukan proses training dari dataset wajah, menyimpan model hasil training, lalu melakukan pengenalan wajah secara real-time menggunakan webcam.

---

## Deskripsi Projek

Projek ini bertujuan untuk mengenali identitas wajah seseorang berdasarkan dataset gambar wajah yang sudah disiapkan sebelumnya. Setiap orang memiliki folder dataset masing-masing, sehingga sistem dapat mempelajari ciri wajah berdasarkan nama folder tersebut.

Secara umum, sistem bekerja dengan membaca gambar wajah dari dataset, mengolah gambar agar memiliki ukuran yang sama, memperbanyak variasi data melalui augmentasi, mengambil ciri wajah menggunakan HOG, lalu melatih model SVM untuk mengenali identitas wajah.

Setelah model selesai dilatih, model akan disimpan ke dalam file khusus agar dapat digunakan kembali tanpa perlu melakukan training ulang. Model tersebut kemudian dapat digunakan untuk mengenali wajah secara real-time melalui webcam.

---

## Metode yang Digunakan

### Haar Cascade

Haar Cascade digunakan untuk mendeteksi keberadaan wajah pada frame webcam. Ketika wajah berhasil terdeteksi, sistem akan mengambil area wajah tersebut untuk diproses lebih lanjut.

Pada projek ini, Haar Cascade hanya digunakan untuk mendeteksi posisi wajah, bukan untuk mengenali identitas wajah. Proses pengenalan tetap dilakukan menggunakan model SVM yang sudah dilatih sebelumnya.

### HOG (Histogram of Oriented Gradients)

HOG digunakan untuk mengambil fitur atau ciri penting dari gambar wajah. Metode ini memperhatikan bentuk, tepi, tekstur, dan arah gradien pada gambar.

Hasil dari proses HOG berupa kumpulan nilai fitur yang mewakili karakteristik wajah. Fitur inilah yang kemudian digunakan sebagai data masukan untuk model SVM.

### SVM (Support Vector Machine)

SVM digunakan sebagai model klasifikasi untuk membedakan wajah antar orang berdasarkan fitur HOG yang telah diekstraksi.

Pada projek ini, SVM digunakan karena cocok untuk data berukuran kecil sampai menengah dan dapat bekerja cukup baik pada data fitur seperti HOG. Model SVM juga dibuat agar dapat menghasilkan nilai confidence, sehingga sistem dapat menentukan apakah wajah termasuk orang yang dikenali atau dianggap sebagai Unknown.

---

## Struktur Dataset

Dataset disimpan di dalam folder `dataset`. Setiap subfolder di dalam folder dataset merupakan label atau nama orang yang akan dikenali oleh sistem.

Contoh struktur dataset:

- dataset
  - Andi
    - foto wajah Andi
  - Budi
    - foto wajah Budi
  - Siti
    - foto wajah Siti

Dengan struktur seperti ini, sistem akan menganggap nama folder sebagai label kelas. Jika terdapat folder bernama Andi, maka semua gambar di dalam folder tersebut akan dianggap sebagai data wajah Andi.

---

## Alur Kerja Sistem

Alur kerja sistem pada projek ini terdiri dari beberapa tahap utama, yaitu:

1. Membaca dataset wajah
2. Melakukan preprocessing gambar
3. Melakukan augmentasi data
4. Mengekstraksi fitur wajah menggunakan HOG
5. Mengubah label nama menjadi bentuk numerik
6. Membagi data menjadi data training dan data testing
7. Melatih model menggunakan SVM
8. Mengevaluasi hasil model
9. Menyimpan model ke file
10. Melakukan pengenalan wajah secara real-time menggunakan webcam

---

## Proses Membaca Dataset

Program akan membaca semua gambar yang terdapat di dalam folder dataset. Setiap subfolder akan dianggap sebagai satu kelas atau satu identitas orang.

Format gambar yang dapat dibaca oleh program antara lain JPG, JPEG, PNG, dan BMP. Jika folder dataset tidak ditemukan atau tidak memiliki subfolder, maka program akan menampilkan pesan error.

---

## Proses Preprocessing

Sebelum gambar digunakan untuk training, gambar akan diproses terlebih dahulu agar memiliki format yang seragam.

Tahapan preprocessing yang dilakukan adalah:

1. Membaca gambar dari dataset
2. Mengubah gambar menjadi grayscale
3. Mengubah ukuran gambar menjadi 64 x 64 piksel

Grayscale digunakan karena proses ekstraksi fitur HOG lebih berfokus pada bentuk, tepi, dan intensitas piksel, bukan pada warna gambar.

---

## Proses Augmentasi Data

Augmentasi data digunakan untuk menambah variasi gambar dari dataset yang tersedia. Tujuannya adalah agar model lebih kuat dalam mengenali wajah pada kondisi yang berbeda.

Augmentasi yang dilakukan pada projek ini meliputi:

1. Gambar asli
2. Flip horizontal
3. Gambar dibuat lebih terang
4. Gambar dibuat lebih gelap
5. Rotasi kecil ke kanan
6. Rotasi kecil ke kiri

Dengan proses ini, satu gambar wajah dapat menghasilkan beberapa variasi gambar baru. Hal ini membantu model agar tidak hanya bergantung pada satu kondisi gambar saja.

---

## Proses Ekstraksi Fitur HOG

Setelah gambar diproses dan diaugmentasi, setiap gambar akan diekstraksi fiturnya menggunakan HOG.

HOG akan mengambil pola bentuk dan tepi dari wajah, kemudian mengubahnya menjadi data fitur dalam bentuk angka. Data fitur ini menjadi dasar bagi model SVM untuk membedakan wajah satu orang dengan orang lainnya.

---

## Proses Label Encoding

Label wajah yang awalnya berupa nama orang akan diubah menjadi angka. Proses ini diperlukan karena model machine learning tidak dapat langsung memproses label dalam bentuk teks.

Sebagai contoh, nama Andi, Budi, dan Siti akan diubah menjadi label numerik. Setelah prediksi selesai, label numerik tersebut dapat dikembalikan lagi menjadi nama asli menggunakan label encoder.

---

## Proses Training Model

Setelah fitur dan label siap, data akan dibagi menjadi data training dan data testing. Pada projek ini, data dibagi menjadi 80% data training dan 20% data testing.

Data training digunakan untuk melatih model SVM, sedangkan data testing digunakan untuk menguji kemampuan model dalam mengenali data yang belum pernah dilihat sebelumnya.

Model SVM kemudian dilatih menggunakan fitur HOG dari setiap gambar wajah. Setelah proses training selesai, model akan dapat mengenali pola fitur dari masing-masing orang yang ada di dataset.

---

## Proses Evaluasi Model

Setelah model selesai dilatih, program akan melakukan evaluasi menggunakan data testing.

Evaluasi yang ditampilkan meliputi:

1. Akurasi model
2. Precision
3. Recall
4. F1-score
5. Jumlah data pada setiap kelas

Evaluasi ini digunakan untuk melihat seberapa baik model dalam mengenali wajah berdasarkan dataset yang digunakan.

---

## Proses Penyimpanan Model

Model yang sudah selesai dilatih akan disimpan ke dalam file `model_face_recognition.pkl`.

File ini berisi model SVM dan label encoder yang sudah digunakan saat training. Dengan adanya file model ini, pengguna tidak perlu melakukan training ulang setiap kali ingin menggunakan sistem pengenalan wajah.

---

## Proses Memuat Model

Ketika pengguna memilih menu testing webcam, program akan memuat model yang sudah disimpan sebelumnya.

Jika file model belum ada, maka program akan menampilkan pesan bahwa model belum ditemukan dan pengguna harus melakukan training terlebih dahulu.

---

## Proses Prediksi Wajah

Pada saat wajah terdeteksi dari webcam, area wajah tersebut akan dipotong dan diproses. Wajah akan diubah menjadi grayscale, diubah ukurannya, lalu diekstraksi fiturnya menggunakan HOG.

Fitur wajah tersebut kemudian dimasukkan ke dalam model SVM untuk diprediksi. Model akan menghasilkan nama orang dan nilai confidence.

Jika nilai confidence lebih rendah dari batas yang ditentukan, maka wajah akan dianggap sebagai Unknown. Jika nilai confidence cukup tinggi, maka sistem akan menampilkan nama orang yang dikenali.

---

## Proses Real-time Webcam

Pada mode real-time, sistem akan membuka webcam dan membaca frame secara terus-menerus.

Setiap frame akan diperiksa untuk mendeteksi wajah. Jika wajah ditemukan, sistem akan menggambar kotak di sekitar wajah dan menampilkan nama hasil prediksi beserta nilai confidence.

Warna kotak yang digunakan:

- Hijau untuk wajah yang berhasil dikenali
- Merah untuk wajah yang dianggap Unknown

Untuk keluar dari mode webcam, pengguna dapat menekan tombol Q.

---

## Cara Menjalankan Program

1. Siapkan folder dataset
2. Masukkan gambar wajah ke dalam subfolder berdasarkan nama orang
3. Jalankan program Python
4. Pilih menu training untuk melatih model
5. Setelah training selesai, pilih menu testing webcam
6. Arahkan wajah ke kamera
7. Sistem akan menampilkan hasil pengenalan wajah

---

## Menu Program

### 1. Training Model dari Dataset

Menu ini digunakan untuk melatih model berdasarkan gambar wajah yang ada di folder dataset.

Pada proses ini, sistem akan membaca dataset, melakukan preprocessing, augmentasi, ekstraksi fitur HOG, training model SVM, evaluasi model, dan menyimpan model hasil training.

### 2. Testing Real-time Menggunakan Webcam

Menu ini digunakan untuk melakukan pengenalan wajah secara langsung melalui webcam.

Sistem akan memuat model yang sudah dilatih, membuka webcam, mendeteksi wajah, lalu menampilkan hasil prediksi nama dan confidence.

### 3. Keluar

Menu ini digunakan untuk keluar dari program.

---

## Kelebihan Projek

Beberapa kelebihan dari projek ini adalah:

1. Dapat melakukan pengenalan wajah secara real-time
2. Menggunakan metode HOG yang cukup ringan
3. Menggunakan SVM yang cocok untuk dataset kecil hingga menengah
4. Memiliki proses augmentasi data
5. Model dapat disimpan dan digunakan kembali
6. Dapat memberikan label Unknown untuk wajah yang confidence-nya rendah
7. Cocok digunakan sebagai pembelajaran dasar computer vision dan machine learning

---

## Keterbatasan Projek

Beberapa keterbatasan dari projek ini adalah:

1. Akurasi sangat bergantung pada kualitas dataset
2. Pencahayaan yang buruk dapat memengaruhi hasil prediksi
3. Posisi wajah yang terlalu miring dapat membuat sistem sulit mengenali wajah
4. Haar Cascade tidak selalu akurat pada semua kondisi
5. Belum menggunakan metode deep learning
6. Dataset yang terlalu sedikit dapat menyebabkan model kurang stabil

---

## Tips Pengambilan Dataset

Agar hasil pengenalan wajah lebih baik, dataset sebaiknya memiliki variasi yang cukup.

Beberapa tips dalam membuat dataset:

1. Gunakan foto wajah yang jelas
2. Ambil foto dari beberapa sudut wajah
3. Gunakan variasi pencahayaan
4. Gunakan beberapa ekspresi wajah
5. Hindari foto yang terlalu blur
6. Pastikan wajah tidak terlalu tertutup
7. Gunakan jumlah foto yang cukup untuk setiap orang

---

## Teknologi yang Digunakan

Teknologi dan library yang digunakan pada projek ini antara lain:

1. Python
2. OpenCV
3. NumPy
4. Scikit-image
5. Scikit-learn
6. Pickle
7. Haar Cascade
8. HOG
9. SVM

---

## Kesimpulan

Projek ini merupakan implementasi sistem pengenalan wajah menggunakan metode HOG dan SVM. HOG digunakan untuk mengekstraksi fitur wajah, sedangkan SVM digunakan untuk melakukan klasifikasi identitas wajah.

Sistem ini mampu melakukan training dari dataset wajah, menyimpan model hasil training, dan melakukan pengenalan wajah secara real-time menggunakan webcam.

Projek ini cocok digunakan sebagai pembelajaran dasar mengenai computer vision, ekstraksi fitur, klasifikasi machine learning, dan pengenalan wajah sebelum masuk ke metode yang lebih kompleks seperti CNN atau deep learning.
