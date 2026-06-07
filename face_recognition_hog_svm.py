import os
import cv2
import numpy as np
import pickle
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# KONFIGURASI PROGRAM
# ============================================================

# Path folder dataset
DATASET_PATH = "dataset"

# Path untuk menyimpan model yang sudah dilatih
MODEL_PATH = "model_face_recognition.pkl"

# Ukuran gambar yang akan diproses (width x height)
IMG_SIZE = (64, 64)

# Threshold kepercayaan: jika confidence lebih rendah dari ini,
# wajah dianggap "Unknown"
CONFIDENCE_THRESHOLD = 0.5

# Konfigurasi HOG (Histogram of Oriented Gradients)
HOG_CONFIG = {
    "orientations": 9,      # Jumlah arah gradien
    "pixels_per_cell": (8, 8),  # Ukuran sel piksel
    "cells_per_block": (2, 2),  # Ukuran blok sel
}


# ============================================================
# FUNGSI: AUGMENTASI DATA
# Data augmentation membantu model lebih robust terhadap variasi
# ============================================================

def augment_image(image):
    """
    Menghasilkan variasi gambar untuk memperkaya dataset.
    Augmentasi yang dilakukan:
      1. Gambar asli (tanpa perubahan)
      2. Flip horizontal (cermin kiri-kanan)
      3. Perbedaan brightness (lebih terang dan lebih gelap)
      4. Rotasi kecil (ke kiri dan ke kanan)
    """
    augmented = []

    # 1. Gambar asli
    augmented.append(image)

    # 2. Flip horizontal (cermin)
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)

    # 3. Brightness lebih terang
    bright = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
    augmented.append(bright)

    # 4. Brightness lebih gelap
    dark = cv2.convertScaleAbs(image, alpha=0.7, beta=-20)
    augmented.append(dark)

    # 5. Rotasi kecil ke kanan (+10 derajat)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M_right = cv2.getRotationMatrix2D(center, -10, 1.0)
    rotated_right = cv2.warpAffine(image, M_right, (w, h))
    augmented.append(rotated_right)

    # 6. Rotasi kecil ke kiri (-10 derajat)
    M_left = cv2.getRotationMatrix2D(center, 10, 1.0)
    rotated_left = cv2.warpAffine(image, M_left, (w, h))
    augmented.append(rotated_left)

    return augmented


# ============================================================
# FUNGSI: EKSTRAKSI FITUR HOG
# HOG menangkap bentuk tepi dan tekstur wajah
# ============================================================

def extract_hog_features(image):
    """
    Mengekstrak fitur HOG dari gambar.
    Input : gambar grayscale (numpy array)
    Output: vektor fitur HOG (1D numpy array)
    """
    features = hog(
        image,
        orientations=HOG_CONFIG["orientations"],
        pixels_per_cell=HOG_CONFIG["pixels_per_cell"],
        cells_per_block=HOG_CONFIG["cells_per_block"],
        block_norm="L2-Hys",  # Normalisasi blok
        visualize=False
    )
    return features


# ============================================================
# FUNGSI: MEMUAT DATASET
# Membaca semua gambar dari folder dataset beserta labelnya
# ============================================================

def load_dataset(dataset_path):
    """
    Membaca dataset dari folder.
    Setiap subfolder = nama orang (label).
    Setiap gambar di dalamnya = data wajah.

    Return: (features, labels)
      features : list vektor HOG
      labels   : list nama orang
    """
    features = []
    labels = []

    # Cek apakah folder dataset ada
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Folder dataset '{dataset_path}' tidak ditemukan!")
        print("Buat folder 'dataset/' dan isi dengan subfolder nama orang.")
        return None, None

    print(f"\n[INFO] Memuat dataset dari: {dataset_path}")
    print("-" * 50)

    # Loop setiap subfolder (nama orang)
    person_folders = [f for f in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, f))]

    if len(person_folders) == 0:
        print("[ERROR] Tidak ada subfolder dalam dataset!")
        return None, None

    for person_name in person_folders:
        person_path = os.path.join(dataset_path, person_name)
        image_files = [f for f in os.listdir(person_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        print(f"  Memuat: {person_name} ({len(image_files)} foto)")

        count_loaded = 0
        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)

            # Baca gambar
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Konversi ke grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize ke ukuran standar
            resized = cv2.resize(gray, IMG_SIZE)

            # Augmentasi: hasilkan beberapa variasi gambar
            augmented_images = augment_image(resized)

            # Ekstrak fitur HOG dari setiap variasi
            for aug_img in augmented_images:
                hog_feat = extract_hog_features(aug_img)
                features.append(hog_feat)
                labels.append(person_name)
                count_loaded += 1

        print(f"           -> Total data setelah augmentasi: {count_loaded}")

    print("-" * 50)
    print(f"[INFO] Total data: {len(features)} sampel dari {len(person_folders)} orang")
    return np.array(features), np.array(labels)


# ============================================================
# FUNGSI: TRAINING MODEL
# Melatih SVM menggunakan fitur HOG dari dataset
# ============================================================

def train_model(dataset_path):
    """
    Melatih model SVM dengan data dari dataset.
    Model yang sudah dilatih disimpan ke file .pkl
    """
    print("\n" + "="*50)
    print("  PROSES TRAINING MODEL")
    print("="*50)

    # Muat dataset
    X, y = load_dataset(dataset_path)
    if X is None:
        return None, None

    # Encode label (nama orang -> angka)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\n[INFO] Kelas yang dilatih: {list(label_encoder.classes_)}")

    # Bagi data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"[INFO] Data training: {len(X_train)} | Data testing: {len(X_test)}")

    # Buat dan latih model SVM
    # kernel='rbf' cocok untuk data non-linear seperti wajah
    # probability=True agar bisa mendapatkan nilai kepercayaan (confidence)
    print("\n[INFO] Melatih model SVM... (mohon tunggu)")
    svm_model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    print("[INFO] Training selesai!")

    # Evaluasi model pada data test
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[HASIL] Akurasi model: {accuracy * 100:.2f}%")

    # Laporan detail per kelas
    print("\n[INFO] Laporan Klasifikasi:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))

    # Simpan model dan label encoder ke file
    model_data = {
        "svm_model": svm_model,
        "label_encoder": label_encoder
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"[INFO] Model disimpan ke: {MODEL_PATH}")

    return svm_model, label_encoder


# ============================================================
# FUNGSI: MEMUAT MODEL YANG SUDAH DISIMPAN
# ============================================================

def load_model():
    """Memuat model SVM dari file .pkl yang sudah disimpan."""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model '{MODEL_PATH}' tidak ditemukan!")
        print("Jalankan training terlebih dahulu (pilih menu 1).")
        return None, None

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    svm_model = model_data["svm_model"]
    label_encoder = model_data["label_encoder"]

    print(f"[INFO] Model berhasil dimuat dari: {MODEL_PATH}")
    print(f"[INFO] Kelas yang dikenali: {list(label_encoder.classes_)}")
    return svm_model, label_encoder


# ============================================================
# FUNGSI: PREDIKSI SATU WAJAH
# Menerima gambar wajah, mengembalikan nama dan confidence
# ============================================================

def predict_face(face_img, svm_model, label_encoder):
    """
    Memprediksi identitas dari gambar wajah.

    Input:
      face_img     : gambar wajah (BGR dari OpenCV)
      svm_model    : model SVM yang sudah dilatih
      label_encoder: encoder untuk label (angka -> nama)

    Output:
      (nama, confidence) - nama "Unknown" jika confidence rendah
    """
    # Preprocessing: grayscale dan resize
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)

    # Ekstraksi fitur HOG
    hog_features = extract_hog_features(resized).reshape(1, -1)

    # Prediksi dengan SVM
    probabilities = svm_model.predict_proba(hog_features)[0]
    max_confidence = np.max(probabilities)
    predicted_index = np.argmax(probabilities)

    # Cek threshold: jika confidence terlalu rendah -> Unknown
    if max_confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", max_confidence

    # Decode label: angka -> nama orang
    predicted_name = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_name, max_confidence


# ============================================================
# FUNGSI: DETEKSI REAL-TIME MENGGUNAKAN WEBCAM
# ============================================================

def run_webcam_detection(svm_model, label_encoder):
    """
    Membuka webcam dan melakukan deteksi wajah secara real-time.
    - Wajah dideteksi menggunakan Haar Cascade (bawaan OpenCV)
    - Setiap wajah yang terdeteksi diklasifikasi menggunakan SVM
    - Tekan 'Q' untuk keluar
    """
    print("\n[INFO] Membuka webcam...")
    print("[INFO] Tekan 'Q' untuk keluar")
    print(f"[INFO] Threshold confidence: {CONFIDENCE_THRESHOLD}")

    # Muat Haar Cascade untuk deteksi wajah
    # File ini sudah tersedia bawaan OpenCV
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("[ERROR] Haar Cascade tidak ditemukan!")
        return

    # Buka webcam (0 = webcam default)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka!")
        return

    print("[INFO] Webcam aktif. Arahkan wajah ke kamera...")

    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame dari webcam.")
            break

        # Konversi ke grayscale untuk deteksi wajah
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah menggunakan Haar Cascade
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,   # Faktor skala pencarian
            minNeighbors=5,    # Minimum tetangga agar dianggap wajah
            minSize=(50, 50)   # Ukuran minimum kotak wajah
        )

        # Proses setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Crop area wajah dari frame
            face_crop = frame[y:y+h, x:x+w]

            # Prediksi identitas wajah
            name, confidence = predict_face(face_crop, svm_model, label_encoder)

            # Tentukan warna kotak berdasarkan hasil prediksi
            if name == "Unknown":
                box_color = (0, 0, 255)   # Merah untuk Unknown
                text_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)   # Hijau untuk wajah dikenal
                text_color = (0, 255, 0)

            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

            # Tampilkan nama dan confidence
            label_text = f"{name} ({confidence*100:.1f}%)"
            cv2.putText(
                frame, label_text,
                (x, y - 10),              # Posisi teks di atas kotak
                cv2.FONT_HERSHEY_SIMPLEX,  # Jenis font
                0.7,                       # Ukuran font
                text_color,
                2                          # Ketebalan teks
            )

        # Tampilkan instruksi di sudut kiri bawah
        cv2.putText(
            frame, "Tekan 'Q' untuk keluar",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # Tampilkan frame
        cv2.imshow("Face Recognition - HOG + SVM", frame)

        # Cek tombol 'Q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan resource
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam ditutup.")


# ============================================================
# MAIN PROGRAM - MENU UTAMA
# ============================================================

def main():
    print("=" * 55)
    print("  SISTEM FACE RECOGNITION BERBASIS HOG + SVM")
    print("  Tugas Kuliah - Politeknik / Universitas")
    print("=" * 55)
    print()
    print("  [1] Training model dari dataset")
    print("  [2] Testing real-time menggunakan webcam")
    print("  [3] Keluar")
    print()

    pilihan = input("Masukkan pilihan (1/2/3): ").strip()

    if pilihan == "1":
        # ---- MODE TRAINING ----
        svm_model, label_encoder = train_model(DATASET_PATH)
        if svm_model is not None:
            print("\n[SUKSES] Model berhasil dilatih dan disimpan!")
            print("Sekarang kamu bisa menjalankan testing (pilih menu 2).")

    elif pilihan == "2":
        # ---- MODE TESTING / WEBCAM ----
        svm_model, label_encoder = load_model()
        if svm_model is not None:
            run_webcam_detection(svm_model, label_encoder)

    elif pilihan == "3":
        print("Keluar dari program. Sampai jumpa!")

    else:
        print("[ERROR] Pilihan tidak valid. Jalankan ulang program.")


# Entry point program
if __name__ == "__main__":
    main()