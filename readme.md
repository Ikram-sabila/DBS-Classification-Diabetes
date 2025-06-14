# Laporan Proyek Machine Learning - Muhammad Ikram Sabila Rasyad

## 📁 **Domain Proyek**

### **Latar Belakang**

Diabetes Mellitus tipe 2 adalah penyakit kronis yang ditandai oleh kadar gula darah tinggi akibat gangguan produksi atau penggunaan insulin. Menurut WHO, diabetes menyebabkan 1,5 juta kematian per tahun di seluruh dunia dan jumlahnya terus meningkat, terutama di negara berkembang \[1]. Deteksi dini terhadap diabetes sangat penting untuk mencegah komplikasi serius seperti penyakit jantung, gagal ginjal, dan kebutaan.

Sayangnya, banyak penderita diabetes tidak menyadari kondisi mereka karena gejalanya yang bersifat “diam-diam.” Oleh karena itu, sistem skrining otomatis berbasis data medis dapat membantu tenaga medis mengidentifikasi risiko diabetes secara lebih cepat dan efisien.

Dataset **Pima Indians Diabetes** dari UCI Machine Learning Repository mencatatkan berbagai indikator medis dari perempuan keturunan Pima Indian yang digunakan untuk memprediksi apakah mereka menderita diabetes atau tidak.

---

### **Urgensi Permasalahan**

* **Mengapa harus diselesaikan?** Karena diabetes sering tidak terdiagnosis hingga stadium lanjut. Deteksi dini dapat mengurangi beban biaya dan meningkatkan kualitas hidup pasien.
* **Bagaimana menyelesaikannya?** Dengan membangun sistem machine learning yang mampu memprediksi risiko diabetes dari fitur-fitur seperti BMI, usia, kadar glukosa, tekanan darah, dll.

---

### **Referensi**

\[1] World Health Organization. (2023). *Diabetes*. \[Online] Available: [https://www.who.int/news-room/fact-sheets/detail/diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)
\[2] Smith, B. et al. (2020). “Machine Learning for Early Detection of Diabetes,” *Journal of Biomedical Informatics*, vol. 102, pp. 103-110.
\[3] National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). (2022). *Diabetes Statistics*.

---

## 💼 **Business Understanding**

### **Problem Statements**

1. **Bagaimana cara memprediksi apakah seseorang menderita diabetes berdasarkan data medis?**
2. **Model machine learning apa yang paling efektif digunakan untuk prediksi diabetes pada dataset Pima Indians?**
3. **Bagaimana kita dapat meningkatkan performa model baseline agar hasilnya dapat diandalkan untuk tujuan skrining medis?**

---

### **Goals**

1. **Membangun sistem prediksi diabetes menggunakan supervised machine learning** yang memanfaatkan fitur medis seperti BMI, Glucose, dan Age.
2. **Mengevaluasi dan membandingkan performa beberapa algoritma klasifikasi** seperti Logistic Regression, Random Forest, SVM, dan KNN.
3. **Melakukan hyperparameter tuning untuk meningkatkan akurasi dan sensitivitas** model terbaik, agar risiko false negative dapat ditekan.

---

### **Solution Statements**

Untuk mencapai goals di atas, dilakukan pendekatan sebagai berikut:

#### ✅ **Solution Statement 1: Multi-Model Comparison**

Menguji beberapa model klasifikasi (Logistic Regression, KNN, SVM, Random Forest) pada dataset yang telah dibersihkan dan dinormalisasi.
→ Tujuan: menemukan model dengan kombinasi **precision, recall, dan ROC-AUC** terbaik.

#### ✅ **Solution Statement 2: Model Improvement dengan Hyperparameter Tuning**

Melakukan tuning pada model terbaik (Random Forest) menggunakan **GridSearchCV** untuk meningkatkan kinerja.
→ Tujuan: menurunkan false negative dan meningkatkan **F1-score** serta **ROC AUC**.

#### ✅ **Metrik Evaluasi yang Digunakan**

Model diukur dengan metrik:

* **Recall (Sensitivity)** → prioritas utama agar penderita tidak terlewat.
* **Precision** → menjaga efisiensi tes lanjutan.
* **F1-Score** → keseimbangan.
* **ROC-AUC** → evaluasi global kemampuan model membedakan kelas.
Berikut versi **sederhana** dari bagian **Data Understanding**, tanpa menyertakan kode:

---

## 📊 **Data Understanding**

### **1. Informasi Dataset**

Dataset yang digunakan dalam proyek ini adalah **Pima Indians Diabetes Dataset** dari UCI Machine Learning Repository, yang juga tersedia di Kaggle. Dataset ini berisi informasi medis dari 768 perempuan keturunan suku Pima Indian di Amerika Serikat, dengan usia minimal 21 tahun. Data ini digunakan untuk memprediksi apakah seseorang menderita diabetes atau tidak berdasarkan berbagai indikator kesehatan.

📎 Sumber data: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

### **2. Deskripsi Variabel**

Berikut adalah variabel-variabel yang terdapat pada dataset:

* **Pregnancies**: Jumlah kehamilan yang pernah dialami pasien.
* **Glucose**: Kadar glukosa dalam darah setelah tes toleransi glukosa oral.
* **BloodPressure**: Tekanan darah diastolik (dalam mm Hg).
* **SkinThickness**: Ketebalan lipatan kulit trisep (dalam mm).
* **Insulin**: Kadar insulin dalam darah 2 jam setelah makan.
* **BMI**: Indeks massa tubuh, sebagai indikator obesitas.
* **DiabetesPedigreeFunction**: Skor risiko diabetes berdasarkan riwayat keluarga.
* **Age**: Usia pasien (dalam tahun).
* **Outcome**: Label target, 1 menunjukkan pasien menderita diabetes, dan 0 menunjukkan tidak.

---

### **3. Exploratory Data Analysis (EDA)**

Untuk memahami karakteristik data, dilakukan analisis awal berupa visualisasi distribusi masing-masing fitur berdasarkan status diabetes (Outcome). Hasilnya menunjukkan bahwa:

* Pasien diabetes cenderung memiliki kadar glukosa dan BMI yang lebih tinggi dibandingkan yang tidak.
* Usia juga menjadi faktor penting, di mana pasien diabetes umumnya lebih tua.

Selain itu, analisis korelasi menunjukkan bahwa fitur yang paling berkaitan dengan diabetes adalah **Glucose**, **BMI**, dan **Age**. Korelasi antar fitur relatif rendah, sehingga tidak terdapat masalah multikolinearitas yang signifikan.

## 🧹 **Data Preparation**

Agar model machine learning dapat dilatih secara optimal, diperlukan beberapa tahapan persiapan data (data preparation). Berikut ini adalah teknik-teknik yang diterapkan secara berurutan dalam proyek ini:

---

### **1. Winsorizing**

Beberapa fitur seperti **Insulin**, **SkinThickness**, dan **BMI** menunjukkan adanya nilai ekstrem (outlier) yang dapat memengaruhi performa model secara negatif. Oleh karena itu, dilakukan teknik **winsorizing** untuk membatasi nilai-nilai ekstrem pada batas persentil tertentu (misalnya 1% dan 99%).

🔹 **Tujuan**: Mengurangi pengaruh outlier tanpa membuang data.

---

### **2. Normalisasi (Min-Max Scaling)**

Karena sebagian besar model machine learning peka terhadap skala data, seluruh fitur numerik dinormalisasi ke rentang 0 hingga 1.

🔹 **Tujuan**: Membuat semua fitur berada dalam skala yang sama agar model seperti KNN dan SVM dapat bekerja optimal.

---

### **3. SMOTE (Synthetic Minority Oversampling Technique)**

Dataset ini memiliki ketidakseimbangan kelas, di mana jumlah pasien non-diabetes lebih banyak daripada yang diabetes. Untuk mengatasi hal ini, digunakan teknik **SMOTE** untuk menambahkan data sintetis pada kelas minoritas.

🔹 **Tujuan**: Menyeimbangkan proporsi kelas agar model tidak bias terhadap kelas mayoritas.

---

### **4. PCA (Principal Component Analysis)**

Dilakukan teknik reduksi dimensi menggunakan **PCA** untuk mereduksi kompleksitas fitur, menghilangkan multikolinearitas, dan mempercepat waktu pelatihan model. Komponen utama yang dipilih mampu menjelaskan sebagian besar variasi data.

🔹 **Tujuan**: Menyederhanakan data tanpa kehilangan informasi penting dan mempercepat proses pelatihan model.

---

### **5. Train-Test Split**

Dataset dibagi menjadi dua bagian:

* **80% data untuk pelatihan (train)**
* **20% data untuk pengujian (test)**

Pembagian ini dilakukan secara acak namun terkontrol agar proporsi kelas tetap seimbang.

🔹 **Tujuan**: Menyediakan data uji yang independen untuk mengevaluasi kinerja model secara objektif.

## 🤖 **Modeling**

Pada tahap ini, dilakukan pembangunan model machine learning untuk memprediksi apakah seorang pasien menderita diabetes atau tidak berdasarkan data yang telah dipersiapkan.

### **1. Model yang Digunakan**

Beberapa algoritma digunakan untuk membandingkan performanya, yaitu:

#### a. Logistic Regression

* Algoritma dasar klasifikasi linear.
* Parameter utama: `max_iter=1000` agar proses konvergensi stabil.
* **Kelebihan**: Sederhana, mudah diinterpretasikan, cocok untuk baseline.
* **Kekurangan**: Kurang mampu menangkap hubungan non-linear antar fitur.

#### b. Random Forest

* Model ensemble berbasis pohon keputusan.
* Parameter utama: `n_estimators=100`, `random_state=42`.
* **Kelebihan**: Tahan terhadap overfitting, dapat menangani data non-linear dan fitur tidak terstandar.
* **Kekurangan**: Interpretasi model lebih kompleks dibanding Logistic Regression.

#### c. K-Nearest Neighbors (KNN)

* Algoritma berbasis kedekatan jarak antar data.
* Parameter utama: `n_neighbors=5`.
* **Kelebihan**: Mudah dipahami, tidak memerlukan proses pelatihan eksplisit.
* **Kekurangan**: Sensitif terhadap skala data dan noise, waktu prediksi lambat pada data besar.

#### d. Support Vector Machine (SVM)

* Algoritma klasifikasi yang mencari hyperplane terbaik untuk memisahkan kelas.
* Parameter: `kernel='rbf'`, `C=1`, `gamma='scale'`.
* **Kelebihan**: Sangat baik dalam klasifikasi pada data berdimensi tinggi.
* **Kekurangan**: Cukup mahal secara komputasi dan kurang transparan dalam interpretasi.

---

### **2. Evaluasi Awal Model**

Setiap model dievaluasi menggunakan teknik **cross-validation** sebanyak 5-fold pada data latih. Metode ini memberikan estimasi akurasi rata-rata yang stabil untuk masing-masing model sebelum dilakukan pelatihan akhir.

---

### **3. Pemilihan Model Terbaik**

Dari hasil evaluasi, **Random Forest** menunjukkan performa terbaik secara konsisten dibandingkan model lainnya. Oleh karena itu, model ini dipilih sebagai **solusi utama**.

---

### **4. Improvement Model**

Model Random Forest kemudian di-*improve* menggunakan teknik **Grid Search** untuk melakukan **hyperparameter tuning** terhadap parameter seperti `max_depth`, `min_samples_split`, dan `n_estimators`.

🔍 Tujuan dari proses ini adalah untuk menemukan kombinasi parameter terbaik agar akurasi dan generalisasi model meningkat.

---

### **Kesimpulan**

Dengan melakukan perbandingan beberapa model dan melakukan tuning pada model terbaik, solusi yang dihasilkan menjadi lebih kuat dan andal dalam memprediksi risiko diabetes berdasarkan data pasien.

## 📊 **Evaluation**

Untuk mengevaluasi performa model klasifikasi dalam memprediksi diabetes, digunakan beberapa metrik evaluasi yang relevan, yaitu:

---

### **1. Accuracy (Akurasi)**

Akurasi mengukur seberapa banyak prediksi yang benar dari seluruh prediksi yang dilakukan.

📌 **Formula**:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

📈 **Hasil**:
Model mencapai **akurasi sebesar 81%**, yang menunjukkan bahwa 81% prediksi model terhadap data uji adalah benar.

---

### **2. Precision**

Precision mengukur seberapa banyak prediksi positif yang benar-benar positif. Ini penting dalam konteks medis agar tidak terlalu banyak pasien yang sehat diklasifikasikan sebagai sakit.

📌 **Formula**:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

📈 **Hasil**:

* Precision kelas 0 (Non-Diabetes): **0.83**
* Precision kelas 1 (Diabetes): **0.79**

---

### **3. Recall (Sensitivity / True Positive Rate)**

Recall mengukur seberapa banyak kasus positif yang berhasil terdeteksi oleh model. Dalam konteks diabetes, recall penting karena menyangkut deteksi pasien yang benar-benar mengidap diabetes.

📌 **Formula**:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

📈 **Hasil**:

* Recall kelas 0: **0.77**
* Recall kelas 1: **0.84**

---

### **4. F1-Score**

F1-Score adalah harmonic mean antara precision dan recall, digunakan ketika kita ingin mempertimbangkan keduanya secara seimbang.

📌 **Formula**:

$$
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

📈 **Hasil**:

* F1 kelas 0: **0.80**
* F1 kelas 1: **0.81**

---

### **5. ROC AUC Score**
![ROC AUC Score](ROC_Curve.png)
ROC AUC (Receiver Operating Characteristic - Area Under Curve) mengukur kemampuan model membedakan antara kelas positif dan negatif.

📌 **Rentang nilai**: 0.5 (random) hingga 1.0 (sempurna)
📈 **Hasil**: Model mencapai **ROC AUC Score sebesar 0.8872**, yang menunjukkan bahwa model memiliki **kemampuan klasifikasi yang sangat baik**.

---

### **6. Confusion Matrix**

Berdasarkan confusion matrix:

|                         | Predicted Non-Diabetes | Predicted Diabetes |
| ----------------------- | ---------------------- | ------------------ |
| **Actual Non-Diabetes** | 76                     | 23                 |
| **Actual Diabetes**     | 16                     | 85                 |

* **True Positives (TP)**: 85
* **True Negatives (TN)**: 76
* **False Positives (FP)**: 23
* **False Negatives (FN)**: 16

Confusion matrix memberikan gambaran nyata tentang kesalahan dan keberhasilan model dalam klasifikasi dua kelas.

---

### **Kesimpulan**

Model Random Forest yang digunakan mampu mencapai performa yang sangat baik, dengan nilai **akurasi 81%** dan **ROC AUC 0.8872**. Ini menunjukkan bahwa model cukup andal dalam memprediksi diabetes. Selain itu, nilai recall yang tinggi untuk kelas diabetes (0.84) menunjukkan bahwa model cukup sensitif dalam mendeteksi pasien yang benar-benar mengidap penyakit ini, yang sangat penting dalam konteks medis.