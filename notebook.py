#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay


# Cell ini memuat semua library yang diperlukan untuk proyek: pandas, numpy, serta matplotlib dan seaborn untuk analisis dan visualisasi; winsorize, SMOTE, dan MinMaxScaler untuk pembersihan, penyeimbangan kelas, dan normalisasi data; PCA dan train_test_split untuk reduksi dimensi dan pemisahan data; empat algoritma klasifikasi inti (Logistic Regression, Random Forest, K‑NN, SVM) beserta cross_val_score dan GridSearchCV guna evaluasi serta tuning; dan akhirnya metrik evaluasi seperti classification_report, confusion_matrix, dan roc_auc_score (plus RocCurveDisplay) untuk menilai kinerja model.

# # Data Loading

# In[2]:


df = pd.read_csv('diabetes.csv')
df


# Kode ini membaca dataset diabetes.csv menggunakan pandas dan menyimpannya dalam variabel df. Perintah df di akhir digunakan untuk menampilkan isi dataset, sehingga pengguna dapat melihat struktur awal data seperti jumlah kolom, baris, dan contoh nilai di setiap fitur.

# # Data Understanding & EDA

# In[3]:


df.isnull().sum()


# Untuk mengecek nilai null yang ada di data

# In[4]:


df.info()


# Untuk melihat informasi utama dari setiap column

# In[5]:


df.describe()


# Kode ini menampilkan ringkasan statistik deskriptif dari setiap kolom numerik dalam dataset, seperti nilai minimum, maksimum, rata-rata (mean), standar deviasi (std), serta kuartil (25%, 50%, 75%). Ini membantu memahami distribusi, skala, dan potensi adanya outlier pada data.

# In[6]:


df['Outcome'].value_counts()


# Untuk melihat jumlah data dari setiap Outcome yang ada

# In[7]:


for c in df.columns[:-1]:
    plt.figure()
    sns.histplot(data=df, x=c, hue='Outcome', kde=True, stat='density')
    plt.title(f'Distribusi {c} menurut Outcome')
    plt.show()


# Melakukan visualisasi distribusi setiap fitur numerik pada dataset terhadap label Outcome. Hal ini membantu memahami bagaimana pola distribusi data dari masing-masing fitur ketika dibedakan berdasarkan apakah pasien menderita diabetes (Outcome=1) atau tidak (Outcome=0).

# In[8]:


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, square=True)
plt.title('Matriks Korelasi antar Fitur')
plt.show()


# Bertujuan untuk melihat korelasi setiap fitur dengan fitur yang lainnya

# In[9]:


group_stats = df.groupby('Outcome').agg(['median', 'mean', 'std'])
group_stats


# Menghitung nilai median, mean, dan standar deviasi (std) dari setiap fitur numerik dalam dataset, dikelompokkan berdasarkan nilai Outcome (0 = tidak diabetes, 1 = diabetes).

# # Data Preprocessing

# ### Winsorize

# In[10]:


plt.figure(figsize=(15,10))
df.boxplot()
plt.xlabel('Value')
plt.ylabel('Variable')
plt.title('Distribution of Variables')
plt.show()


# Kode diatas bertujuan untuk membuatkan Boxplot untuk menampilkan nilai dari setiap colum yang merupakan Outlier

# In[11]:


for col in df.select_dtypes(include=['number']).columns:
    df[col] = winsorize(df[col], limits=[0.05, 0.05])


# Tujuan untuk mengurangi pengaruh outlier ekstrem pada setiap fitur numerik dengan menerapkan winsorizing, yaitu membatasi nilai-nilai ekstrim ke persentil tertentu.
# 
# winsorize(..., limits=[0.05, 0.05]): Memotong nilai-nilai pada setiap kolom ke dalam batas 5% terbawah dan 5% teratas. Nilai-nilai di luar batas ini akan diganti dengan nilai pada batas persentil tersebut.

# ### Normalization

# In[12]:


scaler = MinMaxScaler()

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_scaled


# Melakukan normalisasi fitur numerik ke rentang 0–1 menggunakan MinMaxScaler agar skala data seragam dan tidak bias terhadap fitur dengan nilai besar.

# ### SMOTE

# In[13]:


sm = SMOTE(random_state=42)

X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

print("Sebelum SMOTE:", y.value_counts())
print("Sesudah SMOTE:", pd.Series(y_resampled).value_counts())


# Kode ini menerapkan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan jumlah data pada masing-masing kelas target. SMOTE membuat sampel sintetis pada kelas minoritas agar distribusi kelas menjadi seimbang, sehingga model tidak bias terhadap kelas mayoritas.

# In[14]:


X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled = pd.Series(y_resampled, name='Outcome')
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

df_resampled


# Kode ini menggabungkan kembali data fitur (X_resampled) dan label (y_resampled) hasil SMOTE ke dalam satu DataFrame (df_resampled) untuk memudahkan analisis dan pemrosesan selanjutnya.

# ### PCA

# In[15]:


pca = PCA(n_components=2, random_state=42)

X_pca = pca.fit_transform(X_resampled)

X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
X_pca_df['Outcome'] = y_resampled.values
X_pca_df.head()


# Kode ini menerapkan PCA (Principal Component Analysis) untuk mereduksi dimensi fitur menjadi dua komponen utama (PC1 dan PC2), dengan tujuan menyederhanakan data dan memudahkan visualisasi tanpa kehilangan terlalu banyak informasi penting.

# # Model Training with PCA

# ## Train - Test Split

# In[16]:


X_pca = X_pca_df.drop(['Outcome'], axis=1)
y_pca = X_pca_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_pca, test_size = 0.2, random_state = 42)


# Kode ini memisahkan data fitur (X) dan target (y) dari DataFrame hasil SMOTE, lalu membagi data menjadi data latih dan data uji dengan rasio 80:20 menggunakan train_test_split, untuk keperluan pelatihan dan evaluasi model.

# In[17]:


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=1, gamma='scale')
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}")


# Kode ini membandingkan performa empat model machine learning (Logistic Regression, Random Forest, KNN, dan SVM) menggunakan 5-fold cross-validation dan metrik akurasi untuk mengevaluasi rata-rata kinerja tiap model pada data latih. Model ini dilatih dengan data PCA

# # Model Training without PCA

# ## Train - Test Split

# In[18]:


X = df_resampled.drop(['Outcome'], axis=1)
y = df_resampled['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Kode ini memisahkan data fitur (X) dan target (y) dari DataFrame hasil SMOTE, lalu membagi data menjadi data latih dan data uji dengan rasio 80:20 menggunakan train_test_split, untuk keperluan pelatihan dan evaluasi model.

# In[19]:


print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')


# Kode ini digunakan untuk menampilkan jumlah total sampel pada dataset keseluruhan, serta membandingkannya dengan jumlah sampel pada dataset pelatihan dan pengujian setelah dilakukan pembagian data (train-test split).

# In[20]:


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=1, gamma='scale')
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}")


# Kode ini membandingkan performa empat model machine learning (Logistic Regression, Random Forest, KNN, dan SVM) menggunakan 5-fold cross-validation dan metrik akurasi untuk mengevaluasi rata-rata kinerja tiap model pada data latih. Model ini dilatih tanpa PCA

# ### Fine Tuning Random Forest

# In[21]:


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid,
                           cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)


# Kode ini melakukan hyperparameter tuning Random Forest menggunakan GridSearchCV. Pertama, param_grid mendefinisikan kombinasi nilai yang akan diuji untuk sejumlah parameter penting—jumlah pohon (n_estimators 100 atau 200), kedalaman maksimum pohon (max_depth None, 10, 20), ukuran minimal pemisahan dan daun (min_samples_split, min_samples_leaf), serta opsi bootstrap. GridSearchCV kemudian melatih model pada setiap kombinasi tersebut dengan 5‑fold cross‑validation, mem‐benchmark kinerja berdasarkan akurasi guna mencegah overfitting dan memperoleh estimasi performa yang stabil. Setelah dipanggil fit, objek grid_search memilih konfigurasi terbaik; perintah best_params_ menampilkan set parameter teroptimal, sedangkan best_estimator_ memberikan model Random Forest siap pakai yang diharapkan memiliki generalisasi paling baik pada data baru.

# ### Updating Model Random Forest

# In[22]:


best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print(classification_report(y_test, y_pred))


# Kode ini menggunakan model Random Forest terbaik hasil GridSearchCV untuk memprediksi data uji (X_test). Hasil prediksi (y_pred) kemudian dievaluasi menggunakan classification_report, yang menampilkan metrik akurasi, precision, recall, dan F1-score untuk masing-masing kelas. Ini membantu mengukur seberapa baik model dalam mengklasifikasikan data dengan benar, khususnya dalam konteks ketidakseimbangan kelas.

# # Model Evaluation

# In[23]:


y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))


# Kode ini menggunakan model Random Forest terbaik hasil GridSearchCV untuk memprediksi data uji (X_test). Hasil prediksi (y_pred) kemudian dievaluasi menggunakan classification_report, yang menampilkan metrik akurasi, precision, recall, dan F1-score untuk masing-masing kelas. Ini membantu mengukur seberapa baik model dalam mengklasifikasikan data dengan benar, khususnya dalam konteks ketidakseimbangan kelas.

# In[24]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Diabetes', 'Diabetes'],
            yticklabels=['Non-Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Kode ini menghitung confusion matrix dari prediksi model (y_pred) terhadap label sebenarnya (y_test), lalu menampilkannya sebagai heatmap berwarna biru dengan anotasi angka. Sumbu x mewakili kelas yang diprediksi (Non‑Diabetes, Diabetes) dan sumbu y mewakili kelas aktual, sehingga memudahkan identifikasi jumlah true positive, true negative, false positive, dan false negative—informasi kunci untuk menilai kesalahan dan keberhasilan model secara visual.

# In[25]:


RocCurveDisplay.from_estimator(best_rf, X_test, y_test)
plt.title('ROC Curve')
plt.show()

auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {auc:.4f}")


# Kode ini menampilkan kurva ROC (Receiver Operating Characteristic) untuk model Random Forest terbaik (best_rf) guna mengevaluasi performa klasifikasi dalam membedakan kelas. Kurva ROC memvisualisasikan trade-off antara True Positive Rate dan False Positive Rate. Selain itu, dihitung nilai ROC AUC (Area Under the Curve) menggunakan probabilitas prediksi (y_proba), yang menunjukkan kemampuan model secara keseluruhan—semakin mendekati 1, semakin baik performa model.