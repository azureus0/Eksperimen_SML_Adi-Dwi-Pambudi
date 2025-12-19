#!/usr/bin/env python
# coding: utf-8

# # **1. Perkenalan Dataset**
# 

# Tahap pertama, Anda harus mencari dan menggunakan dataset dengan ketentuan sebagai berikut:
# 
# 1. **Sumber Dataset**:  
#    Dataset dapat diperoleh dari berbagai sumber, seperti public repositories (*Kaggle*, *UCI ML Repository*, *Open Data*) atau data primer yang Anda kumpulkan sendiri.
# 

# Sumber Dataset: https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset

# # **2. Import Library**

# Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning atau deep learning.

# In[116]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# # **3. Memuat Dataset**

# Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.
# 
# Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.
# 
# Jika dataset berupa unstructured data, silakan sesuaikan dengan format seperti kelas Machine Learning Pengembangan atau Machine Learning Terapan

# In[117]:


file_path = "../Exam_Score_raw.csv"
df = pd.read_csv(file_path)

df.head()


# # **4. Exploratory Data Analysis (EDA)**
# 
# Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset.
# 
# Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.

# In[118]:


df.info()


# In[119]:


df.shape


# In[120]:


df.isnull().sum()


# In[121]:


df.duplicated().sum()


# In[122]:


df.describe()


# ### Distribusi Fitur Numerik

# In[123]:


df.hist(figsize = (12,10))
plt.show()


# ### Distribusi Fitur Kategorikal

# In[124]:


# kolom kategorikal
eda_categorical_cols = df.select_dtypes(include=['object'])

for feature in eda_categorical_cols:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Bar Chart (Kiri)
    sns.countplot(x=df[feature], ax=ax[0], color='skyblue', order=df[feature].value_counts().index)
    ax[0].set_title(f'Frekuensi {feature}')
    ax[0].tick_params(axis='x', rotation=45) 
    ax[0].set_xlabel('')

    # 2. Pie Chart (Kanan)
    df[feature].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, 
                                    colors=sns.color_palette('pastel'), ax=ax[1])
    ax[1].set_title(f'Persentase {feature}')
    ax[1].set_ylabel('')

    plt.tight_layout()
    plt.show()


# In[125]:


eda_numerical_cols = df.select_dtypes(include=np.number).columns.drop('student_id')

plt.figure(figsize=(12,10))
sns.heatmap(df[eda_numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.show()


# In[126]:


# Box Plot Outlier
n = len(eda_numerical_cols)

plt.figure(figsize=(15, 5))

for i, col in enumerate(eda_numerical_cols, 1):
    plt.subplot(1, n, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(f'{col}')
    plt.ylabel('') 

plt.tight_layout()
plt.show()


# # **5. Data Preprocessing**

# Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning.
# 
# Jika Anda menggunakan data teks, data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.
# 
# Berikut adalah tahapan-tahapan yang bisa dilakukan, tetapi **tidak terbatas** pada:
# 1. Menghapus atau Menangani Data Kosong (Missing Values)
# 2. Menghapus Data Duplikat
# 3. Normalisasi atau Standarisasi Fitur
# 4. Deteksi dan Penanganan Outlier
# 5. Encoding Data Kategorikal
# 6. Binning (Pengelompokan Data)
# 
# Cukup sesuaikan dengan karakteristik data yang kamu gunakan yah. Khususnya ketika kami menggunakan data tidak terstruktur.

# ### Drop Duplicates

# In[127]:


df.drop_duplicates(inplace=True)
df.duplicated().sum()


# ### Drop Missing Values

# In[128]:


df = df.dropna()
df.isnull().sum()


# ### Drop Columns with 'id'

# In[129]:


columns_to_drop = ['student_id']
df = df.drop(columns=columns_to_drop)
df.info()


# ### Drop Outlier

# In[130]:


# Features to check for outliers
features_to_check = ['study_hours', 'sleep_hours']

# Delete outliers based on IQR
for feature in features_to_check:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]


# ### Binning

# In[131]:


df['sleep_hours_binned'] = pd.cut(df['sleep_hours'], bins=3, labels=['Kurang', 'Cukup', 'Baik'])

df = df.drop(columns='sleep_hours')

df[['sleep_hours_binned']].head()


# ### Splitting

# In[132]:


X = df.drop(columns=['exam_score'])
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)

print(f"Data Train: {X_train.shape[0]} baris")
print(f"Data Test: {X_test.shape[0]} baris")


# ### Encoding (One-Hot Encoding)

# In[133]:


categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns


# In[134]:


encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
).set_output(transform="pandas")

# fit encoder di train
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat  = encoder.transform(X_test[categorical_cols])

# drop kolom kategorikal lama
X_train = X_train.drop(columns=categorical_cols)
X_test  = X_test.drop(columns=categorical_cols)

# gabungkan dengan hasil encoding
X_train = pd.concat([X_train, X_train_cat], axis=1)
X_test  = pd.concat([X_test,  X_test_cat], axis=1)


# ### Scaling (Standardization)

# In[135]:


numerical_cols = ['age', 'study_hours', 'class_attendance']


# In[136]:


scaler = StandardScaler()

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

X_train.head()


# # **Save Dataset**

# In[137]:


os.makedirs("Exam_Score_preprocessing", exist_ok=True)

train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
test_df  = pd.concat([X_test,  y_test.reset_index(drop=True)], axis=1)

train_df.to_csv("Exam_Score_preprocessing/train.csv", index=False)
test_df.to_csv("Exam_Score_preprocessing/test.csv", index=False)

