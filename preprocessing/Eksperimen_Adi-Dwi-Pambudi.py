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

# In[652]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# # **3. Memuat Dataset**

# Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.
# 
# Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.
# 
# Jika dataset berupa unstructured data, silakan sesuaikan dengan format seperti kelas Machine Learning Pengembangan atau Machine Learning Terapan

# In[653]:


file_path = "../Exam_Score_Prediction_raw.csv"
df = pd.read_csv(file_path)

df.head()


# # **4. Exploratory Data Analysis (EDA)**
# 
# Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset.
# 
# Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.

# In[654]:


df.info()


# In[655]:


df.shape


# In[656]:


df.isnull().sum()


# In[657]:


df.duplicated().sum()


# In[658]:


df.describe()


# ### Distribusi Fitur Numerik

# In[659]:


df.hist(figsize = (12,10))
plt.show()


# ### Distribusi Fitur Kategorikal

# In[660]:


# kolom kategorikal
categorical_cols = df.select_dtypes(include=['object'])

for feature in categorical_cols:
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


# In[661]:


numerical_cols = df.select_dtypes(include=np.number).columns.drop('student_id')

plt.figure(figsize=(12,10))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.show()


# In[662]:


# Box Plot Outlier
n = len(numerical_cols)

plt.figure(figsize=(15, 5))

for i, col in enumerate(numerical_cols, 1):
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

# In[663]:


df.drop_duplicates(inplace=True)
df.duplicated().sum()


# ### Drop Columns with 'id'

# In[664]:


columns_to_drop = ['student_id']
df = df.drop(columns=columns_to_drop)
df.info()


# ### Drop Missing Values

# In[665]:


df = df.dropna()
df.isnull().sum()


# ### Drop Outlier

# In[666]:


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

# In[667]:


df['sleep_hours_binned'] = pd.cut(df['sleep_hours'], bins=3, labels=['Kurang', 'Cukup', 'Baik'])

df = df.drop(columns='sleep_hours')

df[['sleep_hours_binned']].head()


# ### Encoding Columns

# In[668]:


categorical_cols = df.select_dtypes(include=['object', 'category']).columns

encoder = OneHotEncoder(sparse_output=False, dtype=int).set_output(transform="pandas")
df_encoded = encoder.fit_transform(df[categorical_cols])
df_final = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)

df_final.head()


# ### Splitting

# In[669]:


X = df_final.drop(columns=['exam_score'])
y = df_final['exam_score']

# splitting (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data Train: {X_train.shape[0]} baris")
print(f"Data Test: {X_test.shape[0]} baris")


# In[670]:


X_train.info()


# ### Scaling (Standarisasi)

# In[ ]:


# numerical_cols = X_train.select_dtypes(include=['number']).columns
numerical_cols = ['age', 'study_hours', 'class_attendance']

scaler = StandardScaler()

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

X_train.head()


# # **Save Dataset**

# In[672]:


df_final.to_csv('Exam_Score_Prediction_preprocessing.csv', index=False)

