import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import skew
pd.set_option('display.max_columns', None)

# Adjusting display settings for pandas
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)

df = pd.read_csv("/Users/Furkan/Desktop/TurkSeries.csv")
data1 = pd.read_csv("/Users/Furkan/Desktop/data1.csv")
data2 = pd.read_csv("/Users/Furkan/Desktop/bolumler.csv")

merged_df = pd.merge(df, data1, left_on="Name", right_on="Field2", how="inner")
merged2_df = pd.merge(df, data2, left_on="Name", right_on="Field2", how="inner")

df["Star1"] = merged_df["Field3_text"]
df["Star2"] = merged_df["Field4_text"]
df["Star3"] = merged_df["Field5_text"]

df["EpisodeCount"] = merged2_df["Field3"]

df.columns.tolist()


################### TF - IDF Matrisi Oluşturma ve Değişken Türetimi
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df['Summary'] = df['Summary'].fillna('Unknown')

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Summary'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# print("TF-IDF Matrix:")
# print(tfidf_df)
top_40_words = tfidf_df.sum().sort_values(ascending=False).head(40)

df['HasLove'] = df['Summary'].apply(lambda x: 1 if 'love' in x.lower() else 0)
df['HasIstanbul'] = df['Summary'].apply(lambda x: 1 if 'istanbul' in x.lower() else 0)

family_keywords = ['father', 'mother', 'daughter', 'son', 'wife', 'husband', 'family','children']
df['HasFamily'] = df['Summary'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in family_keywords) else 0)

##############################33
#Oyuncu Sınıflandırma
################################
#Veri Düzenleme
df_oyuncular = pd.read_csv("/Users/Furkan/Desktop/oyuncular.csv")

df_oyuncular["Yas"][0][6]

df_oyuncular["Yas"] = df_oyuncular["Yas"].apply(lambda x: x[6:8] if isinstance(x, str) and len(x) >= 7 else x)

df_oyuncular.dropna(inplace=True)

print(df_oyuncular['Yas'].unique())
print(df_oyuncular['Proje_Sayısı'].unique())

df_oyuncular['Yas'] = pd.to_numeric(df_oyuncular['Yas'], errors='coerce')
df_oyuncular['Proje_Sayısı'] = pd.to_numeric(df_oyuncular['Proje_Sayısı'], errors='coerce')

#Veri Doldurma
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # veya 'median', 'most_frequent' gibi bir strateji seçebilirsiniz
df_oyuncular[['Yas', 'Proje_Sayısı']] = imputer.fit_transform(df_oyuncular[['Yas', 'Proje_Sayısı']])

#K-Means Optimizasyon
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np

k_values = range(1, 11)
inertia_values = []

for k in k_values:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(df_oyuncular[['Yas', 'Proje_Sayısı']])
    inertia_values.append(kmeans_model.inertia_)

# Dirsek yöntemi grafiği
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

#K-Means Sınıflandırma

num_clusters = 4  # Optimal sınıf sayısını 4 olarak seçtik
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
df_oyuncular['Cluster'] = kmeans_model.fit_predict(df_oyuncular[['Yas', 'Proje_Sayısı']])

# Her sınıf için oyuncu isimlerini içeren bir sözlük oluşturduk
siniflar = {}
for sinif in range(num_clusters):
    siniflar[sinif] = df_oyuncular[df_oyuncular['Cluster'] == sinif]['Name'].tolist()


for sinif, oyuncular in siniflar.items():
    print(f"Sınıf {sinif + 1} Oyuncuları:")
    print(oyuncular)
    print("=" * 30)

# dictionaryden ayrı listelere çevirme işlemi gerçekleştirdik
ayrilmis_listeler = list(siniflar.values())

sinif_0_listesi = ayrilmis_listeler[0]
sinif_1_listesi = ayrilmis_listeler[1]
sinif_2_listesi = ayrilmis_listeler[2]
sinif_3_listesi = ayrilmis_listeler[3]
df.head()
###### Oyuncu Sınıflarından Yararlanarak df üzerinde yeni oyuncu sınıfı değişkenleri oluşturduk
df['Basrol_Class1'] = df.apply(lambda row: 1 if any(star in sinif_0_listesi for star in row) else 0, axis=1)
df['Basrol_Class2'] = df.apply(lambda row: 1 if any(star in sinif_1_listesi for star in row) else 0, axis=1)
df['Basrol_Class3'] = df.apply(lambda row: 1 if any(star in sinif_2_listesi for star in row) else 0, axis=1)
df['Basrol_Class4'] = df.apply(lambda row: 1 if any(star in sinif_3_listesi for star in row) else 0, axis=1)
