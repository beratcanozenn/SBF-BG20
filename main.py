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
##############
# Veri İnceleme - Kontrol Aşaması
##############

def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### İnfo #####################")
    print(dataframe.info())
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df, quan=True)

df.isnull().sum()

missing_percentage = df.isnull().sum() * 100 / len(df)

def grab_col_names(dataframe, cat_th=5, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

###### Boş değerleri doldurma, Veri düzenleme

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Age'].fillna(7, inplace=True)
# df['Age'] = df['Age'].str.extract('(\d+)')

df.dropna(subset=['Years'], inplace=True)
df.dropna(subset=['Genre'], inplace=True)

median_ratings = df.groupby(['Genre'])['Rating'].median()
df['Rating'] = df.apply(lambda row: median_ratings.loc[(row['Genre'])] if pd.isna(row['Rating']) else row['Rating'], axis=1)

df['Field6'] = df['Field6'].str.replace('K', '').astype(float) * 1000
df['Field6'] = df.groupby('Genre')['Field6'].transform(lambda x: x.fillna(x.median()))
df.rename(columns={'Field6': 'VoteCount'}, inplace=True)

df['Rewiev_Count'] = df.groupby('Genre')['Rewiev_Count'].transform(lambda x: x.fillna(x.median()))
df['Rewiev_Count'].fillna(0, inplace=True)
df.rename(columns={'Rewiev_Count': 'Review_Count'}, inplace=True)

df['EpisodeCount'] = pd.to_numeric(df['EpisodeCount'], errors='coerce')
median_ratings = df.groupby(['Seasons'])['EpisodeCount'].median()
df['EpisodeCount'] = df.apply(lambda row: median_ratings.loc[(row['Seasons'])] if pd.isna(row['EpisodeCount']) else row['EpisodeCount'], axis=1)

df["EpsPerSeason"] = df['EpisodeCount'] / df['Seasons']
df["EpsPerSeason"] = df["EpsPerSeason"].round()

#### Kullanılmayan kolonların silinmesi
will_be_deleted= ['Story_Line', 'Genre_Long', 'Location', 'Production','Field1_links','Field3_links','Field4_links','Field9_links','Sound','Popularity','Stars']
df.drop(columns=will_be_deleted, inplace=True)
df.drop(columns='Field9_links', inplace=True)
############ Tür ayırma

genre_counts = df['Genre'].value_counts()

# Çubuk grafik oluştur
plt.figure(figsize=(20,8))
plt.xticks(rotation=45, ha="right")
plt.bar(genre_counts.index, genre_counts.values)
plt.xlabel('Genre')
plt.xticks(fontsize=8)
plt.ylabel('Frekans')
plt.title('Dizi Türleri Sayıları')
plt.show()
######

genre_encoded = pd.get_dummies(df['Genre'])

# Toplamda 27'ten az frekansta bulunan sınıfları "others" adında bir sınıfta topladık
threshold = 27
genre_counts = df['Genre'].value_counts()
rare_genres = genre_counts[genre_counts < threshold].index
genre_encoded['others'] = df['Genre'].apply(lambda x: 1 if x in rare_genres else 0)


result_df = pd.concat([df, genre_encoded], axis=1)
result_df = result_df.drop('Genre', axis=1)
print(result_df)

df = pd.concat([df, genre_encoded], axis=1)
df = df.drop('Genre', axis=1)
print(df)

################### başlangıç bitiş

df['Years']
df['baslangic'] = df['Years'].astype(str).apply(lambda x: x[:4])
df['baslangic'] = pd.to_datetime(df['baslangic'])
df['baslangic'] = df['baslangic'].astype(int)
df['isFinal'] = df['Years'].apply(lambda x: 1 if len(str(x)) > 6 else 0)

################## sezon bilgisi

yeni_liste = []
for eleman in df['Seasons']:
    try:

        eleman_parcalari = eleman.split('\n')


        if len(eleman_parcalari[0]) == 2:
            yeni_liste.append(eleman_parcalari[:1])
        else:

            yeni_liste.append([eleman_parcalari[0]])
    except (IndexError, AttributeError):

        yeni_liste.append([])

df['Seasons'] = yeni_liste

for i in range(len(df["Seasons"])):
    df["Seasons"][i] = df["Seasons"][i][0].replace('[', '').replace(']', '') if len(df["Seasons"][i]) > 0 else ''


df.dropna(subset=['Genre', 'baslangic'], inplace=True)
df['Seasons'] = df['Seasons'].replace('', np.nan)
df['Seasons'] = df.groupby(['Genre'])['Seasons'].transform(lambda x: x.fillna(x.median().round()))

df['Seasons'] = pd.to_numeric(df['Seasons'])

def label_class(season):
    if season == 1:
        return 1
    elif season == 2:
        return 2
    elif 3 <= season <= 5:
        return 3
    elif 5 < season <= 10:
        return 4
    else:
        return 5


df['Sınıf'] = df['Seasons'].apply(label_class)

df.dropna(subset=['Seasons'], inplace=True)

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

#########################################
#Karar Ağacı Modeli
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(train_data, test_data):
    features = ['Age', 'Duration', 'VoteCount', 'Review_Count', 'baslangic', 'EpsPerSeason', 'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Game-Show', 'History', 'Music', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'others', 'HasLove', 'HasFamily', 'HasIstanbul', 'Basrol_Class1', 'Basrol_Class2', 'Basrol_Class3', 'Basrol_Class4']

    target = 'Sınıf'

    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)


    importances = model.feature_importances_

    return accuracy, importances


cumulative_accuracy = 0
cumulative_importances = []
cumulative_report = ""

year_cumulative_accuracy = {}
individual_accuracies = {}

for year in range(1974, 2024):
    train_data = df[df['baslangic'] < year]
    test_data = df[df['baslangic'] == year]

    if not test_data.empty and not train_data.empty:
        accuracy, importances = evaluate_model(train_data, test_data)

        cumulative_accuracy += accuracy
        cumulative_importances.append(importances)
        cumulative_report += f"Sınamanın Yılı: {year}\nDoğruluk: {accuracy}\n{'='*40}\n"

        year_cumulative_accuracy[year] = cumulative_accuracy
        individual_accuracies[year] = accuracy
    else:
        individual_accuracies[year] = None

average_accuracy = cumulative_accuracy / len(range(1974, 2024))

print(f"Kümülatif Ortalama Doğruluk: {average_accuracy}")

print(cumulative_report)

cumulative_importances = pd.DataFrame(cumulative_importances, columns=features)
cumulative_importances = cumulative_importances.mean()
cumulative_importances = cumulative_importances / cumulative_importances.sum()

plt.figure(figsize=(8, 5))
plt.bar(features, cumulative_importances)
plt.xlabel('Özellikler')
plt.ylabel('Feature Importance')
plt.title('Kümülatif Feature Importanceları')
plt.grid(True)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.plot(years, accuracies, marker='o', label='Kümülatif Doğruluk')
plt.xlabel('Yıl')
plt.ylabel('Kümülatif Doğruluk')
plt.title('Yıla Göre Kümülatif Doğruluk')
plt.legend()
plt.grid(True)
plt.show()

individual_years = list(individual_accuracies.keys())
individual_values = list(individual_accuracies.values())

plt.plot(individual_years, individual_values, marker='o', label='Yıllara Göre Doğruluk')
plt.xlabel('Yıl')
plt.ylabel('Doğruluk')
plt.title('Her Yılın Doğruluk Değerleri')
plt.legend()
plt.grid(True)
plt.show()

#########################





#Outlier Detection Adımı
###################

def outlier_thresholds(dataframe, variable, low_quantile=0, up_quantile=0.85):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col in df.columns:
        print(col, check_outlier(df, col))


df["EpsPerSeason"].describe().T

# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df,col)

df.head()

############# Korelasyon Matrisini Hesaplama ###############
correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Korelasyon Matrisi")
plt.show()

