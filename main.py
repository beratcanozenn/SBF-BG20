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