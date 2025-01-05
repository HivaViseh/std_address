import numpy as np
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
#import plot_null_columns
import merge_datasets
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine
import pyodbc
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import os




driver = "driver"

#cnxn = pyodbc.connect(driver=driver, host="host", database="database",
#                      trusted_connection="yes", user="user", password="")

engine = create_engine(f'mssql+pyodbc:?trusted_connection=yes&driver={driver}')


#engine = create_engine('mssql+pyodbc:?trusted_connection=yes')
sql = """

SELECT [STREET_NUMBER]
      ,[STREET_NAME]
      ,[CITY]
      ,[POSTAL_CODE]
      ,[ACTUAL_USE_DESCRIPTION]
      ,[NEIGHBOURHOOD]
      ,[REGIONAL_DISTRICT]
      ,[GEN_PROPERTY_CLASS_DESC]
      ,[Year_Built]
      ,[Number_of_Storeys]
      ,[Predominant_Manual_Class]
      ,[Type_of_Heating]
      ,[Type_of_Construction]
  FROM [TSBC_BCA].[bcassessment].[restricted_tech_vw]
  WHERE [GEN_PROPERTY_CLASS_DESC] = 'Residential'
  """




cities = pd.read_csv(os.path.join(r"C:\Users\hviseh\projects\BCDATA_Eval\bcdata_eval", "matched_cities.csv"))
unique_city_list = cities["CITY"].unique().tolist()

# Initialize an empty list to collect processed chunks
chunks = []

# Read data in chunks
chunksize = 1000  # You can adjust this number based on your memory capacity
for chunk in pd.read_sql(sql, con=engine, chunksize=chunksize):
        # Filter chunk by unique cities
    chunk = chunk[chunk["CITY"].isin(unique_city_list)]
    
    # Drop the unnecessary column and rows with NA values
    chunk = chunk.drop(columns="GEN_PROPERTY_CLASS_DESC")
    chunk = chunk.dropna()
    print("A")
    
    # Append the processed chunk to the list
    chunks.append(chunk)

# Concatenate all chunks into a single DataFrame
df = pd.concat(chunks, ignore_index=True)
print("Hello")
#df["GEN_PROPERTY_CLASS_DESC"] == ['Residential' 'Business And Other' 'Rec/Non Profit' 'Utilities'
# 'Light Industry' None 'Major Industry' 'Farm' 'Managed Forest Land']
#df = df[(df["GEN_PROPERTY_CLASS_DESC"] == 'Business And Other') | (df["GEN_PROPERTY_CLASS_DESC"] == 'Light Industry') | (df["GEN_PROPERTY_CLASS_DESC"] == 'Major Industry')]

'''
df = df[(df["GEN_PROPERTY_CLASS_DESC"] != 'Residential')
      & (df["GEN_PROPERTY_CLASS_DESC"] != 'Managed Forest Land') 
      & (df["GEN_PROPERTY_CLASS_DESC"] != 'Farm') 
      & (df["GEN_PROPERTY_CLASS_DESC"].notna())]
'''
#df = df[df["GEN_PROPERTY_CLASS_DESC"] == 'Residential']
#df = df.drop(columns="GEN_PROPERTY_CLASS_DESC")

#df = df.dropna()

selected_columns = ["STREET_NUMBER", "STREET_NAME", "CITY", "POSTAL_CODE", 
                  "ACTUAL_USE_DESCRIPTION", "NEIGHBOURHOOD", "REGIONAL_DISTRICT", 
                  "Year_Built", "Number_of_Storeys", "Predominant_Manual_Class", 
                  "Type_of_Construction"]



#plot_null_columns.plot_null_columns(df)

# Count the number of occurrences for each set of values in specified columns
df['Repetition_Count'] = df.groupby(selected_columns)[selected_columns[0]].transform('count')

# Filter out rows with repetitions and keep the first occurrence
df_unique = df.drop_duplicates(subset=selected_columns, keep='first')
#print(df_unique[df_unique['Repetition_Count'] > 1])

df1 = pd.read_csv(r"C:\Users\hviseh\projects\BCDATA_Eval\bcdata_eval\Incident Count per Location.csv")

df_unique["city_name"] = merge_datasets.fuzzy_match(df1, df_unique, "city_name", "CITY", threshold=90)
df_unique["street_name"] = merge_datasets.fuzzy_match(df1, df_unique, "street_name", "STREET_NAME", threshold=90)
df_unique["postal_code"] = merge_datasets.fuzzy_match(df1, df_unique, "postal_code", "POSTAL_CODE", threshold=95)
df_unique["street_number"] = merge_datasets.fuzzy_match(df1, df_unique, "street_number", "STREET_NUMBER", threshold=95)

merge_columns = ["city_name", "postal_code", "street_name", "street_number"]

# Merge df2 and df1 on the selected columns
merged_df = pd.merge(df_unique, df1, how='left', on=merge_columns)


selected_columns2 = ["STREET_NUMBER", "STREET_NAME", "CITY", "POSTAL_CODE", 
                  "ACTUAL_USE_DESCRIPTION", 
                  "city_name", "postal_code", "street_name", "street_number", "Repetition_Count" ]

merged_df.drop(columns=selected_columns2, inplace=True)

merged_df.to_csv(r"C:\Users\hviseh\projects\BCDATA_Eval\bcdata_eval\final_data_before_training_Residential_Buildings.csv", index=False)



#subset_merged_df.to_csv("/Projects")

