import numpy as np
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine
import pandas as pd

import matplotlib.pyplot as plt
import plot_null_columns
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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import os

from sklearn.feature_selection import RFE


#Set-Alias -Name poetry -Value C:\Users\hviseh\AppData\Roaming\Python\Scripts\poetry.exe
#poetry export --without-hashes --without dev -f requirements.txt -o requirements.txt


file_name = "final_data_before_training.csv"
df = pd.read_csv(os.path.join(r"C:\Users\hviseh\projects\BCDATA_Eval\bcdata_eval", file_name))


df['Incident Count'] = df['Incident Count'].replace(np.nan, 0)
df = df[df['Number_of_Storeys'] != 118.0]
df['Incident Count'] = df['Incident Count'].apply(lambda x: 1 if x > 1 else x)
df["Building_Age"] = df["Year_Built"].apply(lambda x: 2023-x )
df.drop(columns='Year_Built', inplace=True)


#bins = [1799, 1900, 1950, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2023]  
#labels = ['Before_1900', '1900-1950','1950-1980','1980-1985','1985-1990','1990-1995', '1995-2000', '2000-2005', '2005-2010', '2010-2015', '2015-2023']
#df['Binned_Year_Built'] = pd.cut(df['Year_Built'], bins=bins, labels=labels, right=False)

cols = ['NEIGHBOURHOOD', 'REGIONAL_DISTRICT','Predominant_Manual_Class', 'Type_of_Heating', 'Type_of_Construction']

df_encoded = pd.get_dummies(df, columns=cols)
#df_encoded.drop(columns='Year_Built', inplace=True)

column_to_move = df_encoded.pop('Incident Count')  
df_encoded['Incident Count'] = column_to_move

num_columns = df_encoded.shape[1]
print("Number of columns:", num_columns, df_encoded.columns)

train, valid, test = np.split(df_encoded.sample(frac=1, random_state=42), [int(0.6*len(df_encoded)), int(0.8*len(df_encoded))])
#ranodom_state = between 30 and 42
print("Risk:", len(train[train["Incident Count"] == 1]))
print("No Risk:", len(train[train["Incident Count"] == 0]))

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    #oe = OrdinalEncoder()
    #oe.fit(X)
    #X = oe.transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #Take more from the less class and keep sampling from there to increase the size of our dataset of that smaller 
    #class so that they now match
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y




train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

print("Risk:", sum(y_train==1))
print("No Risk:", sum(y_train==0))









def plot_history(history, title, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend(loc='lower left')

    fig.suptitle(title)
    
    # Add validation loss and accuracy annotations
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    ax1.annotate(f'Val Loss: {val_loss:.4f}', xy=(1, 0), xycoords='axes fraction', fontsize=10,
                 xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom')
    ax2.annotate(f'Val Accuracy: {val_accuracy:.4f}', xy=(1, 0), xycoords='axes fraction', fontsize=10,
                 xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
 



def train_model(X_train, y_train, X_valid, y_valid, num_nodes, dropout_prob, lr, batch_size, epochs):
    input_shape = X_train.shape[1]
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), verbose=0
    )

    return nn_model, history



least_val_loss = float('inf')
least_loss_model = None
epochs=20

save_dir = 'C:/Users/hviseh/Desktop/Results2/'
plot_counter = 1

for num_nodes in [64, 128]:
    for dropout_prob in [0.2]:
        for lr in [0.00001]:
            for batch_size in [128, 256]:
                title = f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}"
                print(title)
                
                model, history = train_model(X_train, y_train, X_valid, y_valid, num_nodes, dropout_prob, lr, batch_size, epochs)
                plot_history(history, title, save_path=os.path.join(save_dir, f"plot_{plot_counter}.png"))
                plot_counter += 1
                
                val_loss, val_accuracy = model.evaluate(X_valid, y_valid)
                print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model

'''
for label in cols[:-1]:
    plt.figure(figsize=(20, 16))
    plt.subplots_adjust(bottom=0.5)
    
    prev_incidents_counts = df[df['Incident Count'] != 0][label].value_counts(normalize=True)

    prev_incidents_counts.plot(kind='bar', color="Red", alpha=0.65, label="Buildings with Previous Incidents", width=1)
    
    no_incidents_counts = df[df['Incident Count'] == 0][label].value_counts(normalize=True)

    no_incidents_counts.plot(kind='bar', color="blue", alpha=0.65, label="Buildings with No Previous Incidents", width=1)
    
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.xticks(fontsize=2, rotation=90)
    plt.legend(loc='upper right', fontsize=12)
    plt.show()

'''


#df_encoded = pd.get_dummies(df, columns=['NEIGHBOURHOOD', 'REGIONAL_DISTRICT', 'Predominant_Manual_Class', 'Type_of_Heating', 'Type_of_Construction'])

#scaler = StandardScaler()
# Fit and transform the selected columns
#df_encoded[['Number_of_Storeys', 'Year_Built']] = scaler.fit_transform(df_encoded[['Number_of_Storeys', 'Year_Built']])


