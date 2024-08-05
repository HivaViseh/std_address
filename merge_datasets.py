import pandas as pd
from fuzzywuzzy import process
import numpy as np


def fuzzy_match(df1, df2, column_1, column_2, threshold=90):
    unique_values = df2[column_2].unique()

    match = {}
    match_column = []

    for value in unique_values:
        column_list = df1[column_1].astype('string').tolist()
        m, r = process.extractOne(str(value), column_list)
 
        if r >= threshold:
            match[value] = m

        else:
            match[value] = None

    for index, row in df2.iterrows():
        match_column.append(match.get(row[column_2], np.nan))

    return match_column   


