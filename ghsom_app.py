import numpy as np
import pandas as pd
from pandas import ExcelWriter
import openpyxl
import datetime
from clustering.ghsom import GHSOM
from clustering.som import SOM
from model import reformat


#-----------------------------------Data pre-process-------------------------------------------------------
# get  training data
df = pd.read_excel('./data/animals.xlsx')

# define which title to be noimal

# df_nominal = df.ix[:, ['Name']]
df_numerical_tmp = df.ix[:, ['Legs', 'Feather', 'Swim', 'Fly', 'Eating Patterns']]
df_numerical = df_numerical_tmp.apply(pd.to_numeric, errors='coerce').fillna(0)
# coerce: if data is null, put NaN (fillna(0) makes it 0)

# sum_of_attribute = np.array(df_numerical.sum(0)) / len(df_numerical.index)

# concat nominal title to _201_107_Zer_T6W_TOS_Joe format
# title_concat( @param1(df): nominal_dataframe,
#               @param2(int): first N char )
# label_abbr = reformat.title_concat(df_nominal,3);

# force str to numeric datatype (if not => NaN) then replace NaN to 0


# get data dim to latter SOM prcess
input_dim = len(df_numerical.columns)
input_row = len(df_numerical.index)
# change data to np array (SOM accept nparray format)
input_data = np.array(df_numerical)



#-----------------------------------SOM process-------------------------------------------------------

# Train a 20x30 SOM with 400 iterations
# print("input_dim =", input_dim)
ghsom = GHSOM(2, 2, input_data, 0.1, 0.1)
som = SOM(2, 2, input_dim, 50)

print('training start : ' + str(datetime.datetime.now()))
som.train(input_data)

# Map data to their closest neurons
mapped = som.map_vects(input_data)
result = np.array(mapped)

new_input, mqe_array = ghsom.clustering_input_data(input_data, result, som._weightages)

if not ghsom.tau1_check(mqe_array):
    error_unit = ghsom.find_error_unit(mqe_array, som._weightages, new_input)
    print("error unit = ", error_unit)
else:
    # for value, input_array in zip(mqe_array, new_input):
    #    if not ghsom.tau2_check(value):
    #        print("som start ", new_input)
    print("GHSOM clustering is done.")

# output_np = np.concatenate((input_data, result), axis=1)

# -------------------------------------Output format-----------------------------------------------------

# output format
# output_np = np.concatenate((df_nominal, result), axis=1)
# output_pd = pd.DataFrame(data=output_np, columns=['Report Date', 'Customer', 'Type', 'Item Short Name', 'Brand', 'Sales', 'axis-x', 'axis-y'])
# print(output_pd)


# write to final csv
# output_pd.to_csv('./result/result.csv')
