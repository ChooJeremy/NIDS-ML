import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
import json
import random
import matplotlib.pyplot as plt

from array import array
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Masking, Dropout
from tensorflow.keras.layers import LeakyReLU


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def flatten_json(nested_json, exclude=['']):
    """Flatten json object with nested keys into a single level.
        Args:
            nested_json: A nested json object.
            exclude: Keys to exclude from output.
        Returns:
            The flattened json object if successful, None otherwise.
    """
    out = {}

    def flatten(x, name='', exclude=exclude):
        if type(x) is dict:
            for a in x:
                if a not in exclude: flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out

def cvt_to_decimal(x):
    if (str(x)!="nan"):
        x = x.replace(":","")
        return int(x, 16)

def preprocess(jsonstr):
    d = pd.DataFrame([flatten_json(x) for x in jsonstr])

    d = d.drop("_score", axis=1)
    # d = d["_source_layers_tcp_tcp.payload"].apply(cvt_to_decimal)
    # d = d["_source_layers_tcp_tcp.segment_data"].apply(cvt_to_decimal)
    d["_source_layers_ip_ip.id"] = d["_source_layers_ip_ip.id"].apply(cvt_to_decimal)
    return d


def cvt_to_numeric_and_cat_and_drop_not_approved_col(d, approved_columns):
    columns_dropped = []
    for i in d:
        if i not in approved_columns:
            d  = d.drop(i, axis=1)
            continue
        u = d[i].unique().tolist()
        if (u == [np.nan]): # is just [nan]
            d = d.drop(i, axis=1)
            print("Column {} has been dropped".format(i))
            continue
        if (len(u)<3 and (np.nan in u)): # the values are only [number, nan]
            d[i] = [x for x in u if not pd.isnull(x)][0]
            print("Column {} has been massaged into one value".format(i))
        if (u == [None]):
            d = d.drop(i, axis=1)
            columns_dropped.append(i)
            print("Column {} has been dropped".format(i))
            continue
        try:
            d[i] = pd.to_numeric(d[i])
            d[i] = d[i].astype(np.int32)
            print("Column {} has been successfully converted to {}".format(i, d[i].dtype))
        except:
            if (d[i].nunique() < 20):
                d[i] = pd.Categorical(d[i])
                print("Column {} has been successfully converted to {}".format(i, d[i].dtype))
            else:
                columns_dropped.append(i)
                print("Column {} cannot be converted and will be dropped. It has: {} unique values".format(i, d[i].nunique()))
                d = d.drop(i, axis=1)
    return d



b_size = 64
num_of_test_files = 20
num_of_val_files = 30

#pd.set_option('display.max_columns', 50)

# normal_dirs =  ["/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/1", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/2", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/3",	
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/1", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/2", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/3",
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/POODLE_TLS/1", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/POODLE_TLS/2", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/POODLE_TLS/3",
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/1", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/2", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/3"]
# breach_dirs = ["/home/3244-1910-0002-X/NIS_TestData/Breach/breach_traffic",
#                "/home/3244-1910-0002-X/NIS_TestData/RC4/captures", 
#                "/home/3244-1910-0002-X/NIS_TestData/Poodle"]
normal_dirs =  ["/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/1"]
breach_dirs = ["/home/3244-1910-0002-X/NIS_TestData/Breach/breach_traffic"]

file_list = breach_dirs + normal_dirs

d = pd.DataFrame()
valiation_dataframe = pd.DataFrame()
test_dataframe = pd.DataFrame()

stacked_df = pd.DataFrame()
test_stacked_df = pd.DataFrame()
val_stacked_df = pd.DataFrame()

row, col, count, num_of_files = (0,)*4
freq = {}
approved_columns = []

limit_files = True
limit = 150

# to find the average number of rows in each file
for parent_dir in file_list:
    start = 0
    count = 0
    for f in os.listdir(parent_dir):
        with open(os.path.join(parent_dir, f), 'r') as fp:
            if not f.endswith(".json"):
                continue
            jsonstr = json.load(fp)
        print(str(parent_dir) + "/" + str(f))
        start = start + 1
        if (count < num_of_test_files+num_of_val_files):
            count += 1
            continue

        d1 = preprocess(jsonstr)
        row += d1.shape[0]
        
        # 1 for attack, 0 for normal data
        if parent_dir in normal_dirs:
            d1.insert(d1.shape[1], "class", [0] * d1.shape[0], True)
            print("Normal")
            print(parent_dir)
        else:
            d1.insert(d1.shape[1], "class", [1] * d1.shape[0], True)
            print("Attack")
            print(parent_dir)


        for i in d1:
            if (i in freq):
                freq[i] += 1.0
            else:
                freq[i]  = 1.0
        count += 1
        if limit_files and start > limit:
            break
avg_row = row // count

# finding which columns to keep
for i in freq:
    if not ((freq[i]/count) < 0.5) and not ("ip" in i) and not ("epoch" in i):
        approved_columns.append(i)
print("Approved columns length: ", len(approved_columns))
np.save("approved_columns.npy", approved_columns)


# to make each have same time steps joining it to the dataframe
for parent_dir in file_list:
    start = 0
    count = 0
    for f in os.listdir(parent_dir):
        with open(os.path.join(parent_dir, f), 'r') as fp:
            if not f.endswith(".json"):
                continue
            jsonstr = json.load(fp)
        start = start + 1
        d1 = preprocess(jsonstr) # the dataframe of a pcap file
        if d1.shape[0]<avg_row: # padding the missing rows as -1
            neg_arr = np.repeat(-1,d1.shape[1])
            neg_df = np.repeat([neg_arr], avg_row-d1.shape[0], axis=0)
            neg_df = pd.DataFrame(neg_df)
            new_cols = {x: y for x, y in zip(neg_df.columns, d1.columns)} # ensure the new columns are lined up
            d1 = d1.append(neg_df.rename(columns=new_cols), ignore_index=True)
        elif d1.shape[0]>avg_row:
            d1 = d1.iloc[:avg_row]

        # 1 for attack, 0 for normal data
        if parent_dir in normal_dirs:
            d1.insert(d1.shape[1], "class", [0] * d1.shape[0], True)
            print("Normal")
            print(parent_dir)
        else:
            d1.insert(d1.shape[1], "class", [1] * d1.shape[0], True)
            print("Attack")
            print(parent_dir)
        
        if (count < num_of_test_files):
            test_dataframe = test_dataframe.append(d1, ignore_index=True)
            print("Added {} to test and it now has {} shape".format(f,test_dataframe.shape))
        elif (count-num_of_test_files < num_of_val_files):
            valiation_dataframe = valiation_dataframe.append(d1, ignore_index=True)
            print("Added {} to validation and it now has {} shape".format(f,valiation_dataframe.shape))
        else:
            num_of_files += 1
            d = d.append(d1, ignore_index=True)
            print("Added {} to train and it now has {} shape".format(f,d.shape))
        count += 1
        
        if limit_files and start > limit:
            break


# converting all columns to numeric and categorical and dropping columns not approved
d = cvt_to_numeric_and_cat_and_drop_not_approved_col(d, approved_columns)
test_dataframe = cvt_to_numeric_and_cat_and_drop_not_approved_col(test_dataframe, approved_columns)
valiation_dataframe = cvt_to_numeric_and_cat_and_drop_not_approved_col(valiation_dataframe, approved_columns)

print("Saving trained_data to pickle")
print(d.describe())
d.to_pickle('trained_data.ml');
# Matching the columns in both test data and train data
train_col = d.columns
test_col = test_dataframe.columns
val_col = valiation_dataframe.columns #EL NUMERO!!!!!
common_columns = list(set(train_col).intersection(test_col, val_col)) #EL NUMERO!!!!!
d = d[common_columns]
test_dataframe = test_dataframe[common_columns]
valiation_dataframe = valiation_dataframe[common_columns] #EL NUMERO!!!!!
np.save("common_columns.npy", common_columns)


new_d = d.copy()
test_data = test_dataframe.copy()
val_data = valiation_dataframe.copy()


# Get y values
y_train = new_d["class"]
y_val = val_data["class"]
y_test = test_data["class"]
new_d = new_d.drop(columns = ["class"])
val_data = val_data.drop(columns = ["class"])
test_data = test_data.drop(columns = ["class"])
dy_train = []
dy_val = []
dy_test = []

print("Train")
print(len(y_train))
for i in range(len(y_train)):
    print(y_train[i])
print("Val")
print(len(y_val))
for i in range(len(y_val)):
    print(y_val[i])
print("Train")
print(len(y_test))
for i in range(len(y_test)):
    print(y_test[i])

# one hot of categorical variables
for i in new_d.select_dtypes(include='category'):
    new_d[i] = new_d[i].cat.codes
for i in test_data.select_dtypes(include='category'):
    test_data[i] = test_data[i].cat.codes
for i in val_data.select_dtypes(include='category'):
    val_data[i] = val_data[i].cat.codes


# Ensure the columns match up
cols = list(new_d.columns.values)
cols.sort()
new_d = new_d[cols]
test_data = test_data[cols]
val_data = val_data[cols]


# Scaling
count = 0
scaler=MinMaxScaler(feature_range=(0,1))
for i in new_d: # each column in the train dataframe
    if new_d[i].max() > 1: # if the data is exceeds 1, scale it. I do not do -1 to 1 because i am using -1 to signify missing steps
        new_d[i] = scaler.fit_transform(new_d[[i]])
        test_data[i] = scaler.transform(test_data[[i]])
        val_data[i] = scaler.transform(val_data[[i]])


# converting train data from 2d array into 3d array
for i in range(0, num_of_files):
    df1 = new_d.iloc[0:avg_row]
    dy_train += [sum(y_train[0:avg_row]) / avg_row]
    #dy_train.append(y_train[avg_row*i])
    y_train = y_train[avg_row:]

    print("Train: i is: {} and d's shape is: {} and stacked_df's shape is: {} and df1's shape is: {}".format(i, new_d.shape, stacked_df.shape, df1.shape))
    if i == 0:
        stacked_df = df1.copy()
        print("Train: We have just copied the df")
    elif i == 1:
        stacked_df = np.stack([stacked_df,df1])
        print("Train: We have stacked the df")
    else:
        df1 = np.expand_dims(df1, axis=0)
        stacked_df = np.concatenate((stacked_df, df1), axis=0)
        print("Train: We have concat the df")
    new_d = new_d.iloc[avg_row:]
print(stacked_df.shape)



# convert test data from 2d array into 3d array
twoDim = True
for i in range(0, num_of_test_files * len(file_list)):
    df1 = test_data.iloc[0:avg_row]
    dy_test += [sum(y_test[0:avg_row]) / avg_row]
   # dy_test.append(y_test[avg_row*i])
    y_test = y_test[avg_row:]

    print("Test: i is: {} and d's shape is: {} and stacked_df's shape is: {} and df1's shape is: {}".format(i, test_data.shape, test_stacked_df.shape, df1.shape))
    if i == 0:
        test_stacked_df = df1.copy()
        print("Test: We have just copied the df")
    elif i == 1:
        twoDim = False
        test_stacked_df = np.stack([test_stacked_df,df1])
        print("Test: We have stacked the df")
    else:
        df1 = np.expand_dims(df1, axis=0)
        test_stacked_df = np.concatenate((test_stacked_df, df1), axis=0)
        print("Test: We have concat the df")
    test_data = test_data.iloc[avg_row:]
if twoDim:
    test_stacked_df = np.expand_dims(test_stacked_df, axis=0)
print(test_stacked_df.shape)



# convert val data from 2d array into 3d array
twoDim = True
for i in range(0, num_of_val_files * len(file_list)):
    df1 = val_data.iloc[0:avg_row]
    dy_val += [sum(y_val[0:avg_row]) / avg_row]
    #dy_val.append(y_val[avg_row*i])
    y_val = y_val[avg_row:]

    print("Val: i is: {} and d's shape is: {} and stacked_df's shape is: {} and df1's shape is: {}".format(i, new_d.shape, stacked_df.shape, df1.shape))
    if i == 0:
        val_stacked_df = df1.copy()
        print("Val: We have just copied the df")
    elif i == 1:
        twoDim = False
        val_stacked_df = np.stack([val_stacked_df,df1])
        print("Val: We have stacked the df")
    else:
        df1 = np.expand_dims(df1, axis=0)
        val_stacked_df = np.concatenate((val_stacked_df, df1), axis=0)
        print("Val: We have concat the df")
    val_data = val_data.iloc[avg_row:]
if twoDim:
    val_stacked_df = np.expand_dims(val_stacked_df, axis=0)
print(val_stacked_df.shape)

print("BAR ---------------------------------------------------")
print(y_test)
print(dy_test)
print(dy_train)
print(dy_val)   

# model
session = tf.Session()

with session:
    # define model
    model = Sequential()
    timesteps = stacked_df.shape[1]
    n_features = stacked_df.shape[2]
    val_stacked_df = tf.convert_to_tensor(val_stacked_df, np.float64)
    test_stacked_df = tf.convert_to_tensor(test_stacked_df, np.float64)
    stacked_df = tf.convert_to_tensor(stacked_df, np.float64) #uncomment if you are not normalizing above

    model.add(Masking(mask_value=-1., input_shape=(stacked_df.shape[1], stacked_df.shape[2])))
    model.add(LSTM(150, return_sequences=True))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False, name='bottleneck'))
    model.add(LeakyReLU(alpha=0.05))
    print("Encoder added")
    model.add(RepeatVector(timesteps))
    print("Adding Decoder")
    model.add(LSTM(50, return_sequences=True))
    model.add(LeakyReLU(alpha=0.05))
    model.add(LSTM(100, return_sequences=True))
    model.add(LeakyReLU(alpha=0.05))
    model.add(LSTM(150, return_sequences=True))
    model.add(LeakyReLU(alpha=0.05))
    model.add(TimeDistributed(Dense(n_features)))
    adam = keras.optimizers.Adam()
    model.compile(optimizer=adam, loss='mse',  metrics=["accuracy"])
    print(model.summary())
    mc1 = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # Training
    print("Train data: ")
    print(stacked_df.shape)
    print(len(dy_train))
    print(dy_train)
    print("Test data: ")
    print(test_stacked_df.shape)
    print(len(dy_test))
    print(dy_test)
    print("Validation data: ")
    print(val_stacked_df.shape)
    print(len(dy_val))
    print(dy_val)

    model.fit(stacked_df, stacked_df, epochs=1000, steps_per_epoch=(num_of_files//b_size)+1, verbose=1, validation_data=(val_stacked_df,val_stacked_df), validation_steps=(num_of_val_files* len(file_list)//b_size)+1)

    # Save model
    model.save("model.h5")
    print("Saved model to disk")

    # get the encoded features
    tf.keras.backend.set_session(session)
    def get_input_tensor(model):
        return model.layers[0].input
    def get_bottleneck_tensor(model):
        return model.get_layer(name='bottleneck').output
    with session.as_default():
            t_input = get_input_tensor(model)
            t_enc = get_bottleneck_tensor(model)
            # enc will store the actual encoded values of x
            enc = session.run(t_enc, feed_dict={t_input:stacked_df.eval()})
    print("enc shape is: ",enc.shape)


    # finding the error
    yhat = model.predict(test_stacked_df, verbose=0, steps=1)
    totalError = 0
    with session.as_default():
        test_stacked_df_eval = test_stacked_df.eval()
        for i in range(0, len(yhat[0,0,:])):
            err = abs(test_stacked_df_eval[0,0,i]-yhat[0,0,i])
            print("Expected: {:<25} Pred: {:<25} Difference: {:<25}".format(test_stacked_df_eval[0,0,i], yhat[0,0,i], err))
            totalError += err
    print("Total Error: ", totalError)

    # Testing
    results = model.evaluate(test_stacked_df, test_stacked_df, verbose=1, steps=(num_of_test_files* len(file_list)//b_size +1))
    print(model.metrics_names)
    print(results)

    encoder_layer = Model(inputs=model.input, outputs=model.get_layer("leaky_re_lu_2").output)
    encoder_layer.trainable = False
 
    # define classifier
    classifier = Sequential()
    classifier.add(encoder_layer)    

    classifier.add(Dense(100, activation="relu"))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(100, activation="relu"))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(1, activation="sigmoid"))
    print(classifier.summary())
 
    classifier.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # Training
    print("Train data: ")
    print(stacked_df.shape)
    print(len(dy_train))
    print("Test data: ")
    print(test_stacked_df.shape)
    print(len(dy_test))
    print("Validation data: ")
    print(val_stacked_df.shape)
    print(len(dy_val))
    classifier_output = classifier.fit(stacked_df, dy_train, epochs=100, steps_per_epoch=(num_of_files//b_size)+1, verbose=1, validation_data=(val_stacked_df, dy_val), validation_steps=(num_of_val_files* len(file_list)//b_size)+1)
    
    # Save classifier
    classifier.save("classifier.h5")
    print("Saved classifier to disk")
 
    print("Training accuracy: ", np.mean(classifier_output.history["acc"]))
    print("Validation accuracy: ", np.mean(classifier_output.history["val_acc"]))

    # Testing
    results = classifier.evaluate(test_stacked_df, dy_test, verbose=1, steps=(num_of_test_files* len(file_list)//b_size))
    print(classifier.metrics_names)
    print(results)
