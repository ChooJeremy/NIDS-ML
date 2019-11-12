import re
from time import sleep
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
import json
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

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Masking, Dropout
from tensorflow.keras.losses import MeanAbsolutePercentageError, MeanSquaredLogarithmicError
from tensorflow.keras.layers import LeakyReLU



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
            #print("Column {} has been dropped".format(i))
            continue
        if (len(u)<3 and (np.nan in u)): # the values are only [number, nan]
            d[i] = [x for x in u if not pd.isnull(x)][0]
            #print("Column {} has been massaged into one value".format(i))
        if (u == [None]):
            d = d.drop(i, axis=1)
            columns_dropped.append(i)
            #print("Column {} has been dropped".format(i))
            continue
        try:
            d[i] = pd.to_numeric(d[i])
            d[i] = d[i].astype(np.int32)
            #print("Column {} has been successfully converted to {}".format(i, d[i].dtype))
        except:
            if (d[i].nunique() < 20):
                d[i] = pd.Categorical(d[i])
                #print("Column {} has been successfully converted to {}".format(i, d[i].dtype))
            else:
                columns_dropped.append(i)
                #print("Column {} cannot be converted and will be dropped. It has: {} unique values".format(i, d[i].nunique()))
                d = d.drop(i, axis=1)
    return d


save_model = True

pd.set_option('display.max_columns', 50)
#normal_dirs = ["/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/1",
#               "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/2",
#               "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/3"]
normal_dirs =  ["/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/1", 
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/2", 
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/3",    
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/1", 
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/2", 
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/3",
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/POODLE_TLS/1", 
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/POODLE_TLS/2", 
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/POODLE_TLS/3",
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/1", 
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/2", 
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/3"]
# normal_dirs =  ["/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/1", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/1", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/1"]
breach_dirs = ["/home/3244-1910-0002-X/NIS_TestData/Breach/breach_traffic",
               "/home/3244-1910-0002-X/NIS_TestData/RC4/captures", 
               "/home/3244-1910-0002-X/NIS_TestData/Poodle"]
file_list = breach_dirs + normal_dirs

# train data
d = pd.DataFrame()
stacked_df = pd.DataFrame()
dy_train = []

# validation data
val_dataframe = pd.DataFrame()
val_stacked_df = pd.DataFrame()
dy_val = []

# test data
test_dataframe = pd.DataFrame()
test_stacked_df = pd.DataFrame()
dy_test = []

row, col, count, num_of_train_files, num_of_val_files, num_of_test_files = (0,)*6
freq = {}
approved_columns = []

limit_files = True
limit = 100
max_files = limit * len(file_list)
b_size = 10
num_of_val_files_per_folder = 20
num_of_test_files_per_folder = 200



# to find the average number of rows in each file
#for parent_dir in file_list:
#    start = 0
#    for f in os.listdir(parent_dir):
#        with open(os.path.join(parent_dir, f), 'r') as fp:
#            if not f.endswith(".json"):
#                continue
#            jsonstr = json.load(fp)
#        print(str(parent_dir) + "/" + str(f))
#        start = start + 1
#        if count < num_of_val_files_per_folder:
#            count += 1
#            continue
#        d1 = preprocess(jsonstr)
#        row += d1.shape[0]
#
#        # 1 for attack, 0 for normal data
#        if parent_dir in normal_dirs:
#            d1.insert(d1.shape[1], "class", [0] * d1.shape[0], True)
#        else:
#            d1.insert(d1.shape[1], "class", [1] * d1.shape[0], True)
#
#        for i in d1:
#            if (i in freq):
#                freq[i] += 1.0
#            else:
#                freq[i]  = 1.0
#        count += 1
#        if limit_files and start > limit:
#            break
#avg_row = row // count

# finding which columns to keep
#for i in freq:
#    if not ((freq[i]/count) < 0.5) and not ("ip" in i) and not ("epoch" in i):
#        approved_columns.append(i)
#print("Approved columns length: ", len(approved_columns))
#np.save("approved_columns.npy", approved_columns)

approved_columns = np.load("approved_columns.npy", allow_pickle=True)
#print("Approved columns length: ", len(approved_columns))
#print("Average row: "  + str(avg_row))
avg_row = 1382


# to make each have same time steps joining it to the dataframe
#for parent_dir in file_list:
#    count = 0
#    start = 0
#    for f in os.listdir(parent_dir):
#        with open(os.path.join(parent_dir, f), 'r') as fp:
#            if not f.endswith(".json"):
#                continue
#            jsonstr = json.load(fp)
#        start = start + 1
#        d1 = preprocess(jsonstr) # the dataframe of a pcap file
#        if d1.shape[0]<avg_row: # padding the missing rows as -1
#            neg_arr = np.repeat(-1,d1.shape[1])
#            neg_df = np.repeat([neg_arr], avg_row-d1.shape[0], axis=0)
#            neg_df = pd.DataFrame(neg_df)
#            new_cols = {x: y for x, y in zip(neg_df.columns, d1.columns)} # ensure the new columns are lined up
#            d1 = d1.append(neg_df.rename(columns=new_cols), ignore_index=True)
#        elif d1.shape[0]>avg_row:
#            d1 = d1.iloc[:avg_row]
#
#        # 1 for attack, 0 for normal data
#        if parent_dir in normal_dirs:
#            d1.insert(d1.shape[1], "class", [0] * d1.shape[0], True)
#        else:
#            d1.insert(d1.shape[1], "class", [1] * d1.shape[0], True)
#        
#        if (count < num_of_val_files_per_folder):
#            num_of_val_files += 1
#            val_dataframe = val_dataframe.append(d1, ignore_index=True)
#            print("Added {} to test and it now has {} shape".format(f,val_dataframe.shape))
#        elif (start < limit):
#            num_of_train_files += 1
#            d = d.append(d1, ignore_index=True)
#            print("Added {} to train and it now has {} shape".format(f,d.shape))
#        else:
#            num_of_test_files += 1
#            test_dataframe = test_dataframe.append(d1, ignore_index=True)
#
#        count += 1
#        if limit_files and start > limit:
#            break

d = pd.read_pickle('trained_data.ml')
print("Data loaded")
# Matching the columns in both test data and train data
common_columns = np.load("common_columns.npy", allow_pickle=True)
#Make our test set fit the common columns
d = d[common_columns]


while True:
    # get all the pcap files
    list_dir = "ls ./packet_captured"
    process = os.popen(list_dir)
    output = process.read()
    process.close()
    #print(output)
    pcaps = re.findall(r'(\d+\.\d+\.\d+\.\d+\.\d+\.pcap)', output)
    #print(ips)
    for pcap in pcaps:
        print("analysing %s" %pcap)

        os.system("tshark -r ./packet_captured/" + str(pcap) + " -l -n -T json > ./packet_captured/" + str(pcap) + ".json");

        # call the classifier to decide whether the pcap contains malicious traffic
        #Read the test file

        test_file="./packet_captured/" + str(pcap) + ".json"
        with open(test_file, 'r') as fp:
            jsonstr = json.load(fp)
        #start = start + 1
        d1 = preprocess(jsonstr) # the dataframe of a pcap file
        if d1.shape[0]<avg_row: # padding the missing rows as -1
            neg_arr = np.repeat(-1,d1.shape[1])
            neg_df = np.repeat([neg_arr], avg_row-d1.shape[0], axis=0)
            neg_df = pd.DataFrame(neg_df)
            new_cols = {x: y for x, y in zip(neg_df.columns, d1.columns)} # ensure the new columns are lined up
            d1 = d1.append(neg_df.rename(columns=new_cols), ignore_index=True)
        elif d1.shape[0]>avg_row:
            d1 = d1.iloc[:avg_row]
        d1.insert(d1.shape[1], "class", [0] * d1.shape[0], True)        
        num_of_test_files += 1
        test_dataframe = test_dataframe.append(d1, ignore_index=True)
        count += 1

        new_d = d.copy()

        # converting all columns to numeric and categorical and dropping columns not approved
        test_dataframe = cvt_to_numeric_and_cat_and_drop_not_approved_col(test_dataframe, approved_columns)

        #Only take the common columns. Columns which end up empty (nan), replace with the mean or the mode of the training data.
        test_dataframe = test_dataframe.reindex(common_columns, axis="columns")
        #test_dataframe = test_dataframe[common_columns]
        for i in test_dataframe:
            if str(new_d[i].dtype) == "category":
                test_dataframe[i] = new_d[i].fillna(new_d[i].mode()[0])
            else:
                test_dataframe[i] = test_dataframe[i].fillna(new_d[i].mean())
        #test_dataframe = test_dataframe.where(pd.notna(test_dataframe), -1, axis='rows')

        # Get y values
        
        #val_data = val_dataframe.copy()
        test_data = test_dataframe.copy()

        #y_train = new_d["class"]
        #y_val = val_data["class"]
        y_test = test_data["class"]
        new_d = new_d.drop(columns = ["class"])
        #val_data = val_data.drop(columns = ["class"])
        test_data = test_data.drop(columns = ["class"])


        # one hot of categorical variables
        # Re-represent categorical variables
        test_data = cvt_to_numeric_and_cat_and_drop_not_approved_col(test_data, approved_columns)
        for i in new_d.select_dtypes(include='category'):
            new_d[i] = new_d[i].cat.codes
        for i in test_data.select_dtypes(include='category'):
            test_data[i] = test_data[i].cat.codes


        # Ensure the columns match up
        cols = list(new_d.columns.values)
        cols.sort()
        new_d = new_d[cols]
        #val_data = val_data[cols]
        test_data = test_data[cols]
        #test_data = test_data.reindex(cols, axis="columns")
        # Values that don't match, we replace with the mean from our train set.
        #test_data.where(pd.notna(test_data), -1, axis='rows')

        # Scaling
        count = 0
        scaler=MinMaxScaler(feature_range=(0,1))
        for i in new_d: # each column in the train dataframe
            if new_d[i].max() > 1: # if the data is exceeds 1, scale it. I do not do -1 to 1 because i am using -1 to signify missing steps
                new_d[i] = scaler.fit_transform(new_d[[i]])
                #val_data[i] = scaler.transform(val_data[[i]])
                test_data[i] = scaler.transform(test_data[[i]])


        # convert test data from 2d array into 3d array
        twoDim = True
        for i in range(0, 1):
            df1 = test_data.iloc[0:avg_row]
            dy_test += [sum(y_test[0:avg_row]) / (avg_row)]
            
            #print("Test: i is: {} and d's shape is: {} and stacked_df's shape is: {} and df1's shape is: {}".format(i, test_data.shape, test_stacked_df.shape, df1.shape))
            if i == 0:
                test_stacked_df = df1.copy()
                #print("Test: We have just copied the df")
            elif i == 1:
                twoDim = False
                test_stacked_df = np.stack([test_stacked_df,df1])
                #print("Test: We have stacked the df")
            else:
                df1 = np.expand_dims(df1, axis=0)
                test_stacked_df = np.concatenate((test_stacked_df, df1), axis=0)
                #print("Test: We have concat the df")
            test_data = test_data.iloc[avg_row:]
        if twoDim:
            test_stacked_df = np.expand_dims(test_stacked_df, axis=0)


        test_stacked_df = tf.convert_to_tensor(test_stacked_df, np.float64)

        # model
        session = tf.Session()

        with session:
            # define model
            model = load_model("model.h5")

            # finding the error
            yhat = model.predict(test_stacked_df, verbose=0, steps=1)
            results = model.evaluate(test_stacked_df, test_stacked_df, verbose=1, steps=(1))

            # define classifier
            classifier = load_model("classifier.h5")

            # finding the error
            yhat = classifier.predict(test_stacked_df, verbose=0, steps=1)
            malicious = 0
            print(yhat)
            if yhat[0][0] < 0.5:
                print("Detected non-malicious")
                malicious = 0;
            else:
                print("Detected malicious")
                malicious = 1;
        
        #remember to delete the old pcap file
        os.system("rm ./packet_captured/%s" % pcap)
        os.system("rm ./packet_captured/%s" % pcap + ".json")
        if malicious:
            # get the actual ip
            ip = '.'.join(i for i in pcap.split('.')[:-2])
            # fork a child process to connect to ssh of web server and drop the malicious ip
            pid = os.fork()
            if pid == 0:
                print("ssh root@cs3244.oulove.me 'iptables -I INPUT -s %s -j DROP'" % ip)
                #os.popen("ssh root@cs3244.oulove.me 'iptables -I INPUT -s %s -j DROP'" % ip)
                exit()
    sleep(3)



