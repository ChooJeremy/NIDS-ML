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


save_model = True

pd.set_option('display.max_columns', 50)
#normal_dirs = ["/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/1",
#               "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/2",
#               "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/3"]
normal_dirs =  ["/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/1", 
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/2", 
                "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/3"]
                #"/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/1", 
                #"/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/2", 
                #"/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/3"]
                #"/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/POODLE_TLS/1", 
                #"/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/POODLE_TLS/2", 
                #"/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/POODLE_TLS/3",
                #"/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/1", 
                #"/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/2", 
                #"/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/3"]
# normal_dirs =  ["/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/breach/1", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/outsslv3/poodle/1", 
#                 "/home/3244-1910-0002-X/NIS_TestData/Normal/normal/output_TLS/RC4/1"]
breach_dirs = ["/home/3244-1910-0002-X/NIS_TestData/Breach/breach_traffic"]
               #"/home/3244-1910-0002-X/NIS_TestData/RC4/captures",
               #"/home/3244-1910-0002-X/NIS_TestData/Poodle"]
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
avg_row = 570

num_of_test_files = 0
limit = 20
# to make each have same time steps joining it to the dataframe
for parent_dir in file_list:
    count = 0
    start = 0
    for f in os.listdir(parent_dir):
        if parent_dir in normal_dirs:
            limit = 90
        else: 
            limit = 270
        with open(os.path.join(parent_dir, f), 'r') as fp:
            if not f.endswith(".json"):
                continue
            jsonstr = json.load(fp)
        start = start + 1
        print("Reading file:", os.path.join(parent_dir, f))
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
        else:
            d1.insert(d1.shape[1], "class", [1] * d1.shape[0], True)
        
        num_of_test_files += 1
        test_dataframe = test_dataframe.append(d1, ignore_index=True)

        count += 1
        if limit_files and start > limit:
            break
#Read the test file
#test_file="/home/3244-1910-0002-X/realtime_detection/test2.json"
#with open(test_file, 'r') as fp:
#    jsonstr = json.load(fp)
##start = start + 1
#d1 = preprocess(jsonstr) # the dataframe of a pcap file
#if d1.shape[0]<avg_row: # padding the missing rows as -1
#    neg_arr = np.repeat(-1,d1.shape[1])
#    neg_df = np.repeat([neg_arr], avg_row-d1.shape[0], axis=0)
#    neg_df = pd.DataFrame(neg_df)
#    new_cols = {x: y for x, y in zip(neg_df.columns, d1.columns)} # ensure the new columns are lined up
#    d1 = d1.append(neg_df.rename(columns=new_cols), ignore_index=True)
#elif d1.shape[0]>avg_row:
#    d1 = d1.iloc[:avg_row]
#d1.insert(d1.shape[1], "class", [1] * d1.shape[0], True)        
#num_of_test_files += 1
#test_dataframe = test_dataframe.append(d1, ignore_index=True)
#count += 1
#
# converting all columns to numeric and categorical and dropping columns not approved
#d = cvt_to_numeric_and_cat_and_drop_not_approved_col(d, approved_columns)
#val_dataframe = cvt_to_numeric_and_cat_and_drop_not_approved_col(val_dataframe, approved_columns)
#print(test_dataframe)
#print("Before call")
test_dataframe = cvt_to_numeric_and_cat_and_drop_not_approved_col(test_dataframe, approved_columns)
#print(test_dataframe.shape)
#print(test_dataframe.select_dtypes(include='category').shape)
#print("After call")
#print(test_dataframe)

d = pd.read_pickle('trained_data.ml')
print("Data loaded")
# Matching the columns in both test data and train data
#common_columns = list(set(d.columns).intersection(val_dataframe.columns))
common_columns = np.load("common_columns.npy", allow_pickle=True)
#common_columns = ['_source_layers_tcp_tcp.window_size_scalefactor', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: SessionTicket TLS (len=0)_ssl.handshake.extension.len', '_source_layers_eth_eth.src_tree_eth.lg', '_source_layers_tcp_tcp.srcport', '_source_layers_tcp_tcp.options_tree_tcp.options.mss', '_source_layers_ssl_ssl.record_ssl.handshake', '_source_layers_ssl_ssl.record_ssl.handshake_ssl.handshake.type', '_source_layers_tcp_tcp.flags_tree_tcp.flags.syn_tree__ws.expert__ws.expert.group', '_source_layers_eth_eth.dst_tree_eth.dst_resolved', '_source_layers_eth_eth.src_tree_eth.src_resolved', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: ec_point_formats (len=4)_ssl.handshake.extensions_ec_point_formats_length', '_source_layers_tcp_tcp.flags_tree_tcp.flags.ack', '_source_layers_tcp_tcp.flags_tree_tcp.flags.res', '_source_layers_frame_frame.offset_shift', '_source_layers_tcp_tcp.flags_tree_tcp.flags.syn', '_source_layers_ssl_ssl.record_ssl.handshake_ssl.handshake.version', '_source_layers_tcp_tcp.flags_tree_tcp.flags.syn_tree__ws.expert_tcp.connection.sack', '_source_layers_tcp_tcp.flags_tree_tcp.flags.fin_tree__ws.expert__ws.expert.group', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: renegotiation_info (len=1)_ssl.handshake.extension.type', '_source_layers_tcp_tcp.port', '_source_layers_tcp_tcp.flags_tree_tcp.flags.push', '_source_layers_ssl_ssl.record_ssl.handshake_ssl.handshake.session_id_length', '_source_layers_ssl_ssl.record_ssl.handshake_ssl.handshake.comp_methods_length', '_source_layers_tcp_tcp.flags_tree_tcp.flags.ns', '_source_layers_eth_eth.dst_tree_eth.addr', '_source_layers_tcp_tcp.hdr_len', '_source_layers_tcp_Timestamps_tcp.time_delta', '_source_layers_frame_frame.ignored', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: ec_point_formats (len=4)_ssl.handshake.extension.len', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: renegotiation_info (len=1)_ssl.handshake.extension.len', '_source_layers_tcp_tcp.ack', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: ec_point_formats (len=4)_ssl.handshake.extensions_ec_point_formats_ssl.handshake.extensions_ec_point_format', '_type', '_source_layers_eth_eth.src_tree_eth.ig', 'class', '_source_layers_tcp_tcp.flags_tree_tcp.flags.syn_tree__ws.expert__ws.expert.message', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: SessionTicket TLS (len=0)_ssl.handshake.extension.data', '_source_layers_tcp_tcp.window_size', '_source_layers_eth_eth.dst_tree_eth.ig', '_source_layers_eth_eth.type', '_source_layers_eth_eth.padding', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: SessionTicket TLS (len=0)_ssl.handshake.extension.type', '_source_layers_eth_eth.dst_tree_eth.lg', '_source_layers_frame_frame.marked', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: renegotiation_info (len=1)_Renegotiation Info extension_ssl.handshake.extensions_reneg_info_len', '_source_layers_tcp_tcp.seq', '_source_layers_tcp_tcp.flags_tree_tcp.flags.ecn', '_source_layers_tcp_tcp.window_size_value', '_source_layers_frame_frame.time_delta', '_source_layers_ssl_ssl.record_ssl.handshake_ssl.handshake.comp_method', '_source_layers_tcp_tcp.flags_tree_tcp.flags.cwr', '_source_layers_eth_eth.dst', '_source_layers_tcp_tcp.nxtseq', '_source_layers_frame_frame.time_delta_displayed', '_source_layers_tcp_tcp.flags_tree_tcp.flags.urg', '_source_layers_tcp_tcp.flags_tree_tcp.flags.fin', '_source_layers_eth_eth.src_tree_eth.addr_resolved', '_source_layers_ssl_ssl.record_ssl.record.content_type', '_source_layers_frame_frame.time_relative', '_source_layers_tcp_tcp.len', '_source_layers_tcp_tcp.urgent_pointer', '_source_layers_eth_eth.src', '_source_layers_tcp_tcp.checksum.status', '_source_layers_eth_eth.src_tree_eth.addr', '_source_layers_tcp_tcp.flags_tree_tcp.flags.fin_tree__ws.expert__ws.expert.message', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: extended_master_secret (len=0)_ssl.handshake.extension.len', '_source_layers_frame_frame.cap_len', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: ec_point_formats (len=4)_ssl.handshake.extension.type', '_source_layers_tcp_tcp.flags_tree_tcp.flags.fin_tree__ws.expert__ws.expert.severity', '_source_layers_tcp_tcp.stream', '_source_layers_tcp_tcp.flags_tree_tcp.flags.syn_tree__ws.expert__ws.expert.severity', '_source_layers_tcp_Timestamps_tcp.time_relative', '_source_layers_tcp_tcp.options_tree_tcp.options.mss_tree_tcp.option_len', '_source_layers_tcp_tcp.flags_tree_tcp.flags.str', '_source_layers_tcp_tcp.options_tree_tcp.options.mss_tree_tcp.option_kind', '_source_layers_ssl_ssl.record_ssl.handshake_Extension: extended_master_secret (len=0)_ssl.handshake.extension.type', '_source_layers_tcp_tcp.flags_tree_tcp.flags.fin_tree__ws.expert_tcp.connection.fin', '_source_layers_frame_frame.encap_type', '_source_layers_ssl_ssl.record_ssl.record.version', '_source_layers_tcp_tcp.flags', '_source_layers_frame_frame.len', '_source_layers_eth_eth.dst_tree_eth.addr_resolved', '_source_layers_ssl', '_index', '_source_layers_tcp_tcp.flags_tree_tcp.flags.reset', '_source_layers_tcp_tcp.options_tree_tcp.options.mss_tree_tcp.options.mss_val', '_source_layers_ssl_ssl.record_ssl.handshake_ssl.handshake.comp_methods_ssl.handshake.comp_method', '_source_layers_tcp_tcp.dstport', '_source_layers_frame_frame.number']
#Make our test set fit the common columns
d = d[common_columns]
new_d = d.copy()
#Convert test_dataframe to only those containing the common_columns
test_dataframe = test_dataframe.reindex(common_columns, axis="columns")
#test_dataframe = test_dataframe[common_columns]
# TODO: Values that don't match, we replace with the mean from our train set. (d.mean() or d.mode() or d.median(). Right now, it's just -1)
print(test_dataframe.shape)
print(new_d.shape)
for i in test_dataframe:
    print(i)
    print(test_dataframe[i].dtype, " " , new_d[i].dtype)
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

# Save our training data so we don't need to redo everything
#d.to_pickle('trained_data.ml')
#np.save("common_columns.npy", common_columns)

# Ensure the columns match up
cols = list(new_d.columns.values)
cols.sort()
new_d = new_d[cols]
#val_data = val_data[cols]
test_data = test_data[cols]
#test_data = test_data.reindex(cols, axis="columns")
# Values that don't match, we replace with the mean from our train set.
#test_data.where(pd.notna(test_data), -1, axis='rows')

print(test_data.shape)
print(new_d.shape)

# Scaling
count = 0
scaler=MinMaxScaler(feature_range=(0,1))
for i in new_d: # each column in the train dataframe
    if new_d[i].max() > 1: # if the data is exceeds 1, scale it. I do not do -1 to 1 because i am using -1 to signify missing steps
        new_d[i] = scaler.fit_transform(new_d[[i]])
        #val_data[i] = scaler.transform(val_data[[i]])
        test_data[i] = scaler.transform(test_data[[i]])


# converting train data from 2d array into 3d array
#for i in range(0, num_of_train_files):
#    dy_train += [sum(y_train[0:avg_row]) / (avg_row)]
#    df1 = new_d.iloc[0:avg_row]
#    print("Train: i is: {} and d's shape is: {} and stacked_df's shape is: {} and df1's shape is: {}".format(i, new_d.shape, stacked_df.shape, df1.shape))
#    if i == 0:
#        stacked_df = df1.copy()
#        print("Train: We have just copied the df")
#    elif i == 1:
#        stacked_df = np.stack([stacked_df,df1])
#        print("Train: We have stacked the df")
#    else:
#        df1 = np.expand_dims(df1, axis=0)
#        stacked_df = np.concatenate((stacked_df, df1), axis=0)
#        print("Train: We have concat the df")
#    new_d = new_d.iloc[avg_row:]
#print(stacked_df.shape)



# convert val data from 2d array into 3d array
twoDim = True
#for i in range(0, num_of_val_files):
#    df1 = val_data.iloc[0:avg_row]
#    dy_val += [sum(y_val[0:avg_row]) / (avg_row)]
#    
#    print("Val: i is: {} and d's shape is: {} and stacked_df's shape is: {} and df1's shape is: {}".format(i, val_data.shape, val_stacked_df.shape, df1.shape))
#    if i == 0:
#        val_stacked_df = df1.copy()
#        print("Val: We have just copied the df")
#    elif i == 1:
#        twoDim = False
#        val_stacked_df = np.stack([val_stacked_df,df1])
#        print("Val: We have stacked the df")
#    else:
#        df1 = np.expand_dims(df1, axis=0)
#        val_stacked_df = np.concatenate((val_stacked_df, df1), axis=0)
#        print("Val: We have concat the df")
#    val_data = val_data.iloc[avg_row:]
#if twoDim:
#    val_stacked_df = np.expand_dims(val_stacked_df, axis=0)
#print(val_stacked_df.shape)


# convert test data from 2d array into 3d array
twoDim = True
print("y_test", y_test)
for i in range(0, num_of_test_files):
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
print(dy_test)


test_stacked_df = tf.convert_to_tensor(test_stacked_df, np.float64)

# model
session = tf.Session()

with session:
    # define model
    model = load_model("model.h5")
    print(model.summary())

    # finding the error
    print("Autoencoder Test:")
    yhat = model.predict(test_stacked_df, verbose=0, steps=1)
    totalError = 0
    with session.as_default():
        test_stacked_df_eval = test_stacked_df.eval()
        for i in range(0, len(yhat[0,0,:])):
            err = abs(test_stacked_df_eval[0,0,i]-yhat[0,0,i])
            print("Expected: {:<25} Pred: {:<25} Difference: {:<25}".format(test_stacked_df_eval[0,0,i], yhat[0,0,i], err))
            totalError += err
    print("Total Error: ", totalError)
    results = model.evaluate(test_stacked_df, test_stacked_df, verbose=1, steps=(num_of_test_files))
    print(model.metrics_names)
    print(results)


    # define classifier
    classifier = load_model("classifier.h5")
    print(classifier.summary())

    # finding the error
    print("Classifier Test:")
    yhat = classifier.predict(test_stacked_df, verbose=0, steps=num_of_test_files)

    print(len(yhat))
    print(len(dy_test))
    for i in range(0, len(dy_test)):
        print("Expected: ",dy_test[i],", got: ",'%f' % yhat[i][0])
    # print(yhat)
    if yhat[0][0] < 0.5:
        malicious = 0;
    else:
        malicious = 1;

    print(malicious)
    totalError = 0

    print(dy_test)
    
    results = classifier.evaluate(test_stacked_df, dy_test, verbose=1, steps=(num_of_test_files))
    print(classifier.metrics_names)
    print(results)
    
    with session.as_default():
        test_stacked_df_eval = test_stacked_df.eval()
        for i in range(0, len(yhat) - 1):
            err = abs(test_stacked_df_eval[0,0,i]-yhat[i])
            totalError += err
    print("Total Error: ", totalError)

