from filtering import *

import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
from scipy.signal import freqz    # Computes the frequency response of a digital filter

sampling_freq = 1000
cutoff_freq = 100

low_cutoff_freq = 200
high_cutoff_freq = 300

# Conventional training set
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "209", "215", "220", "230"]

# Conventional testing set
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]

import wfdb

wfdb.show_ann_labels()

import wfdb

for rec in DS1:
    annotations = wfdb.rdann("mitdb/" + rec, "atr")
    annotations.get_contained_labels()

sampfrom = 0
sampto = 2000
record = wfdb.rdrecord("mitdb/100", sampfrom=sampfrom, sampto=sampto)    # record = (signals, fields)
annotations = wfdb.rdann("mitdb/100", "atr", sampfrom=sampfrom, sampto=sampto)





rec_index = "119"
sampfrom = 0
sampto = 1000

record = wfdb.rdrecord("mitdb/" + rec_index, sampfrom=sampfrom, sampto=sampto)    # record = (signals, fields)
annotations = wfdb.rdann("mitdb/" + rec_index, "atr", sampfrom=sampfrom, sampto=sampto)



def autocorr (x, mode="full"):
    y = np.convolve(x, x, mode)
    return y[int(y.size/2) :] # / y[int(y.size/2)]    # Normalized



import wfdb
from statsmodels.tsa.stattools import levinson_durbin
from collections import OrderedDict

from filtering import butter_filter

def extract_features (record_path, length_qrs, length_stt, ar_order_qrs, ar_order_stt, sampfrom=0, sampto=-1, use_filter=True):

    print(record_path)
    qrs_stt_rr_list = list()
    sampto = 10000
    #print(sampto)

    if sampto < 0:
        raw_signal, _ = wfdb.rdsamp(record_path, channels=[0], sampfrom=sampfrom, sampto="end")
        annotations = wfdb.rdann(record_path, extension="atr", sampfrom=sampfrom, sampto=None)
    else:
        raw_signal, _ = wfdb.rdsamp(record_path, channels=[0], sampfrom=sampfrom, sampto=sampto)
        annotations = wfdb.rdann(record_path, extension="atr", sampfrom=sampfrom, sampto=sampto)

    raw_signal = raw_signal.reshape(-1)
    
    # Filtering
    if use_filter:
        filter_1 = butter_filter(raw_signal, filter_type="highpass", order=3, cutoff_freqs=[1], sampling_freq=annotations.fs)
        filter_2 = butter_filter(filter_1, filter_type="bandstop", order=3, cutoff_freqs=[58, 62], sampling_freq=annotations.fs)
        signal = butter_filter(filter_2, filter_type="lowpass", order=4, cutoff_freqs=[25], sampling_freq=annotations.fs)
    else:
        signal = raw_signal
    
    annotation2sample = list(zip(annotations.symbol, annotations.sample))

    for idx, annot in enumerate(annotation2sample):
        beat_type       = annot[0]    # "N", "V", ... etc.
        r_peak_pos      = annot[1]    # The R peak position
        pulse_start_pos = r_peak_pos - int(length_qrs / 2) + 1    # The sample postion of pulse start (start of QRS)

        # We treat only Normal, VEB, and SVEB signals
        print(beat_type)
        if beat_type == "N" or beat_type == "S" or beat_type == "V":
            qrs = signal[pulse_start_pos : pulse_start_pos + length_qrs]
            stt = signal[pulse_start_pos + length_qrs + 1 : pulse_start_pos + length_qrs + length_stt]
            #print(qrs.size)
            if qrs.size > 0:
                _, qrs_arcoeffs, _, _, _ = levinson_durbin(qrs, nlags=ar_order_qrs, isacov=False)
            else:
                qrs_arcoeffs = None
            #print(stt.size)  
            if stt.size > 0:
                #print(stt.shape)
                _, stt_arcoeffs, _, _, _ = levinson_durbin(stt, nlags=ar_order_stt, isacov=False)
            else:
                stt_arcoeffs = None

            pre_rr_length  = annotation2sample[idx][1] - annotation2sample[idx - 1][1] if idx > 0 else None
            post_rr_length = annotation2sample[idx + 1][1] - annotation2sample[idx][1] if idx + 1 < annotations.ann_len  else None
            _type = 1 if beat_type == "V" else 0

        
            beat_list = list()
            beat_list = [("record", record_path.rsplit(sep="/", maxsplit=1)[-1]), ("type", _type), 
                         ("pre-RR", pre_rr_length), ("post-RR", post_rr_length)
                        ]
            for idx, coeff in enumerate(qrs_arcoeffs):
                beat_list.append(("qrs_ar{}".format(idx), coeff))
            for idx, coeff in enumerate(stt_arcoeffs):
                beat_list.append(("stt_ar{}".format(idx), coeff))
            
            beat_dict = OrderedDict(beat_list)
            
            qrs_stt_rr_list.append(beat_dict)
    return qrs_stt_rr_list

def series2arCoeffs (series):
    if series.size > 0:
        return np.concatenate(series.tolist()).reshape(series.size, -1)
    else:
        return None


import numpy as np
import pandas as pd
import seaborn as sns


length_qrs = 40
length_stt = 100

lst = list()

for i in DS1:
    rec_index = i

    # Tweak the use_filter param
    lst.extend(
        extract_features("mitdb/" + rec_index, length_qrs, length_stt, ar_order_qrs=3, ar_order_stt=3, use_filter=True)
    )

df = pd.DataFrame(lst)
df.dropna(inplace=True)

y = df["type"].values

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
X = df[df.columns[2:]].values
X = scale(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

print("Number of Normals : {}".format(df[df["type"] == 0].shape[0]))
print("Number of VEBs    : {}".format(df[df["type"] == 1].shape[0]))

from sklearn.svm import SVC

clf = SVC(class_weight="balanced")
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score

def my_metrics(y_true, y_pred):
    print("Accuracy  : {}".format(accuracy_score(y_true, y_pred)))
    print("Precision : {}".format(precision_score(y_true, y_pred)))
    print("Recall    : {}".format(recall_score(y_true, y_pred)))

my_metrics(y_test, y_predict)

################################################
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(11,activation='relu',input_dim=8))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(X_train,y_train,epochs=10,batch_size = 32)

Y_pred_nn = model.predict(X_test)

rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded

score_nn = round(accuracy_score(Y_pred_nn,y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

from sklearn.metrics import confusion_matrix

matrix= confusion_matrix(y_test,Y_pred_nn)

sns.heatmap(matrix,annot = True, fmt = "d")
plt.show()
from sklearn.metrics import precision_score

print(y_test.shape)
Y_pred_nn=np.array(Y_pred_nn)
print(Y_pred_nn.shape)

precision = precision_score(y_test,Y_pred_nn,average='weighted')

print("Precision: ",precision)


from sklearn.metrics import recall_score

recall = recall_score(y_test,Y_pred_nn,average='weighted')

print("Recall is: ",recall)

print((2*precision*recall)/(precision+recall))


CM = pd.crosstab(y_test, Y_pred_nn)
print(CM)


###


X_train=np.array(X_train)
print(X_test.shape)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test=np.array(X_test)


X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



# Importing Tensorflow
import tensorflow as tf

# Initialising the RNN
regressor = tf.keras.models.Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(tf.keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(tf.keras.layers.Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
regressor.add(tf.keras.layers.Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
regressor.add(tf.keras.layers.Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(tf.keras.layers.LSTM(units = 50))
regressor.add(tf.keras.layers.Dropout(0.2))

# Adding the output layer
regressor.add(tf.keras.layers.Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
print(X_train.shape)

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)



Y_pred_rnn = regressor.predict(X_test)

rounded = [round(x[0]) for x in Y_pred_rnn]

Y_pred_rnn = rounded

score_rnn = round(accuracy_score(Y_pred_rnn,y_test)*100,2)


print("The accuracy score achieved using Neural Network is: "+str(score_rnn)+" %")

from sklearn.metrics import confusion_matrix

matrix= confusion_matrix(y_test,Y_pred_rnn )

print(matrix)
sns.heatmap(matrix,annot = True, fmt = "d")
plt.show()

from sklearn.metrics import precision_score

Y_pred_rnn=np.array(Y_pred_rnn)
precision = precision_score(y_test,Y_pred_rnn,average='weighted' )

print("Precision: ",precision)


from sklearn.metrics import recall_score

recall = recall_score(y_test,Y_pred_rnn ,average='weighted')

print("Recall is: ",recall)

print((2*precision*recall)/(precision+recall))


CM = pd.crosstab(y_test, Y_pred_rnn )
print(CM)


scores = [score_nn,score_rnn]
algorithms = ["Neural Network","recurrent neural network"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")

import matplotlib.pyplot as plt
import seaborn as sns
print(scores)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(algorithms,(scores))
plt.show()
