import os
import tensorflow as tf
import keras
import pickle
import sklearn
import numpy as np
from pickle import load
import keras.backend as K
from keras.layers import Layer, Embedding
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

f = open('C:/Users/sisod/Downloads/New folder/Project/Data/Pickles/Dataset.p', 'rb')
d_final3 = pickle.load(f)

f = open('C:/Users/sisod/Downloads/New folder/Project/Data/Pickles/embedding_matrix.p', 'rb')
embedding_matrix = pickle.load(f)

f = open('C:/Users/sisod/Downloads/New folder/Project/Data/Pickles/Audio_features.p', 'rb')
emb = pickle.load(f)

MAX_NB_WORDS = 40000
MAX_SEQUENCE_LENGTH = 128

text = d_final3['text']
# mohit -> to remove empty space
text = [t for t in text if t is not None and t.strip() != ""]

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token=True)
tokenizer.fit_on_texts(text)
data = np.zeros((len(text), MAX_SEQUENCE_LENGTH), dtype='int32')

labels1 = []
label_index1 = {}
for label in d_final3['Sarcasm']:
    labelid = len(label_index1)
    label_index1[label] = labelid
    labels1.append(label)

print(len(labels1))

labels2 = []
label_index2 = {}
for label in d_final3['Humour']:
    labelid = len(label_index1)
    label_index2[label] = labelid
    labels2.append(label)

print(len(labels2))

embedding = []
for i in range(len(emb)):
    embedding.append(emb[i])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels1 = np.asarray(labels1)
labels2 = np.asarray(labels2)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels1.shape)


f = open('C:/Users/sisod/Downloads/New folder/Project/Data/Pickles/d_features.p', 'rb')
data = pickle.load(f)

test_data = data[14000:15576]
test_label1 = labels1[14000:15576]
test_label2 = labels2[14000:15576]
test_embedding = embedding[14000:15576]
# mohit -> for printing these values
test_data.shape, test_label1.shape, test_label2.shape, len(test_embedding)

data = data[0:14000]
labels1 = labels1[0:14000]
labels2 = labels2[0:14000]
embedding = embedding[0:14000]
data.shape, labels1.shape, len(embedding)

EMBEDDING_DIM = 300
print(embedding_matrix.shape)
# mohit -> changed the first parameter to manual it was len(word_index) + 1
embedding_layer = Embedding(19572, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False, name='embedding',
                            mask_zero=True, input_shape=(6, 128))

VALIDATION_SPLIT = 0.2
indices = np.arange(data.shape[0])
data = data[indices]
labels1 = labels1[indices]
labels2 = labels2[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
print(labels1)
print(labels2)

train_x1 = data[:-nb_validation_samples]
train_x2 = embedding[:-nb_validation_samples]
train_y1 = labels1[:-nb_validation_samples]
train_y2 = labels2[:-nb_validation_samples]
dev_x1 = data[-nb_validation_samples:]
dev_x2 = embedding[-nb_validation_samples:]
dev_y1 = labels1[-nb_validation_samples:]
dev_y2 = labels2[-nb_validation_samples:]

train_x2 = np.array(train_x2)
dev_x2 = np.array(dev_x2)

train_x2 = train_x2.reshape((11200, 128, 1))
train_x2.shape

dev_x2 = dev_x2.reshape((2800, 128, 1))
dev_x2.shape

print(dev_y1)
print(dev_y2)

model = tf.keras.models.load_model('C:/Users/sisod/Downloads/New folder/Project/15k211mode3.tf')
print(model.metrics_names)
eval_results= model.evaluate([dev_x2,dev_x1],[dev_y2,dev_y1])
for i in eval_results:
    print(i)

yh_pred,ys_pred = model.predict([dev_x2,dev_x1], verbose=0)

for i in range(len(yh_pred)):
    if yh_pred[i]<0.5:
        yh_pred[i]=0
    else:
        yh_pred[i]=1
print(yh_pred)
        

# for i in ys_pred:
#     if i<0.5:
#         i=0
#     else:
#         i=1
# print(ys_pred)
        
# y_pred_classes = np.argmax(y_pred, axis = 1) 
# y_pred_classes = np.argmax(y_pred, axis=1)

# Convert validation observations to one hot vectors
# yh_true,ys_true = np.argmax([dev_y2,dev_y1], axis=1)

# compute the confusion matrix
classification_report = classification_report(dev_y2, yh_pred)

print(classification_report)
# print("Restored model, accuracy: {:5.2f}%".format(100*eval_results))
