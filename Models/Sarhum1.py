import os
import re
import h5py
import json
import scipy
import pickle
import random
import string
import subprocess
import collections
import keras.utils
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from scipy import spatial
import keras.backend as K
from keras import utils as np_utils
from attention import AttentionLayer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import backend, initializers, regularizers, constraints, optimizers, layers
from keras.backend import clear_session
# from keras_self_attention import SeqSelfAttention

# def install(name):
#     subprocess.call(['pip', 'install', name])
# install('keras_self_attention')
# def install1(name):
#     subprocess.call(['unzip', name])
# install1('embedding_matrix.zip')
# def install2(name):
#     subprocess.call(['unzip', name])
# install2('/content/Sarcasm-master/Data/Pickles/d_features.zip')

from tqdm.auto import tqdm
tqdm.pandas()

from PIL import Image
from pickle import load
from gensim import models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from nltk.translate.bleu_score import corpus_bleu
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, Sequential, model_from_json, load_model
from keras.layers import Layer, Input, Dense, LSTM, Embedding, Dropout, add, TimeDistributed, Bidirectional, Lambda, GRU, Reshape, Flatten, concatenate, multiply, Lambda, Convolution1D
# from keras.layers.recurrent import LSTM

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

max_sentences = 10
max_words = 128
# was 300 but changed to 10
word_encoding_dim = 10
sentence_encoding_dim = 128

#SeqSelfAttention:
from keras.backend import manual_variable_initialization
manual_variable_initialization(True)


class SeqSelfAttention(keras.layers.Layer):

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):
        """Layer initialization.

        For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf

        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param attention_width: The width of local attention.
        :param attention_type: 'additive' or 'multiplicative'.
        :param return_attention: Whether to return the attention weights for visualization.
        :param history_only: Only use historical pieces of data.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                                  in additive mode.
        :param use_attention_bias: Whether to use bias while calculating the weights of attention.
        :param attention_activation: The activation used for calculating the weights of attention.
        :param attention_regularizer_weight: The weights of attention regularizer.
        :param kwargs: Parameters for parent class.
        """
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError(
                'No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e -= 10000.0 * (1.0 - K.cast(lower <= indices, K.floatx())
                            * K.cast(indices < upper, K.floatx()))
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            e -= 10000.0 * ((1.0 - mask) *
                            (1.0 - K.permute_dimensions(mask, (0, 2, 1))))

        # a_{t} = \text{softmax}(e_t)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        a = e / K.sum(e, axis=-1, keepdims=True)

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba,
                          (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa),
                          (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa),
                        K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}


with keras.utils.custom_object_scope({'SeqSelfAttention': SeqSelfAttention}):
    def build_word_encoder(max_words, embedding_matrix, encoding_dim=word_encoding_dim):
        vocabulary_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]
        sentence_input = Input(shape=(max_words,), dtype='int32')
        embedded_sentences = embedding_layer(sentence_input)
        word_embed, sample = SeqSelfAttention(attention_width=3, attention_activation='sigmoid', name='attention',
                                        return_attention=True)(embedded_sentences)
        encoded_sentences = Bidirectional(GRU(int(encoding_dim / 2), return_sequences=True))(word_embed)
        return Model(inputs=[sentence_input], outputs=[encoded_sentences], name='word_encoder')


    def build_sentence_encoder(self, max_sentences, summary_dim, encoding_dim=sentence_encoding_dim):
        text_input = Input(shape=(max_sentences, summary_dim))
        encoded_sentences = Bidirectional(GRU(int(encoding_dim / 2), return_sequences=True))(text_input)
        return Model(inputs=[text_input], outputs=[encoded_sentences], name='sentence_encoder')


    in_tensor = Input(shape=(max_sentences, max_words))

    word_encoder = build_word_encoder(max_words, embedding_matrix, word_encoding_dim)
    word_rep = TimeDistributed(word_encoder, name='word_encoder')(in_tensor)
    ####################################################################################################
    sentence_rep = TimeDistributed(Bidirectional(LSTM(128)))(word_rep)
    sentence_rep1 = TimeDistributed(SeqSelfAttention(attention_width=3, attention_activation='sigmoid', name='attention'))(
        word_rep)
    doc_rep = build_sentence_encoder(max_sentences, word_encoding_dim, sentence_encoding_dim)(sentence_rep)
    doc_summary = AttentionLayer(name='sentence_attention')(doc_rep)
    ####################################################################################################
    layer2 = Bidirectional(LSTM(128, return_sequences=True))(doc_rep)
    '''encodings=TimeDistributed(LSTM(128))(layer2)'''
    print(sentence_rep.shape)
    word_att, sample = SeqSelfAttention(attention_width=3, attention_activation='sigmoid', name='attention',
                                  return_attention=True)(layer2)
    fe_inputs = Input(shape=(128, 1))

    # mohit -> in dono mtlb
    a = Convolution1D(128, kernel_size=128, padding='same')(fe_inputs)
    b = Convolution1D(128, kernel_size=128, padding='same')(a)
    # mohit -> bilsmt karna hai
    ####################################################################################################
    # fe_layer1, ha, cella = Bidirectional(LSTM(128, return_sequences=True, return_state=True, dropout=0.4))(b)
    # att_audio = AttentionLayer(name='audio_attention')(fe_layer1)
    # Ha = concatenate([ha, att_audio])
    # ####################################################################################################
    # seq_layer2, ht, cell = Bidirectional(LSTM(128, return_sequences=True, return_state=True, dropout=0.3))(word_att)
    # att_out = AttentionLayer(name='sentence_attention')(seq_layer2)
    # # print(att_out.shape)
    # Ht = concatenate([ht, att_out])
    # HAT = concatenate([ht, ha])
    # HAT_T = Dense(units=Ht.shape[-1], activation=K.relu)(HAT)
    # HT = multiply([Ht, HAT_T])
    # HA = multiply([Ha, HAT_T])
    # all_input = concatenate([HAT, HA, HT])


    # Define the Bidirectional LSTM layers
    bi_lstm_audio = Bidirectional(LSTM(128, return_sequences=True, return_state=True, dropout=0.4))
    bi_lstm_text = Bidirectional(LSTM(128, return_sequences=True, return_state=True, dropout=0.3))

    # Apply the Bidirectional LSTM layers to your inputs
    bi_layer1, forward_h_audio, forward_c_audio, backward_h_audio, backward_c_audio = bi_lstm_audio(b)
    bi_layer2, forward_h_text, forward_c_text, backward_h_text, backward_c_text = bi_lstm_text(word_att)

    # Define the AttentionLayer (assuming you've already implemented it)
    att_audio = AttentionLayer(name='audio_attention')(bi_layer1)
    att_out = AttentionLayer(name='sentence_attention')(bi_layer2)

    # Concatenate the forward and backward states and attention outputs
    Ha = concatenate([forward_h_audio, backward_h_audio, att_audio])
    Ht = concatenate([forward_h_text, backward_h_text, att_out])

    # Define HAT_T and compute HT and HA as before
    HAT = concatenate([Ha, Ht])
    HAT_T = Dense(units=HAT.shape[-1], activation=K.relu)(HAT)
    Ht = Dense(1024)(Ht)
    Ha = Dense(1024)(Ha)
    HT = multiply([Ht, HAT_T])
    HA = multiply([Ha, HAT_T])

    # Concatenate all inputs
    all_input = concatenate([HAT, HA, HT])

    decoder_layer1 = Dense(128, activation='relu')(all_input)
    decoder_layer2 = Dense(128, activation='relu')(all_input)
    outputs1 = Dense(1, activation='sigmoid')(decoder_layer1)
    outputs2 = Dense(1, activation='sigmoid')(decoder_layer2)
    model_sh2 = Model(inputs=[fe_inputs, in_tensor], outputs=[outputs1, outputs2])
    model_sh2.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    model_sh2.get_metrics_result
    # print("evaluation_metrics:")
    # for i in metrics:
    #     print(i)
    model_sh2.summary()

    model_encoding = Model(inputs=[in_tensor], outputs=[att_out])

    train_x1.shape

    model_name = '15khumour+sarasmmodel2.tf'
    n_epochs = 15
    batch_size = 32
    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    print(train_x1.shape)
    history = model_sh2.fit([train_x2, train_x1], [train_y2, train_y1], epochs=n_epochs, batch_size=batch_size,
                            validation_data=([dev_x2, dev_x1], [dev_y2, dev_y1]), callbacks=[checkpoint])
    model_sh2.save('15k211mode3.tf')
