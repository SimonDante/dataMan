import os
import sys
import numpy as np
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

BASEDIR = ''
EMBEDDING = os.path.join(BASEDIR , 'embedding')
TEXT_DATA_DIR = os.path.join(BASEDIR , 'dataIn/tweetLabeled/Earthquake')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

embeddings_index = {}
with open(os.path.join(EMBEDDING, 'crisis_embeddings.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

text = np.array([])
tid = np.array([])
labels = np.array([])
ttext = np.array([])
ttid = np.array([])
tlabels = np.array([])
with open(os.path.join(TEXT_DATA_DIR ,'tweet_obtain_mix_final.csv') , 'r' , encoding = 'utf-8') as csvFile:
        csvReader = csv.DictReader(csvFile)
        fieldname = csvReader.fieldnames
        for row in csvReader:
            tid = np.append(tid , row[fieldname[0]])
            text = np.append(text , row[fieldname[1]])
            labels = np.append(labels , row[fieldname[2]])
            
with open(os.path.join(TEXT_DATA_DIR , '2015_nepal_eq_cf_labels_final.csv') , 'r' , encoding = 'utf-8') as testFile:
    csvReader = csv.DictReader(testFile)
    fieldname = csvReader.fieldnames
    for row in csvReader:
        ttid = np.append(ttid , row[fieldname[0]])
        ttext = np.append(ttext , row[fieldname[1]])
        tlabels = np.append(tlabels , row[fieldname[2]])
            
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

testTokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
testTokenizer.fit_on_texts(ttext)
tsequences = testTokenizer.texts_to_sequences(ttext)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
tdata = pad_sequences(tsequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
tlabels = to_categorical(np.asarray(tlabels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data
y_train = labels
x_val = tdata
y_val = tlabels

print('Preparing embedding matrix.')

num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

print('Training model.')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 2, activation='relu')(embedded_sequences)
x = MaxPooling1D(2)(x)
x = Conv1D(128, 2, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(128, 2, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=20,
          epochs=10,
          validation_data=(x_val, y_val))