# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:02:04 2018

@author: Manzoor Elahi
"""

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import random
from nmt_utils import *


num_samples = 900000  # Number of samples to train on.
data_path = 'data/US_patents_1976-Sep2016_1product_reactions_train.csv'


Tx = 18
dataset = []
input_characters = set()
target_characters = set()
lines = open(data_path).read().split('\n')
for line in lines[3: min(num_samples, len(lines) - 1)]:
    input_text, target_text, *c = line.split('\t')
    input_text,_ = input_text.split('>')
    input_text = input_text.split()
    target_text = target_text.split()
    #input_text = input_text.replace(" ", "")
    #target_text = target_text.replace(" ", "")
    if len(input_text)<=Tx and len(target_text)<len(input_text):
        ds = (input_text,target_text)
        dataset.append(ds)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

Ty = len(max(dataset[:][1], key=len))
input_characters = sorted(list(input_characters)) + ['<unk>', '<pad>']
target_characters = sorted(list(target_characters)) + ['<unk>', '<pad>']
reactants_vocab = {v:k for k,v in enumerate(input_characters)}
products_vocab = {v:k for k,v in enumerate(target_characters)}
inv_products_vocab = {v:k for k,v in products_vocab.items()}

X, Y, Xoh, Yoh = preprocess_data(dataset, reactants_vocab, products_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

index = 13
print("Reactants:", dataset[index][0])
print("Product:", dataset[index][1])
print()
print("Reactants after preprocessing (indices):", X[index])
print("Product after preprocessing (indices):", Y[index])
print()
print("Reactants after preprocessing (one-hot):", Xoh[index])
print("Product after preprocessing (one-hot):", Yoh[index])


repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes = 1)


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    s_prev = repeator(s_prev)
    concat = concatenator([a,s_prev])
    e = densor(concat)
    alphas = activator(e)
    context = dotor([alphas,a])
    
    return context


n_a = 256
n_s = 512
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(products_vocab), activation=softmax)


def model(Tx, Ty, n_a, n_s, reactants_vocab_size, products_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    X = Input(shape=(Tx, reactants_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0    
    outputs = []   
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)    
    for t in range(Ty):   
        context = one_step_attention(a, s)       
        s, _, c = post_activation_LSTM_cell(inputs = context, initial_state = [s,c])
        out = output_layer(s)        
        outputs.append(out)
    model = Model(inputs = [X, s0, c0], outputs = outputs)
    return model


model = model(Tx, Ty, n_a, n_s, len(reactants_vocab), len(products_vocab))
model.summary()


m = len(dataset)
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))


model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])


model.load_weights('weights256.h5')

for i in range(5):
    print(i)
    model.fit([Xoh, s0, c0], outputs, epochs=20, batch_size=64)
    model.save('model256.h5')
    model.save_weights('weights256.h5') 
    

output = []
pad = '<pad>'
start = 23
end = 33
prediction = model.predict([Xoh[start:start+end,:,:], s0[start:start+end,:], c0[start:start+end,:]])
count = 0
for i in range(10):
    p = np.argmax(np.array(prediction)[:,i,:], axis = 1)
    p = int_to_string(p,inv_products_vocab)
    o2 = []
    for x in p:
        if x != pad:
            o2.append(x)
    o2 = ''.join(o2)
    if o2 == ''.join(dataset[start+i][1]):
        count += 1
    output.append([''.join(dataset[start+i][0]),''.join(dataset[start+i][1]),o2,o2 == ''.join(dataset[start+i][1])])
    
print(count)

for elem in output:
    print(elem)


am = plot_attention_map(model, reactants_vocab, inv_products_vocab, text = dataset[99][0], n_s = 512, num = 6, Tx = 18, Ty = 18)
