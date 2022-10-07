# PARTE 1

# permite el procesamiento del lenguaje natural
import nltk
nltk.download("punkt")
# minimizar las palabras
from nltk.stem.lancaster import LancasterStemmer
# instanciar el minimizador
stemmer = LancasterStemmer()

# permite trabajar con arreglos
import numpy

# trabajar con redes neuronales
import tflearn
import tensorflow
from tensorflow.python.framework import ops

# permite manejar base de datos (json)
import json

# permite crear numeros aleatorios
import random

# permite guardar todo el entrenamiento en un solo archivo
import pickle

import requests
import os
import matplotlib as mltp

# permite cargar datos
import dload

import itertools

print("Primera parte correcta!")

# descargar datos de github
dload.git_clone('https://github.com/boomcrash/data_bot.git')

# crear variable para datos y hacerlos escribibles
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+'/data_bot/data_bot-main/data.json', 'r') as file:
  database = json.load(file)

words = []
all_words = []
tags = []
aux = []
auxA = []
auxB = []
training = []
exit = []

try:
  # crear carpeta y guardar variables
  with open('Entrenamiento/brain.pickle', 'rb') as pickleBrain:
    all_words,tags,training,exit = pickle.load(pickleBrain)
    print("Ya no hice el except, porque tienes todo en brain.pickle")

except:
  for intent in database['intents']:
    for pattern in intent["patterns"]:
      auxWords = nltk.word_tokenize(pattern)
      # guardar las palabras
      auxA.append(auxWords)
      auxB.append(auxWords)
      # guardar tags
      aux.append(intent["tag"])
  
  # simbolos a ignorar
  ignore_words = ['?', "|", ",", ".", "'", '"', "$", "-", "_", "%", "/", '(', ')', '=', '*', '#']
      
  for w in auxB:
    if w not in ignore_words:
      words.append(w)
  
  words = sorted(set(list(itertools.chain.from_iterable(words))))
  # print(words)

  tags = sorted(set(aux))
  # print(tags)

  # convertir a minuscula
  all_words = [stemmer.stem(w.lower()) for w in words]
  # print(len(all_words))

  all_words = sorted(list(set(all_words)))
  tags = sorted(tags)

  # crear salida falsa
  null_exit = [0 for _ in range(len(tags))]
  print(null_exit)

  for i, document in enumerate(auxA):
    bucket = []
    # minuscula y quitar signos
    auxWords = [stemmer.stem(w.lower()) for w in document if w != "?"]
  
    for w in all_words:
      if w in auxWords:
        bucket.append(1)
      else:
        bucket.append(0)

    exit_row = null_exit[:]
    exit_row[tags.index(aux[i])] = 1
    training.append(bucket)
    exit.append(exit_row)
  
  # print(training)
  # print(exit)
  training = numpy.array(training)
  # print(training)
  exit = numpy.array(exit)

  # crear archivo pickle
  with open('Entrenamiento/brain.pickle', 'wb') as pickleBrain:
    pickle.dump((all_words, tags, training, exit), pickleBrain)

  
# Parte 3
tensorflow.compat.v1.reset_default_graph()
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)
# crear red neuronal
net = tflearn.input_data(shape=[None, len(training[0])])
# redes intermedias
net = tflearn.fully_connected(net, 100, activation='Relu')
net = tflearn.fully_connected(net, 50)
net = tflearn.dropout(net, 0.5)
# neurona de salida
net = tflearn.fully_connected(net, len(exit[0]), activation='softmax')

# regresion
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')
model = tflearn.DNN(net)

if os.path.isfile(dir_path+"/Entrenamiento/model.tflearn.index"):
  model.load(dir_path+"/Entrenamiento/model.tflearn")
  print("Ya existen datos entrenados")
else:
  model.fit(training, exit, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=1000)
  model.save("Entrenamiento/model.tflearn")

# parte 4
def response(texto):
  if texto == "duerme":
    print("Ha sido un gusto, vuelve pronto")
    return False
  else:
    bucket = [0 for _ in range(len(all_words))]
    processed_sentence = nltk.word_tokenize(texto)
    processed_sentence = [stemmer.stem(palabra.lower()) for palabra in processed_sentence]
    for individual_word in processed_sentence:
      for i, palabra in enumerate(all_words):
        if palabra == individual_word:
          bucket[i]=1
    results = model.predict([numpy.array(bucket)])
    index_results = numpy.argmax(results)
    max = results[0][index_results]

    target = tags[index_results]

    for tagAux in database["intents"]:
      if tagAux['tag'] == target:
        answer = tagAux['responses']
        answer = random.choice(answer)

    print(answer)
    return True

print("Habla conmigo: ")
bool = True
while bool == True:
  texto = input()
  bool = response(texto)
