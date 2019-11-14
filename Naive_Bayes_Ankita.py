# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:09:08 2019

@author: palan
"""
############ Import ############

import nltk
import pandas
import numpy
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import os
warnings.filterwarnings("ignore")

############ Check for the features ############
def word_feat(msg):
	words = word_tokenize(msg)
	features = {}
	for word in word_features:
		features[word] = (word in words)
	return features

############ Load data ############
dirpath = os.getcwd()
print("current directory is : " + dirpath)
filepath = " "
#If no filepath given as input
filepath = input("Enter file path for dataset(If the file is kept on run directory press Enter twice):")
if not filepath.strip():
	filepath = dirpath
#If no Filename given as input, Filename is fixed and provided

filename = input('Enter Training File Name : ')
if not filename.strip():
	filename = 'GEASTrain.txt'
stop_filename = input('Enter File Name for stop words: ')
if not stop_filename.strip():
	stop_filename = 'StopWords.txt'
filename = filepath+'\\'+filename

datafile = pd.read_csv(filename,names=['code'],encoding='unicode-escape')
datafile[['code','name_of_code']] = datafile["code"].str.split(" ", 1, expand=True)
datafile["name_of_code"] = datafile["name_of_code"].str.strip(" ")
#data = np.transpose(datafile.iloc[:,[0,1]].values)
#stop_filename = filepath+'\\'+stop_filename


with open(stop_filename, 'r', encoding='unicode-escape') as stop_filename:
	stop_file = [stop.rstrip('\n') for stop in stop_filename]




#######################################
#print(data,"\n")
classes = datafile["code"]
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
#txt_msg = data[1]
#txt_msg = " ".join(txt_msg)

############ Data Cleaning ############
cleaned = datafile["name_of_code"].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','email')
cleaned = cleaned.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','url')
cleaned = cleaned.str.replace(r'£|\$', 'currency')
cleaned = cleaned.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonen')
cleaned = cleaned.str.replace(r'\d+(\.\d+)?', 'num')
cleaned = cleaned.str.replace(r'[^\w\d\s]', ' ')
cleaned = cleaned.str.replace(r'\s+', ' ')
cleaned = cleaned.str.replace(r'^\s+|\s+?$', '')
cleaned = cleaned.str.lower()

############ Add stop words ############
#stop_words = open("english")
stop_words = set(stop_file)
#cleaned = cleaned.apply(lambda l: ' '.join(term for term in l.split() if term not in stop_words))
cleaned = set(cleaned) - stop_words
cleaned = list(cleaned)
############ Stem word ############
stm = nltk.PorterStemmer()
cleaned = datafile["name_of_code"].apply(lambda l: ' '.join(stm.stem(term) for term in l.split()))

############ Add and check ############
all_words = []
for msg in cleaned:
	words = word_tokenize(msg)
	for w in words:
		all_words.append(w)

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:1000]
print("common features",word_features,"\n")
np.random.shuffle(Y)
messages = list(zip(cleaned, Y))

seed = 10
np.random.seed = seed

############ Shuffle data ############
np.random.shuffle(messages)

############ function call ############
fset = [(word_feat(text), label) for (text, label) in messages]

############ Data split ############
training, testing = model_selection.train_test_split(fset, test_size = 0.25, random_state=seed)

############ model training ############
model = SklearnClassifier(MultinomialNB())
model.train(training)

############ accuracy ############
accuracy = nltk.classify.accuracy(model, testing)*100
accuracy = float(accuracy)
print("Accuracy of our model is: {}".format(accuracy))
