#now we will create the neurons of BOSS brain
import numpy as np
#nltk = natural language tool kit
import nltk
from nltk.stem.porter import PorterStemmer

# here we will make a object which will keep all the data coming,process it and thn gives output
Stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return Stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,words):
    sentence_word = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words),dtype=np.float32)
    
    for idx, w in enumerate(words):
        if w in sentence_word:
            bag[idx] = 1
        
    return bag
