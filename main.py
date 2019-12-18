# IMPORTS
import os
import json
import nltk
import fastai
import sklearn
import numpy as np
import pandas as pd
from os import path
from random import shuffle
from datetime import datetime
# from fastai
from fastai.text import *
from fastai.callbacks import *
# from nltk
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from sklearn
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# CLASSES
class Review:
    def __init__(self, js):
        self.book_id = js["book_id"]
        self.title = js["title"]
        self.rating = int(js["rating"])
        self.sentiment = int(js["sentiment"])
        self.review_text = js["review_text"]
        self.processed_tokens = self.review_text

    def __repr__(self):
        return self.title

    def to_dict(self):
        return {
            "book_id" : self.book_id,
            "title" : self.title,
            "rating" : self.rating,
            "sentiment" : self.sentiment,
            "review_text" : self.review_text,
            "processed_tokens" : self.processed_tokens,
        }

class Project2:
    def __init__(self, file, cap = 50000):
        print("Project 2 Initialized.\nProcessing " + str(cap) + " pieces of data...")
        self.population = cap
        self.reviewList = list()

        idx = 0
        print("Processing Text...")
        positives = 0
        negatives = 0
        for line in file:
            d = json.loads(line)

            sent = int(d["sentiment"])
            if(positives == cap/2 and negatives == cap/2): 
                print("Positives: " + str(positives))
                print("Negatives: " + str(negatives))
                break
            if(sent == 0): 
                if(negatives == cap/2):
                    continue
                negatives = negatives + 1
            else:
                if(positives == cap/2):
                    continue
                positives = positives + 1
            
            r = Review(d)
            r.processed_tokens = self.processText(r.processed_tokens)
            r.processed_tokens = " ".join(r.processed_tokens)
            self.reviewList.append(r)

            idx = idx + 1
            print("Progress: [IDX #" + str(idx) + "] - " + str(round(idx/cap*100, 5)) + "%")
    
    ### NLP_Processing
    def fixTag(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    # Output a list of cleaned tokens
    def processText(self, text):
        all_tokens = word_tokenize(text)  # convert to tokens
        all_pos_tags = pos_tag(all_tokens)  # tag tokens

        # Convert to Lower Case
        lower_tokens = [t.lower() for (t, pos) in all_pos_tags]

        # Remove Stopwords
        stoplist = stopwords.words('english')
        stoplist.extend([">", "<", ")", "(", "``", "''",".", "'", ";", "'s", ",", "n't",":","-","!","?","...","'ve","'m","'re","'ll","'d","--"])
        stoplist_tokens = [t for t in lower_tokens if t not in stoplist]
        stoplist_tokens = " ".join(stoplist_tokens)

        # Stem the words
        lemmatizer = WordNetLemmatizer()
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(stoplist_tokens))
        wn_tagged = map(lambda x: (x[0], self.fixTag(x[1])), nltk_tagged)
        result = []
        for word, tag in wn_tagged:
            if tag is None:
                result.append(word)
            else:
                result.append(lemmatizer.lemmatize(word, tag))
        
        return result
    ### NLP_Processing END

class Naive_Bayes:
    def __init__(self,l,f):
        self.filename = f
        self.reviewList = l
        self.test_data = list()
        self.training_data = list()
        self.vect = TfidfVectorizer()
        self.population = len(self.reviewList)
        self.splitData(self.population/2) # DEFAULT training data set size set to half the population
        self.learn()
    
    def splitData(self,n):
        temp = list()
        temp.extend(self.reviewList)
        idx=0
        while(len(self.training_data)<n):
            if(idx>self.population):
                print("WARNING (133:17): Infinite loop detected. Breaking.")
                break
            shuffle(temp)
            self.training_data.append(temp.pop())
            idx = idx + 1
        self.test_data.extend(temp)

    def process(self):
        text = [' '.join(t.processed_tokens) for t in self.training_data]
        self.vect.fit(text)
        X_training = self.vect.transform(text)
        Y_training = [t.sentiment for t in self.training_data]

        test = [' '.join(t.processed_tokens) for t in self.test_data]
        test = self.vect.transform(test)
        expected = [t.sentiment for t in self.test_data]
        print("X Length: " + str(X_training.shape))
        return X_training,Y_training,test,expected
    
    def learn(self):
        # training
        clf = MultinomialNB()
        X,Y,test,expected = self.process()
        clf.fit(X,Y)
        # testing
        pred = clf.predict(test)
        print(confusion_matrix(pred,expected))
        print("\nRESULTS\nAccuracy: " + str(accuracy_score(pred,expected)*100) + "%")
    
    def writeToFile(self):
        print("Naive Bayes is writing to file: " + self.filename)

class ULMFit:
    def __init__(self,l,f):
        self.filename = f
        self.reviewList = l
        self.test_data = list()
        self.training_data = list()
        self.vect = TfidfVectorizer()
        self.population = len(self.reviewList)

        df = pd.DataFrame.from_records([review.to_dict() for review in l])
        self.df_train, self.df_test = train_test_split(df.iloc[:,[3,4]], test_size = 0.3, random_state = 12)
        self.learn()

    def learn(self):
        data_lm = TextLMDataBunch.from_df(train_df = self.df_train, valid_df = self.df_test, min_freq = 1, path = "")
        data_clas = TextClasDataBunch.from_df(path = "", train_df = self.df_train, valid_df = self.df_test, vocab=data_lm.train_ds.vocab, bs=32)

        learn = language_model_learner(data_lm, arch = AWD_LSTM, pretrained = True, drop_mult=0.7)
        learn.fit_one_cycle(1, 1e-2)   # one epoch

        learn.unfreeze()
        learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)
        learn.fit_one_cycle(1, 1e-3)

        learn.freeze_to(-2)
        learn.fit_one_cycle(1, slice(5e-3/2, 5e-3))

        learn.freeze_to(-3)
        learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))

        learn.unfreeze()
        learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))

        preds, targets = learn.get_preds()
        predictions = np.argmax(preds, axis=1)
        print("Preds Size: " + str(len(preds)))
        print(preds[:10])
        print("Prediction Size: " + str(len(predictions)))
        print(predictions[:10])
        print("Targets Size: " + str(len(targets)))
        print(targets[:10])

        print("RESULTS SUMMARY")
        print("Accuracy Score: ")
        print(accuracy_score(predictions,targets))
        print("\nConfusion Matrix (Diagonal): ")
        print(confusion_matrix(predictions,targets).diagonal())
        print("\nConfusion Matrix (Sum): ")
        print(confusion_matrix(predictions,targets).sum(axis=1))

# MAIN
def main():
    f = open("Data/graphic_novel_final.json",'r')
    project2 = Project2(f,1000)
    f.close()
    # alg1 = Naive_Bayes(project2.reviewList,"test.md")
    alg2 = ULMFit(project2.reviewList,"test.md")

main()