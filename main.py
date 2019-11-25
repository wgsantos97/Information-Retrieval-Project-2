# IMPORTS
import os
import json
import nltk
import sklearn
import numpy as np
from os import path
from random import shuffle
from datetime import datetime
# nltk
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# sklearn
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# CLASSES
class Review:
    def __init__(self, js):
        self.book_id = js["book_id"]
        self.title = js["title"]
        self.rating = int(js["rating"])
        self.sentiment = int(js["sentiment"])
        self.review_text = js["review_text"]
        self.processed_text = self.review_text

    def __repr__(self):
        return self.title

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
            r.processed_text = self.processText(r.processed_text)
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

class Base_Algorithm:
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
                print("WARNING (108:17): Infinite loop detected. Breaking.")
                break
            shuffle(temp)
            self.training_data.append(temp.pop())
            idx = idx + 1
        self.test_data.extend(temp)

    def process(self):
        text = [' '.join(t.processed_text) for t in self.training_data]
        self.vect.fit(text)
        X_training = self.vect.transform(text)
        Y_training = [t.sentiment for t in self.training_data]

        test = [' '.join(t.processed_text) for t in self.test_data]
        test = self.vect.transform(test)
        expected = [t.sentiment for t in self.test_data]
        print("X Length: " + str(X_training.shape))
        return X_training,Y_training,test,expected
    
    def learn(self):
        raise Exception('ERROR (91:9) - the abstract method \'learn\' is being called from parent class.')
    
    def writeToFile(self):
        raise Exception('ERROR (94:9) - the abstract method \'writeToFile\' is being called from parent class.')

class Naive_Bayes(Base_Algorithm):
    def __init__(self,l,f):
        super().__init__(l,f)
    
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


# MAIN
def main():
    f = open("Data/graphic_novel_final.json",'r')
    project2 = Project2(f,10000)
    f.close()
    nb = Naive_Bayes(project2.reviewList,"test.md")

main()