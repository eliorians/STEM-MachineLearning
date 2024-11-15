# dataset used: https://www.kaggle.com/datasets/shivamb/spam-url-prediction

#imports here
import pandas as pd
import time 
from sklearn.preprocessing import LabelEncoder
#model training imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#evaluation imports
from sklearn.metrics import accuracy_score, confusion_matrix
#plotting imports
import seaborn as sns
from matplotlib import pyplot as plt

def main():
    #Using Carson's mac relative path
    #TODO add our own test urls
    data = pd.read_csv("SpamEmailClassification/url_spam_classification.csv")
    print(data)

    #TODO better notes on how this works
    # Convert URLs to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    x = vectorizer.fit_transform(data['url'])   #use as x_train
    #print(x)

    # Encode labels (spam or not spam)
    encoder = LabelEncoder()
    y = encoder.fit_transform(data['is_spam'])  #use as y_train
    #print(y)

    #TODO use our own urls as test set    
    #split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2) 

    #choose a model (start with logistic regression)
    #TODO mess with more models
    model = LogisticRegression(max_iter= 500, n_jobs= -1 )

    #fit the model
    model.fit(x_train, y_train)

    #generate predictions
    y_predictions = model.predict(x_test)

    #evaluate predictions
    accuracy = accuracy_score(y_test, y_predictions)
    accuracyPercent = str(accuracy * 100)
    print(f"accuracy for {model}: {accuracyPercent}%")

    #TODO 
    #Plot things
    #Add out own csv of url's and train on their data, test on ours (can test one at a time)
    #Look at the same problem with other models (Pytorch, tensor flow) if time allows
    

#Running main
if __name__ == "__main__":
    main()