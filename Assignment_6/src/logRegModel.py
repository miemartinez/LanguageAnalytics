#!/usr/bin/env python
"""
Specify file path of the csv file of Game of Thrones lines and name of the output cross validation graphs. You can also specify size of the test data in percentage points. The default will be 0.25. The output will be a classification matrix saved as png, a classification report saved as csv (this will also be printed in the terminal) and a cross validation graph saved as png. These will all be saved in a folder called output in the parent directory of the working directory.
Parameters:
    input_file: str <filepath-of-csv-file>
    output_filename: str <name-of-png-file>
    test_size: float <size-of-test-data>
Usage:
    logRegModel.py -f <filepath-of-csv-file> -o <name-of-png-file> -t <size-of-test-data>
Example:
    $ python3 logRegModel.py -f ../data/Game_of_Thrones_Script.csv -o LogReg_cross_validation.png -t 0.25
    
## Task
- Train a multiple logistic regression model to classify lines from GoT characters into their respective seasons (1-8). 
- Output is in the form of two png files (classification matrix and cross validation graphs) and a csv file of the classification report which will also be printed in the terminal. All outputs can be found in the path "../output".
"""

# libraries
# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim
import pandas as pd
import numpy as np
#import gensim.downloader

# import my classifier utility functions - see the Github repo!
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
#from sklearn.preprocessing import LabelBinarizer

# matplotlib
import matplotlib.pyplot as plt


import argparse

# argparse 
ap = argparse.ArgumentParser()
# adding argument
ap.add_argument("-f", 
                "--input_file", 
                required = True, 
                help= "Path to the Game of Thrones csv-file")
ap.add_argument("-o", 
                "--output_filename", 
                default = "logReg_model_history.png", 
                help = "Name of output file")
ap.add_argument("-t", 
                "--test_size", 
                default = 0.25, 
                help = "Size of test data")

# parsing arguments
args = vars(ap.parse_args())


def main(args):
    # get variables from argparse
    filepath = args["input_file"]
    out_name = args["output_filename"]
    test_size = args["test_size"]
    
    # Create out directory if it doesn't exist in the data folder
    dirName = os.path.join("..", "output")
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        # print that it has been created
        print("\nDirectory " , dirName ,  " Created ")
    else:   
        # print that it exists
        print("\nDirectory " , dirName ,  " already exists")
    
    # Start message to user
    print("\n[INFO] Initializing the construction of a logistic regression model...")
    sentences, labels = logRegModel(filepath = filepath, test_size = test_size)
    print("\n[INFO] Running cross validation on the logistic regression model...")
    cross_validation(sentences = sentences, labels = labels, out_name = out_name)
    
    
    # end message
    print("\nYou have now successfully trained and evaluated a logistic regression model on the Game of Thrones data. Have a nice day!")
    
def logRegModel(filepath, test_size):
    # load the data with pandas
    got = pd.read_csv(filepath)
    
    # get the values in each cell; returns a list
    sentences = got['Sentence'].values
    labels = got['Season'].values
    
    # train and test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                    labels, 
                                                    test_size=test_size, # define test size
                                                    random_state=42) # random state for reproducibility

    # initialize count vectorizer
    vectorizer = CountVectorizer()

    # First we do it for our training data...
    X_train_feats = vectorizer.fit_transform(X_train)
    #... then we do it for our test data
    X_test_feats = vectorizer.transform(X_test)
    # We can also create a list of the feature names. 
    feature_names = vectorizer.get_feature_names()
    
    print("\n[INFO] Training the model...")
    # fitting the data to the model
    classifier = LogisticRegression(random_state=42, max_iter = 1000).fit(X_train_feats, y_train)
    # predicting on the test data
    y_pred = classifier.predict(X_test_feats)
    
    # define path for classification report
    path1 = os.path.join("..", "output", "classification_report_logRegModel.csv")
    
    # making the classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    # print to terminal
    print(classifier_metrics)
    
    # making the classification report as dict
    classifier_metrics_dict = metrics.classification_report(y_test, y_pred, output_dict=True)
    # transpose and make into a dataframe
    classification_metrics_df = pd.DataFrame(classifier_metrics_dict).transpose()
    # saving as csv
    classification_metrics_df.to_csv(path1)
    # print that the csv file has been saved
    print(f"\n[INFO] Classification metrics are saved as {path1}")
    
    
    
    # specify path
    path2 = os.path.join("..", "output", "classification_matrix_logRegModel.png")
    # make classification matrix
    clf.plot_cm(y_test, y_pred, normalized=True)
    # save as png
    plt.savefig(path2)
    # print that the png file has been saved
    print(f"\n[INFO] Classification matrix are saved as {path2}")
    
    
    return sentences, labels


    
def cross_validation(sentences, labels, out_name):
    # initialize count vectorizer
    vectorizer = CountVectorizer()
    
    # Vectorize full dataset
    X_vect = vectorizer.fit_transform(sentences)

    # initialise cross-validation method
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    
    # define output path
    path3 = os.path.join("..", "output", out_name)
    # run on data
    model = LogisticRegression(random_state=42, max_iter = 1000)
    clf.plot_learning_curve(model, title, X_vect, labels, cv=cv, n_jobs=4)
    # save as png
    plt.savefig(path3)
    
    # print that the png file has been saved
    print(f"\n[INFO] Cross validation plots are saved as {path3}")

if __name__ == "__main__":
    main(args)

