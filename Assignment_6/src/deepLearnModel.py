#!/usr/bin/env python
"""
Specify file path of the csv file of Game of Thrones lines and name of the output model history plot. You can also specify size of the test data in percentage points. The default will be 0.25. Furthermore, the user can specify pooling method, optimizer and number of epochs to train over. The output will be a summary of the model architecture saved as both txt and png and a model history plot saved as png. These will all be saved in a folder called output in the parent directory of the working directory.

Parameters:
    input_file: str <filepath-of-csv-file>
    output_filename: str <name-of-png-file>
    test_size: float <size-of-test-data>
    optimizer: str <optimization-method>
    pooling: str <pooling-method>
    n_epochs: int <number-of-epochs>
Usage:
    deepLearnModel.py -f <filepath-of-csv-file> -o <name-of-png-file> -t <size-of-test-data> -opt <optimization-method> -p <pooling-method> -e <number-of-epochs>
Example:
    $ python3 deepLearnModel.py -f ../data/Game_of_Thrones_Script.csv -o deepLearn_model_history.png -t 0.25 -opt adam -p MaxPooling -e 10
    
## Task
- Train a convolutional neural network to classify lines from GoT characters into their respective seasons (1-8). 
- Output is in the form of two png files (model architecture and plot of model history) and a txt file of the model architecture. These outputs can be found in the path "../output". Model accuracy for training and testing data will also be printed in the terminal.
"""

# libraries
# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim, contextlib
import pandas as pd
import numpy as np
import gensim.downloader
from contextlib import redirect_stdout

# import my classifier utility functions - see the Github repo!
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import SGD, Adam
#from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

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
                default = "CNN_model_history.png", 
                help = "Name of output file")
ap.add_argument("-t", 
                "--test_size", 
                default = 0.25, 
                help = "Size of test data")
ap.add_argument("-opt", 
                "--optimizer", 
                default = 'adam', 
                help = "Optimizer: Choose between 'SGD' or 'adam'.")
ap.add_argument("-p", 
                "--pooling", 
                default = "MaxPooling", 
                help = "Pooling: Choose between 'MaxPooling' and 'AveragePooling'")
ap.add_argument("-e", 
                "--n_epochs", 
                default = 20, 
                help = "Number of epochs")

# parsing arguments
args = vars(ap.parse_args())




def main(args):
    # get variables from argparse
    filepath = args["input_file"]
    out_name = args["output_filename"]
    test_size = args["test_size"]
    opt = args["optimizer"]
    pool = args["pooling"]
    epochs = int(args["n_epochs"])
    
    # Create out directory if it doesn't exist in the data folder
    dirName = os.path.join("..", "output")
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        # print that it has been created
        print("\nDirectory " , dirName ,  " Created ")
    else:   
        # print that it exists
        print("\nDirectory " , dirName ,  " already exists")
        
    
    # make train and test data ready
    vocab_size, maxlen, X_train_pad, X_test_pad, trainY, testY = train_test_preprocessing(filepath = filepath, 
                                                                                          test_size = test_size)
    # train the model and return the model summary and the model history
    model, history = training_model(opt = opt,
                                    pool = pool,
                                    epochs = epochs,
                                    vocab_size = vocab_size, 
                                    maxlen = maxlen,
                                    X_train_pad = X_train_pad, 
                                    X_test_pad = X_test_pad, 
                                    trainY = trainY, 
                                    testY = testY)
    
    # evaluate the model
    evaluate_model(model = model, 
                   history = history,
                   epochs = epochs,
                   out_name = out_name,
                   X_train_pad = X_train_pad, 
                   X_test_pad = X_test_pad, 
                   trainY = trainY, 
                   testY = testY)
    
    print("\nYou have now successfully trained and evaluated a deep learning model on the Game of Thrones data. Have a nice day!")
        
    
    
    
    
    
def plot_history(H, epochs, out_name):
    """
    Utility function for plotting model history using matplotlib
    
    H: model history 
    epochs: number of epochs for which the model was trained
    out_name: filename for the output png
    """
    # name for saving output
    figure_path = os.path.join("..", "output", out_name)
    # Visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    

    
    
def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix




def train_test_preprocessing(filepath, test_size):
    '''
    Making the training and testing data from the filepath to csv.
    Finding max length of sentences.
    Tokenizing the data and padding to match max length.
    Transforming labels to binarized vectors.
    Returning vocabulary size, max length of sentences, padded training and test data and binarized y labels for training and testing.
    '''
    # load the data with pandas
    got = pd.read_csv(filepath)
    
    # get the values in each cell; returns a list
    sentences = got['Sentence'].values
    labels = got['Season'].values
    
    # train and test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                        labels, 
                                                        test_size=test_size, 
                                                        random_state=42)
    
    # finding max length of lines to use for finding padding maxlen
    # empty list
    k = []
    # for loop to append length of each line in the training data
    for i in X_train:
        # append to empty list
        k.append(len(i))
    # define maxlen
    maxlen = max(k) + 4
    
    # initialize tokenizer
    tokenizer = Tokenizer(num_words=5000)
    # fit to training data
    tokenizer.fit_on_texts(X_train)

    # tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)

    # overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    
    
    # pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks, 
                            padding='post', # sequences can be padded "pre" or "post"
                            maxlen=maxlen)
    # pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                           padding='post', 
                           maxlen=maxlen)
    
    # transform labels to binarized vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(y_train)
    testY = lb.fit_transform(y_test)
    
    return vocab_size, maxlen, X_train_pad, X_test_pad, trainY, testY





def training_model(opt, pool, epochs, vocab_size, maxlen, X_train_pad, X_test_pad, trainY, testY):
    '''
    Training the model. 
    Define model architecture and save as txt and png.
    Fitting the data to the model and returning the model and the history.
    '''
    
    # define regularizer
    l2 = L2(0.0001)
    # define embedding size we want to work with
    embedding_dim = 100
    
    # if user specifies pooling method as MaxPooling
    if pool == "MaxPooling":
        # initialize Sequential model
        model = Sequential()
        # add Embedding layer
        model.add(Embedding(input_dim=vocab_size,     # vocab size from Tokenizer()
                            output_dim=embedding_dim, # user defined embedding size
                            input_length=maxlen))     # maxlen of padded docs
        model.add(Conv1D(256, 5, # kernel size = 5
                        activation='relu',
                        kernel_regularizer=l2)) # kernel regularizer
        
        # DIFFERENCE BETWEEN THE TWO STATEMENTS: MaxPool layer
        model.add(GlobalMaxPool1D())
        # Add Dense layer; 128 neurons; ReLU activation
        model.add(Dense(128, 
                        activation='relu'))
        # Add prediction nodes; softmax activation function
        model.add(Dense(8, 
                        activation='softmax'))

        # compile model
        model.compile(loss='categorical_crossentropy', # categorical loss function (as the classification is more than binary)
                      optimizer=opt, # user specified optimization
                      metrics=['accuracy'])

        # print summary
        model.summary()
    
    # else if pooling method is average pooling
    elif pool == "AveragePooling":
        # initialize Sequential model
        model = Sequential()
        # add Embedding layer
        model.add(Embedding(input_dim=vocab_size,     # vocab size from Tokenizer()
                            output_dim=embedding_dim, # user defined embedding size
                            input_length=maxlen))     # maxlen of padded docs
        model.add(Conv1D(256, 5, # kernel size = 5
                        activation='relu',
                        kernel_regularizer=l2)) # kernel regularizer
        # DIFFERENCE BETWEEN THE TWO STATEMENTS: AveragePool layer
        model.add(GlobalAveragePooling1D())
        # Add Dense layer; 128 neurons; ReLU activation
        model.add(Dense(128, 
                        activation='relu'))
        # Add prediction node; softmax activation
        model.add(Dense(8, 
                        activation='softmax'))

        # compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        # print summary
        model.summary()
    
    # else print not a valid method
    else:
        print("Not a valid pooling method. Choose between 'MaxPooling' or 'AveragePooling'.")
    
    # name for saving model summary
    model_path = os.path.join("..", "output", "CNN_model_summary.txt")
    # Save model summary
    with open(model_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
            
    # name for saving plot
    plot_path = os.path.join("..", "output", "CNN_model_architecture.png")
    # Visualization of model
    plot_DL_model = plot_model(model,
                            to_file = plot_path,
                            show_shapes=True,
                            show_layer_names=True)
    
    print(f"\n[INFO] Model architecture is saved as txt in '{model_path}' and as png in '{plot_path}'.")
    
    
    
    # training the model
    print(f"\n[INFO] Training the model with opt = {opt} and pooling = {pool}")
    history = model.fit(X_train_pad, trainY,
                    epochs=epochs,
                    verbose=False,
                    validation_data=(X_test_pad, testY))
    
    return model, history





def evaluate_model(model, history, epochs, X_train_pad, X_test_pad, trainY, testY, out_name):
    '''
    Evaluate the model. 
    Print the model accuracy for the training and testing data. 
    Save the model history as png.
    '''
    
    # Evaluate the model
    print(f"\n[INFO] Evaluating the model...")
    loss, accuracy = model.evaluate(X_train_pad, trainY, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_pad, testY, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    
    # plot history
    print(f"Plotting the model history across {epochs} epochs")
    plot_history(history, epochs=epochs, out_name = out_name)
    
    
    
if __name__ == "__main__":
    main(args)