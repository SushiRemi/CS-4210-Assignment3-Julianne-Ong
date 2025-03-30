#-------------------------------------------------------------------------
# AUTHOR: Julianne Ong
# FILENAME: perceptron.py
# SPECIFICATION: Creates and trains a single layer perceptron and a multi-layer perceptron to classify optically recgonized handwritten digits.
# FOR: CS 4210- Assignment #3
# TIME SPENT: Start - 3/29/2025 6:20 PM ||||| End - 3/29/2025 7:00 PM ||||| Total Time: 40 Minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

#n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0] ## Learning rate
n = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001] ## Learning rate flipped (flipped so it shows the progression of accuracy better)
r = [True, False] ## Shuffle
a = [0, 1] ## 0 for single layer perceptron algorithm, 1 for multi layer perceptron algorithm

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test


highAccSLP = 0.0
highAccMLP = 0.0

def truncate(f, n): # simple function for making the accuracy results more readable
    #Truncates/pads a float f to n decimal places without rounding
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

for rate in n: #iterates over n, each learning rate

    for shuffle in r: #iterates over r, shuffle is true or false

        #iterates over both algorithms
        #-->add your Pyhton code here


        for algorithm in a: #iterates over the algorithms (single layer, multi layer)

            #Create a Neural Network classifier
            #if Perceptron then
            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Pyhton code here
            if (algorithm == 0):
                clf = Perceptron(eta0=rate, shuffle=shuffle, max_iter=1000)
            elif (algorithm == 1):
                clf = MLPClassifier(activation='logistic', learning_rate_init=rate, hidden_layer_sizes=25, shuffle=shuffle, max_iter=1000)


            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            totalPred = 0
            correctPred = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                y_pred = clf.predict([x_testSample])
                totalPred += 1
                if(y_pred == y_testSample):
                    correctPred += 1
            
            currentAcc = correctPred/totalPred


            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here

            if(algorithm == 0):
                highAccSLP = max(highAccSLP, currentAcc)
                print("Highest Perceptron accuracy so far: " + str(truncate(highAccSLP, 4)) + ", Parameters: learning rate=" + str(rate) + ", shuffle=" + str(shuffle))
            elif(algorithm == 1):
                highAccMLP = max(highAccMLP, currentAcc)
                print("Highest MLP accuracy so far: " + str(truncate(highAccMLP, 4)) + ", Parameters: learning rate=" + str(rate) + ", shuffle=" + str(shuffle))



## Coded with love by Julianne Ong







