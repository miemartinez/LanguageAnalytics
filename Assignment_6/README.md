# Assignment 6 - Text classification using Deep Learning
This repository contains all of the code and data related to Assignment 6 for Language Analytics.
The data can also be found on Kaggle:
https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons

In the data folder there is a csv file with all the lines from Game of Thrones and labels of the season to which they belong.
The src folder contains two scripts: logRegModel.py and deepLearnModel.py and both scripts takes the path to the csv file as a required input. 
For the logRegModel.py, the user can also specify size of the test data in percentage points. The default will be 0.25.
For the deepLearnModel.py, the user can also define test size as well as pooling method, optimizer and number of epochs to train over. Defaults are specified for all of these.

The outputs of both scripts are saved in the created outpu folder. For the logistic regression script this involves two png files 
(classification matrix and cross validation graphs) and a csv file of the classification report which will also be printed in the terminal.
For the deep learning model, the output consists of two png files (model architecture and plot of model history) and a txt file of the model architecture. For this model,
the script also prints accuracy on training and testing data in the terminal. <br>


## logRegModel.py

__Parameters:__ <br>
```
    input_file: str <filepath-of-csv-file>
    output_filename: str <name-of-png-file>, default = logReg_model_history.png
    test_size: float <size-of-test-data>, default = 0.25
```
    
__Usage:__ <br>
```
    logRegModel.py -f <filepath-of-csv-file> -o <name-of-png-file> -t <size-of-test-data>
```
    
__Example:__ <br>
```
    $ python3 logRegModel.py -f ../data/Game_of_Thrones_Script.csv -o LogReg_cross_validation.png -t 0.25
```


## deepLearnModel.py

__Parameters:__ <br>
```
    input_file: str <filepath-of-csv-file>
    output_filename: str <name-of-png-file>
    test_size: float <size-of-test-data>
    optimizer: str <optimization-method>
    pooling: str <pooling-method>
    n_epochs: int <number-of-epochs>
```
    
__Usage:__ <br>
```
    deepLearnModel.py -f <filepath-of-csv-file> -o <name-of-png-file> -t <size-of-test-data> -opt <optimization-method> -p <pooling-method> -e <number-of-epochs>
```
    
__Example:__ <br>
```
    $ python3 deepLearnModel.py -f ../data/Game_of_Thrones_Script.csv -o deepLearn_model_history.png -t 0.25 -opt adam -p MaxPooling -e 10
```



## Virtual Environment
To ensure dependencies are in accordance with the ones used for the script, you can create the virtual environment "GoT_venv" by running the bash script create_GoT_venv.sh
```
    $ bash ./create_GoT_venv.sh
```
After creating the environment, you have to activate it. And then you can run either of the scripts with the dependencies:
```
    $ source GoT_venv/bin/activate
    $ cd src
    $ python3 deepLearnModel.py -f ../data/Game_of_Thrones_Script.csv
```
The outputs will appear in the output folder which will be created (if it doesn't already exists).


### Results:
For a view of the results when running the model over 50 epochs with adam optimizer and max pooling see the output folder in this repo. <br>

__Training and testing results using alternating SGD, adam, max and average pooling across 50 epochs:__
Train and test acc: 0.26, 0.18 for SGD with MaxPooling (50 epochs) <br>
Train and test acc: 0.94, 0.24 for adam with MaxPooling (50 epochs) <br>
Train and test acc: 0.67, 0.20 for adam with AvgPooling (50 epochs) <br>
Train and test acc: 0.16, 0.17 for SGD with AvgPooling (50 epochs) <br>

