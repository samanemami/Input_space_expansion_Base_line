# Base line
This directory includes the following;
* [_multivariate](_multivariate.py)
* [training](training.py)
* [output_extraction](output_extraction.py)
  * Dataset
      * [Dataset](Dataset/Dataset.py)
      * [info](Dataset/info.txt)

## _multivariate

Includes three different experiments. The first experiment is a simple regression model that trains over the input and tries to predict each output. The secon experiment, reveals the best possible results by using the rest of outputs (m-1) as the input to predict the selected outout (m'). The final design augmented the input with the outputs to predict the target.  

## training

Trains different models (NN, RF, Bgg) with the three different approaches and saves the results in CSV format.

## output_extraction

To have a pivot table from the generated CSV files, the output_extraction needs to be used.

## Dataset
### Dataset

includes a method to extract the variables from the MTR datasets and deal with the missing values.

### info

MTR dataset information.


