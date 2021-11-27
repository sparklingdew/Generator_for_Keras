# Generator for Keras model

When a dataset is too large to be handled all at once, a generator can be used to continuously feed our model with small portions of the dataset.

This generator processes 1D datasets, typical of signals. Signals may come in different lengths, thus, this generator pads them with zeros to a certain given dimension.

## Data format

Each signal (X) is one dimensional and is stored in a .npy file. These files load much much faster than, for instance, .csv files. Hence, it is worth saving the dataset into this format.

All output values (y's) are stored in one .csv file with two columns. The first column is the name of each .npy file, and the second column is the corresponding output (y).

To feed the present model, a dummie dataset is generated. The signal (X) is a random set of random real numbers in the range [0, 10]. The output is a polynomial using the first three values of the signal.

## Data pre-processing

The dataset is split into train, validation and test sets. Then, train and validation sets are normalized using the generator class.

## Model

A Keras regression CNN is used to model the dataset. 

## Model evaluation

The model evaluation is included to show how to use the generator class in this part of the analysis. Errors and predictions are shown in plots.
