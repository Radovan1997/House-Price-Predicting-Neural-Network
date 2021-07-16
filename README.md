# Introduction to ML - Neural Networks Coursework


## Part 1 implements a neural network model


## Installation

To make sure that you can run the project please install the following libraries if you have not already

```bash
pip install numpy
pip install pickle
```


## Usage

To get started using our library please import the main scripts

```python
import part1_nn_lib
```

### Implementing a neural network

To create a neural network input the input dimension, neurons and activation functions and then
create an instance of a `MultiLayerNetwork` class using 

```python
net = MultiLayerNetwork(input_dim, neurons, activations)
```

Download your data using `np.loadtxt()` which will then be split into training testing sets.
In addition a preprocessor is created based on the x training data.

To implement a trainer the inputs to be implemented by the user are the batch size, the number
of episodes the learning rate and the loss function.



## Part 2 develops a neural network model that predicts the median house price of block groups in California

We are trying to predict the `median_house_value`.  The training dataset has the following attributes:

```bash
longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
population, households, median_income, ocean_proximity.

```

The model was developed and trained on the data that can be found in `housing.csv`.


## Installation

To make sure that you can run the project please install the following libraries if you have not already

```bash
pip install numpy
pip install matplotlib
pip install torch
pip install pandas
pip install sklearn
pip install pickle
```

Or run `export PYTHONUSERBASE=/vol/lab/ml/mlenv` on lab machines.

## Usage

To see how our model trains and performs on our existing data you can just run the `part2_house_value_regression.py` file. 

```bash
python3 part2_house_value_regression.py
```

Alternatively, run the following commands to train the network with `X_train` and `y_train` datasets with default parameters, and evaluate with `X_test` and `y_test`.  The testsets are also provided to the fit function for early stopping:
```python
# Initiate the regressor with default parameters
regressor = Regressor()
# Train the network.  The optional X_test and y_test parameters enable early stopping
regressor.fit(X_train, y_train, X_test, y_test)
# Save the trained model into a pickle file so you can return to it later
save_regressor(regressor)
```

```python
# Load a previously saved model
regressor = load_regressor()
# Get the predictions on the test set
predictions = regressor.predict(X_test)
# Get the RMS error on the test set
error = regressor.score(X_test, y_test)
```

### Cross Validating 

To perform cross validation on your own data you can use our cross validation function.

```python
cross_validation(X, y, folds = 10, parameters = None, verbose = False)
```

Input your housing data as `X` and the median house prices as `y`. You can also select the number of folds and whether you want prints from the updates. You can also input your own paramenters as a dictionary. 

A sample parameters dictionary would be

```python
parameters = {'activation_function': 'elu', 'hidden_layers': [20, 20], 'dropout': 0, 'learning_rate': 0.2, 'nb_epoch': 1500, 'columns': ["median_income", "avg_bedrooms_per_room"]}

```

The full list of parameters available can be found in the introductory commments in the constructor of the Regressor class in `part2_house_value_regression.py`.
 
### Hyper Parameter Optimisation

To find your optimal hyperparamers you can run
```python
RegressorHyperParameterSearch(X, y)
```
`X` and `y` are the input data.

To select what hyperparameters to do a grid search over, you need to edit the arrays in the function. The function returns the best hyperparameter choice, it's error array and the errors of all other choices.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
