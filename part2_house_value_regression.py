import torch
import pickle
import time
import copy
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt

class Regressor():

    def __init__(self, x = None, nb_epoch = None, parameters = None, verbose = False):

        """ 
        Initialise the model.
          
        Arguments:
            - x {any} -- dummy variable to adhere to coursework specs
            - nb_epoch {int} -- number of epochs to train the network.  Will overwrite any epochs given in parameters
            - parameters {dict} -- dictionary of values to set hyperparameters or None for default values
                Supported hyperparameter choices are:
                    - 'activation_function': 'relu', 'sigmoid', 'tanh', 'elu', 'leakyrelu'. Default: 'elu'
                    - 'hidden_layers': list of ints Default: [20, 20].  Each element is a hidden layer with that number of neurons.
                    - 'dropout': int between 0 and 1.  Default: 0
                    - 'learning_rate': float.  Default: 0.2
                    - 'minibatch_size' int.  Default: 256
                    - 'nb_epoch': int.  Default: 1500.  Maximum number of epochs.
                    - 'columns': list of attributes to use.  Default: ['median_income', 'avg_bedrooms_per_room', 'avg_rooms_per_house', 'avg_household_size', 'housing_median_age', 'longitude', 'latitude']
                        The full list of available columns is: ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity', 'avg_household_size', 'avg_bedrooms_per_room', 'avg_rooms_per_house']

        """
        self.parameters = parameters
        self.verbose = verbose
        self.network = None
        self.input_size = None
        self.output_size = None
        self.preprocessing_parameters = None

        # Save the number of epochs into the parameters dictionary
        if nb_epoch is not None:
            if self.parameters is None:
                self.parameters = {'nb_epoch': nb_epoch}
            else:
                self.parameters['nb_epoch'] = nb_epoch

        return


    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """
        # Parse parameters with defaults if not provided
        columns = ['median_income', 'avg_bedrooms_per_room', 'avg_rooms_per_house', 'avg_household_size', 'housing_median_age', 'longitude', 'latitude']
        if self.parameters is not None:
            if 'columns' in self.parameters:
                columns = self.parameters['columns']
        
        # Copy the dataframe to ensure we're not editing a pointer
        x = x.copy()
        
        # Check that we haven't tried to get predictions from the model before we've trained it
        if training == False and self.preprocessing_parameters is None:
            raise KeyError('New data used on the model before the model has been trained.')
        elif self.preprocessing_parameters == None:
            self.preprocessing_parameters = {}

        # Drop ocean proximity and replace with encoding later
        x_ocean = x["ocean_proximity"]
        x.drop(columns=['ocean_proximity'], axis=1, inplace=True)

        # Save the columns titles and indicies for when we recreate dataframes from numpy arrays later
        x_columns = x.columns
        x_indices = x.index

        # Set missing values to median
        if training:
            self.preprocessing_parameters['imputer'] = SimpleImputer(strategy="median")
            self.preprocessing_parameters['imputer'].fit(x)
        x = self.preprocessing_parameters['imputer'].transform(x)
        x = pd.DataFrame(x, columns=x_columns, index=x_indices)

        # Create combinations
        x["avg_household_size"] = x["population"] / x["households"]
        x["avg_bedrooms_per_room"] = x["total_bedrooms"] / x["total_rooms"]
        x["avg_rooms_per_house"] = x["total_rooms"] / x["households"]

        # Standardise the data
        # Uncomment the line below if you want to do min/max on the longitude and latitude
        # x_to_standardise = x.drop(columns = ['longitude', 'latitude'])
        x_to_standardise = x
        x_columns = x_to_standardise.columns
        x_to_standardise = x_to_standardise.values
        if training:
            self.preprocessing_parameters['scaler'] = preprocessing.StandardScaler()
            self.preprocessing_parameters['scaler'].fit(x_to_standardise)
        x_to_standardise = self.preprocessing_parameters['scaler'].transform(x_to_standardise)
        x_standardised = pd.DataFrame(x_to_standardise, columns=x_columns, index=x_indices)
        x = x_standardised

        # Do min max on the longitude and latitude (did not improve the results)
        '''x_to_minmax = x[['longitude', 'latitude']]
        x_columns = x_to_minmax.columns
        x_to_minmax = x_to_minmax.values
        if training:
            self.preprocessing_parameters['min_max_scaler'] = preprocessing.MinMaxScaler(feature_range=(-1,1))
            self.preprocessing_parameters['min_max_scaler'].fit(x_to_minmax)
        x_to_minmax = self.preprocessing_parameters['min_max_scaler'].transform(x_to_minmax)
        x_minmax = pd.DataFrame(x_to_minmax, columns=x_columns, index=x_indices)

        # Combine
        #x_standardised.reset_index(inplace=True, drop=True)
        #x_minmax.reset_index(inplace=True, drop=True)
        x = pd.concat([x_minmax, x_standardised], axis=1)'''

        # Remove columns we don't want, as defined by the provided parameters to the class constructor
        columns_to_remove = set(x.columns) - set(columns)
        x.drop(columns=columns_to_remove, axis=1, inplace=True)

        if self.verbose:
            print(x.head())

        # One Hot encoding for ocean_proximity
        if training:
            self.preprocessing_parameters['ocean_encoder'] = LabelBinarizer()
            self.preprocessing_parameters['ocean_encoder'].fit(x_ocean)
        encoded = self.preprocessing_parameters['ocean_encoder'].transform(x_ocean)
        encoded_df = pd.DataFrame(encoded)
        encoded_df.reset_index(inplace=True, drop=True)
        x.reset_index(inplace=True, drop=True)
        x = pd.concat([x, encoded_df], axis=1)

        if self.verbose:
            print(x.head())

        # Convert to torch tensor
        x = torch.tensor(x.to_numpy(), dtype=torch.float32)
        
        if isinstance(y, pd.DataFrame):
            y = torch.tensor(y.to_numpy(), dtype=torch.float32)
        else:
            y = None
        
        return x, y
        

    def fit(self, x, y, validation_set_x = None, validation_set_y = None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
            - validation_set_x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - validation_set_y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        # Parse parameters with defaults if not provided
        minibatch_size = 256
        nb_epoch = 1500
        if self.parameters is not None:
            if 'minibatch_size' in self.parameters:
                minibatch_size = self.parameters['minibatch_size']
            if 'nb_epoch' in self.parameters:
                nb_epoch = self.parameters['nb_epoch']

        # Preprocess the data
        X, Y = self._preprocessor(x, y, True)
        self.input_size = X.shape[1]
        self.output_size = 1

        # Process the validation set if we were given it
        if validation_set_x is not None:
            validation_set_x, validation_set_y = self._preprocessor(validation_set_x, validation_set_y, False)

        # Initialise variables before the loop
        training_loss = []
        validation_loss = []
        best_model = None
        start_time = time.time()
        counter = 0

        while True:
            counter += 1
            # Reinitialise the model
            self.initialise_network(self.input_size, self.output_size)

            for epoch in range(nb_epoch):
                step_losses = []
                # Put the network into training mode (used for dropout)
                self.network.train()
                # Randomly shuffle indicies
                all_indices = np.random.permutation(len(X))
                # Split indicies into chunked minibatches
                if len(X) > minibatch_size:
                    minibatch_indicies = np.array_split(all_indices, round(len(X)/minibatch_size))
                else:
                    minibatch_indicies = [all_indices]
                for i in minibatch_indicies:
                    inputs = X[i]
                    labels = Y[i]

                    # Get the outputs and loss from the network
                    self.optimiser.zero_grad()
                    outputs = self.network(inputs)
                    loss = torch.nn.MSELoss()(outputs, labels)

                    # Do back propagation and update the network
                    loss.backward()
                    self.optimiser.step()

                    step_losses.append(np.sqrt(loss.item()))

                training_loss.append(sum(step_losses)/len(step_losses))

                # Process early stopping
                if validation_set_x is not None:
                    # Put the network into evaluation mode for dropout
                    self.network.eval()
                    # Get the predictions for the validation set
                    with torch.no_grad():
                        predictions = self.network(validation_set_x)
                        loss = torch.nn.MSELoss()(predictions, validation_set_y)
                        validation_loss.append(np.sqrt(loss.item()))

                    # Save the model if it's the minimum
                    if validation_loss[-1] == min(validation_loss):
                        best_model = copy.deepcopy(self.network.state_dict())
                    else:
                        # Check if we should stop early
                        min_loss_idx = validation_loss.index(min(validation_loss))
                        if epoch - min_loss_idx > 50:
                            if self.verbose:
                                print(f'Stopping early with minimum error of {min(validation_loss):.1f}')
                            break
            # Check if we didn't converge
            if training_loss[-1] > 200000:
                # If we didn't, retry upto 3 times
                if counter >= 3:
                    if self.verbose:
                        print(f'Did not converge on attempt {counter}.  Error {training_loss[-1]:.1f}.  Break.')
                    break
                if self.verbose:
                    print(f'Did not converge on attempt {counter}.  Error {training_loss[-1]:.1f}.  Retrying.')
            else:
                break

            if self.verbose and epoch % 50 == 0:
                print(f'Epoch {epoch} finished in {time.time()-start_time:.1f} seconds with training loss {training_loss[-1]:.1f} and validation loss {validation_loss[-1]:.1f}')
                start_time = time.time()

        # Use the best model
        if best_model is not None:
            self.network.load_state_dict(best_model)

        # Save the training loss and validation loss as we have to return self.
        self.training_loss = training_loss
        self.validation_loss = validation_loss

        return self

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        # Preprocess the input
        X, _ = self._preprocessor(x, training = False)
        # Make sure we're in evaluation mode for dropout
        self.network.eval()
        # Get the predictions from the network and return
        with torch.no_grad():
            outputs = self.network(X)
        return outputs.detach().numpy()


    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
        # Preprocess the input
        X, Y = self._preprocessor(x, y = y, training = False)
        # Make sure we're in evaluation mode for dropout
        self.network.eval()
        # Get the predictions from the network and calculate the loss
        with torch.no_grad():
            predictions = self.network(X)
            loss = torch.nn.MSELoss()(predictions, Y).detach().numpy()
        return np.sqrt(loss.item())

    
    def initialise_network(self, input_size, output_size):
        """
        Function to initialise the network

        Arguments:
            - input_size {int} -- Number of input nodes
            - output_size {int} -- Number of output nodes

        Returns:
            None

        """
        # Parse parameters with defaults if not provided
        learning_rate = 0.2
        if 'learning_rate' in self.parameters:
            learning_rate = self.parameters['learning_rate']

        # Create network and define the optimiser
        self.network = Network(self.input_size, self.output_size, self.parameters, self.verbose)
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=learning_rate)


    def get_model_parameters(self):
        """
        Function to return all model parameters required to recreate the model later

        Arguments:
            None

        Returns:
            {dict} -- A dictionary that can be later passed into load_model_parameters() to recreate the model

        """
        # Collect the attributes we need to save into a dictionary and return
        hyperparameters = self.parameters
        input_size = self.input_size
        output_size = self.output_size
        state_dict = self.network.state_dict()
        preprocessing_parameters = self.preprocessing_parameters

        model_data = (input_size, output_size, hyperparameters, preprocessing_parameters, state_dict)
        return model_data


    def load_model_parameters(self, model_data):
        """
        Function to initialise the class with the data saved from get_model_parameters()

        Arguments:
            {dict} -- A dictionary from save_model_parameters() that contains the parameters to recreate the class

        Returns:
            None

        """
        # Unpack the parameters
        (input_size, output_size, hyperparameters, preprocessing_parameters, state_dict) = model_data
        # Save them in the right place
        self.parameters = hyperparameters
        self.preprocessing_parameters = preprocessing_parameters
        self.input_size = input_size
        self.output_size = output_size
        # Initialise the network
        if self.network is None:
            self.initialise_network(input_size, output_size)
        self.network.load_state_dict(state_dict)


# Use Pytorch to create the neural network
class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, parameters = None, verbose = False):
        """
        Create a new network

        Arguments:
            - input_dimension {int} -- Number of input nodes
            - output_dimension {int} -- Number of output nodes
            - parameters {dict} -- dictionary of values to set hyperparameters or None for default values
                Supported hyperparameter choices are:
                    - 'activation_function': 'relu', 'sigmoid', 'tanh', 'elu', 'leakyrelu'. Default: 'elu'
                    - 'hidden_layers': list of ints Default: [20, 20].  Each element is a hidden layer with that number of neurons.
                    - 'dropout': int between 0 and 1.  Default: 0
            - verbose {boolean} -- True for prints. False to be quiet.

        Returns:
            None

        """
        super(Network, self).__init__()

        # Parse parameters with defaults if not provided
        hidden_layers = [20, 20]
        activation_function = 'elu'
        dropout = 0
        if parameters is not None:
            if 'hidden_layers' in parameters:
                hidden_layers = parameters['hidden_layers']
            if 'activation_function' in parameters:
                activation_function = parameters['activation_function']
            if 'dropout' in parameters:
                dropout = parameters['dropout']
                if dropout > 1:
                    dropout = 0.5
        
        # Loop through layers and create network
        layer_dims = [input_dimension, *hidden_layers, output_dimension]
        layers = []
        i = 0
        for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
            # Create the linear layer
            layers.append(torch.nn.Linear(input_dim, output_dim))
            # Add the activation function
            # Make sure we put ReLU as the last activation function
            if activation_function == 'sigmoid' and i < len(hidden_layers):
                layers.append(torch.nn.Sigmoid())
            elif activation_function == 'tanh' and i < len(hidden_layers):
                layers.append(torch.nn.Tanh())
            elif activation_function == 'elu' and i < len(hidden_layers):
                layers.append(torch.nn.ELU())
            elif activation_function == 'leakyrelu' and i < len(hidden_layers):
                layers.append(torch.nn.LeakyReLU())
            else:
                layers.append(torch.nn.ReLU())
            # Add dropout on the hidden layers
            if dropout > 0 and i < len(hidden_layers):
                layers.append(torch.nn.Dropout(dropout))
            i += 1
        self.seq = torch.nn.Sequential(*layers)

        if verbose:
            print(self.seq)

    # Pass data through the network to obtain output
    def forward(self, input):
        return self.seq(input)


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """    
    model_data = trained_model.get_model_parameters()
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(model_data, target)
    print("Saved model in part2_model.pickle")

def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    with open('part2_model.pickle', 'rb') as target:
        model_data = pickle.load(target)

    trained_model = Regressor()
    trained_model.load_model_parameters(model_data)
    print("Loaded model in part2_model.pickle")
    return trained_model


def RegressorHyperParameterSearch(X, y): 
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        X = Input data 
        y = target values
        
    Returns:
        {list} -- [best parameters in dictionary, list of best parameter errors, list of other parameter options and errors]

    """

    # Select what values you want to search over
    # Input as list and the function will run cross validation on every combination using a for loop
    columns = ['longitude', 'latitude', 'housing_median_age', 'population', 'median_income', 'ocean_proximity', 'avg_household_size', 'avg_rooms_per_house']
    nb_epoch_options = [1500]
    learning_rate_options = [0.2]
    dropout_options = [0.0]
    activation_function_options = ["elu"]
    hidden_layers_options = [[8,8], [10,10], [20, 20], [30,30]]
    results_array = []

    # initializing the best_params dictionary to the first option
    best_params = {"nb_epoch": nb_epoch_options[0], "learning_rate": learning_rate_options[0],"dropout": dropout_options[0], "activation_function": activation_function_options[0],"hidden_layers": hidden_layers_options[0], "columns": columns}
    
    # initializing best_score to the cross_validation result of best_params
    best_score = cross_validation(X,y, parameters=best_params)

    # loop through every option
    for n1 in range(len(nb_epoch_options)):
        for n2 in range(len(learning_rate_options)):
            for n3 in range(len(dropout_options)):
                for n4 in range(len(activation_function_options)):
                    for n5 in range(len(hidden_layers_options)):
                        # set our test_params dictionary to the parameters we are testing
                        test_params = {"nb_epoch": nb_epoch_options[n1],"learning_rate": learning_rate_options[n2],"dropout": dropout_options[n3],"activation_function": activation_function_options[n4],"hidden_layers": hidden_layers_options[n5], "columns": columns}
                        print("Testing:", test_params)

                        # find out how this combination performed
                        test_score = cross_validation(X, y, parameters = test_params)
                        print("Test Score:", test_score)
                        
                        # Pickle results array
                        results_array += [[test_params, test_score]]
                        with open('search_results.pickle', 'wb') as target:
                            pickle.dump(results_array, target)
                        
                        # if our test_score was lower than the best score
                        # we make that test score our new best 
                        # and set best_params to the current option
                        if (sum(test_score) < sum(best_score)):
                            print("FOUND NEW BEST!")
                            best_score = test_score
                            best_params = test_params
                        
                        print("______________________")
    
    # return the best_option and it's score and all the other results
    return [best_params, best_score, results_array]


def attribute_sensitivity_search(X, y):
    """
    Run through each attribute and do cross-validation with and without the column and pickle the results for later processing.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
        - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

    Returns:
        None
    """
    # List of all attributes we should cycle through
    all_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity', 'avg_household_size', 'avg_bedrooms_per_room', 'avg_rooms_per_house']
    # Default columns to use
    default_cols = {'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'population', 'households', 'median_income', 'ocean_proximity', 'avg_household_size'}
    # Output file name
    pickle_name = 'column_results_1.pickle'
    # Parameters to use for the analysis
    parameters = {'nb_epoch': 1500, 'learning_rate': 0.2, 'dropout': 0, 'activation_function': 'elu', 'hidden_layers': (20, 20)}
    # Dictionary for the results
    results = {'column': [], 'with': [], 'without': []}
    # Dictionary to check if we're trying to compute the error for a combination we've already processed
    results_dictionary = {}

    # Loop through every combination
    for c in all_columns:
        print(f'Processing column {c}')
        results['column'].append(c)

        # Create the columns parameters with and without c:
        default_cols.add(c)
        with_c = tuple(sorted(default_cols))
        without_c = default_cols.remove(c)
        without_c = tuple(sorted(default_cols))
        
        # Run cross validation on both
        for i, columns in enumerate([with_c, without_c]):
            parameters['columns'] = columns

            # Check to see if we've already computed this result before
            key = frozenset(parameters.items())
            if key in results_dictionary:
                # We have processed this before, so get the result
                print('Retrieving previously calculated result')
                test_scores = results_dictionary[key]
            else:
                # We haven't processed this before, so do cross validation
                test_scores = cross_validation(X, y, 10, parameters = parameters, verbose = True)
                results_dictionary[key] = test_scores
            if i == 0:
                # Save the results 'with' the column added
                results['with'].append(test_scores)
            else:
                # Save the results 'without' the column added
                results['without'].append(test_scores)

        # Pickle all the data so we can plot it
        with open(pickle_name, 'wb') as target:
            pickle.dump(results, target)


def cross_validation(X, y, folds = 10, parameters = None, verbose = False):
    """
    Do cross validation

    Arguments:
        - X {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
        - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
        - folds {int} -- number of folds in the cross validation
        - parameters {dict} -- dictionary of parameters for the Regressor class.  See the Regressor class for options.
        - verbose {boolean} -- True for prints.  False to stay quiet.

    Returns:
        {list} -- List of errors from each fold.
    """
    # Copy the data provided and ensure reset the indices in case we were passed a slice.
    X = X.copy().reset_index(drop = True)
    y = y.copy().reset_index(drop = True)

    # Get the indices for the folds
    if verbose:
        print(f'Cross validating with the following parameters: {parameters}')
    data_split_ind = train_test_split_ind(X, folds, seed = 42)
    indices = np.arange(len(X))
    errors = []

    if verbose:
        # Prepare live graphing to monitor progress
        fig, ax = plt.subplots()
        plt.ion()

    # Process each fold
    for counter, test_ind in enumerate(data_split_ind, 1):
        # Remove the test indicies to leave the training indices
        train_ind = np.delete(indices, test_ind)
        # Get the train and test data
        X_train = X.loc[train_ind, :]
        y_train = y.loc[train_ind, :]
        X_test = X.loc[test_ind, :]
        y_test = y.loc[test_ind, :]

        # Train the model
        regressor = Regressor(parameters = parameters, verbose = False)
        regressor.fit(X_train, y_train, X_test, y_test)
        train_loss = regressor.training_loss
        validation_loss = regressor.validation_loss

        if verbose:
            # Plot the training and validation loss over time
            ax.clear()
            ax.plot(train_loss, label='Training loss')
            ax.plot(validation_loss, label='Validation loss')
            ax.set_ylabel('RMS error')
            ax.legend()
            ax.set_ylim([45000, 75000])
            ax.grid()
            ax.set_xlabel('Epochs')
            plt.show()
            plt.pause(1)
            plt.tight_layout()
            plt.savefig(f'training_loss_{counter}')

        # Evaluate the performance and save to the errors list
        e = regressor.score(X_test, y_test)
        errors.append(e)
        if verbose:
            print(f'Fold {counter} error: {e:.1f}')

    if verbose:
        print(f'Results of CV: {errors}')
    return errors


def train_test_split_ind(data, n_groups, seed = None):
    """
    Randomly splits data into n groups

    Arguments:
        - data {np.array} -- Array to split into groups
        - n_groups {pd.DataFrame} -- Number of groups
        - seed {int} -- seed for the random number generator

    Returns:
        {list} -- List of numpy arrays
    """
    if seed is not None and seed > 0:
        np.random.seed(seed)
    all_indices = np.random.permutation(len(data))
    return np.array_split(all_indices, n_groups)


def train_test_final_model():
    '''
    Train on all the data and produce final model
    '''

    output_label = "median_house_value"
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    X = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Split off a test set
    data_split_ind = train_test_split_ind(X, 10, seed = 42)
    all_indices = np.arange(len(X))
    test_ind = data_split_ind[0]

    train_ind = np.delete(all_indices , test_ind)
    X_train = X.loc[train_ind, :]
    y_train = y.loc[train_ind, :]
    X_test = X.loc[test_ind, :]
    y_test = y.loc[test_ind, :]

    # Set the parameters
    columns = ['median_income', 'avg_bedrooms_per_room', 'avg_rooms_per_house', 'avg_household_size', 'housing_median_age', 'longitude', 'latitude']
    parameters = {'activation_function': 'elu', 'hidden_layers': [20, 20], 'dropout': 0, 'learning_rate': 0.2, 'nb_epoch': 1500, 'columns': columns}
    
    # Get the estimated error from cross-validation
    errors = cross_validation(X_train, y_train, parameters = parameters)
    print(f'All errors: {errors}')
    print(f'Mean error: {sum(errors)/len(errors):.0f}')

    # Run this one more time but save the final model into a pickle
    regressor = Regressor(parameters = parameters)
    regressor.fit(X_train, y_train, X_test, y_test)
    save_regressor(regressor)

    # Error
    error = regressor.score(X_test, y_test)
    print(f'Regressor error: {error:.0f}')

    # Predictions
    predictions = regressor.predict(X_test)
    with open('final_model_errors.pickle', 'wb') as target:
        pickle.dump([predictions, X_test, y_test], target)    

if __name__ == "__main__":
    train_test_final_model()