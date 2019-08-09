import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Binary Classification
# Exoplanet hunting

#--- Getting the data in order ---

df_train = pd.read_csv('exoTrain.csv') 	# 5087 rows
df_test = pd.read_csv('exoTest.csv') 	# 570 rows
# first column is label vector
	# 2 is an exoplanet star
	# 1 is a non-exoplanet-star.
# then 3197 columns 
	# the light intensity recorded for each star, at a different point in time.

# convert datasets into arrays
dataset_train = df_train.values
dataset_test = df_test.values

# now make sure scale of input features is similar
# scale the dataset so that all the input features lie between 0 and 1 inclusive
min_max_scaler = preprocessing.MinMaxScaler()
train_scaled = min_max_scaler.fit_transform(dataset_train)
test_scaled = min_max_scaler.fit_transform(dataset_test)

# split dataset into input features X and feature to predict Y
X_train = train_scaled[:,1:3197]
Y_train = train_scaled[:,0]
X_test = test_scaled[:,1:3197]
Y_test = test_scaled[:,0]

# split the training dataset into a training set and a validation set
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15) 


#--- Neural network ---
# Architecture of NN:
	# Input layer: 3197 neurons
	# Hidden layer 1: 42 neurons, ReLU activation
	# Hidden layer 2: 42 neurons, ReLU activation
	# Output Layer: 1 neuron, Sigmoid activation	

# Specify Keras model sequentially
model = Sequential([ Dense(42, activation='relu', input_shape=(3196,)), Dense(42, activation='relu'), Dense(1, activation='sigmoid'),])

# configure the model:
	# which algorithm to use for optimization : stochastic gradient descent
	# which loss function : binary cross entropy
	# other metrics to track
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# train
model.fit(X_train, Y_train, batch_size=42, epochs=100, validation_data=(X_val, Y_val))

# evaluate model on the test set
evaluation = model.evaluate(X_test, Y_test)
print("Accuracy: ", evaluation[1]) # output the accuracy only by accessing the second element


