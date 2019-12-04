import numpy as np

class NeuralNetwork:

	def __init__(self, lr, batchSize, epochs):
		# Method to initialize a Neural Network Object
		# Parameters
		# lr - learning rate
		# batchSize - Mini batch size
		# epochs - Number of epochs for training
		self.lr = lr
		self.batchSize = batchSize
		self.epochs = epochs
		self.layers = []

	def addLayer(self, layer):
		# Method to add layers to the Neural Network
		self.layers.append(layer)

	def train(self, trainX, trainY, validX=None, validY=None):
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# Feel free to print accuracy at different points using the validate() or computerAccuracy() functions of this class
		###############################################
		# TASK 2c (Marks 0) - YOUR CODE HERE
		# raise NotImplementedError
		for i in range(self.epochs):
			#print(trainX,trainY)
			rand=np.arange(len(trainX))
			np.random.shuffle(rand)
			trainX = trainX[rand]
			trainY = trainY[rand]
			a=0
			for m in range(int(trainX.shape[0]/self.batchSize)):
				trainXi=trainX[a:a+self.batchSize]
				trainYi=trainY[a:a+self.batchSize]
				a+=self.batchSize
				activation=[trainXi]
				#print(trainY)
				for l in self.layers:
					activation.append(l.forwardpass(activation[-1]))
				#print(activation[-1])
				delta = self.crossEntropyDelta(trainYi,activation[-1])
				#print(delta)
				for k in range(len(self.layers)-1, -1, -1):
					#print(activation[k])
					delta = self.layers[k].backwardpass(activation[k], delta)
					self.layers[k].updateWeights(self.lr)
			print("Iteration "+str(i))
		print(self.validate(validX,validY)[1])

		###############################################
		
	def crossEntropyLoss(self, Y, predictions):
		# Input 
		# Y : Ground truth labels (encoded as 1-hot vectors) | shape = batchSize x number of output labels
		# predictions : Predictions of the model | shape = batchSize x number of output labels
		# Returns the cross-entropy loss between the predictions and the ground truth labels | shape = scalar
		###############################################
		# TASK 2a (Marks 3) - YOUR CODE HERE
		# raise NotImplementedError
		# print(Y.shape)
		# print(predictions.shape) 
		'''log_likelihood = -np.log(p[range(m),Y])
		loss = np.sum(log_likelihood) / m
		return loss'''
		return -np.sum(Y*np.log(predictions))
		###############################################

	def crossEntropyDelta(self, Y, predictions):
		# Input 
		# Y : Ground truth labels (encoded as 1-hot vectors) | shape = batchSize x number of output labels
		# predictions : Predictions of the model | shape = batchSize x number of output labels
		# Returns the derivative of the loss with respect to the last layer outputs, ie dL/dp_i where p_i is the ith 
		#		output of the last layer of the network | shape = batchSize x number of output labels
		###############################################
		# TASK 2b (Marks 3) - YOUR CODE HERE
		# raise NotImplementedError
		#val[val>0]=-val[val>0]
		#print(predictions)
		#print(Y)
		#print(predictions)
		#print(-np.divide(Y,predictions))
		return -1*np.divide(Y,(predictions+1e-13)) 
		###############################################
		
	def computeAccuracy(self, Y, predictions):
		# Returns the accuracy given the true labels Y and final output of the model
		correct = 0
		for i in range(len(Y)):
			if np.argmax(Y[i]) == np.argmax(predictions[i]):
				correct += 1
		accuracy = (float(correct) / len(Y)) * 100
		return accuracy

	def validate(self, validX, validY):
		# Input 
		# validX : Validation Input Data
		# validY : Validation Labels
		# Returns the predictions and validation accuracy evaluated over the current neural network model
		valActivations = self.predict(validX)
		pred = np.argmax(valActivations, axis=1)
		if validY is not None:
			valAcc = self.computeAccuracy(validY, valActivations)
			return pred, valAcc
		else:
			return pred, None

	def predict(self, X):
		# Input
		# X : Current Batch of Input Data as an nparray
		# Output
		# Returns the predictions made by the model (which are the activations output by the last layer)
		# Note: Activations at the first layer(input layer) is X itself		
		activations = X
		for l in self.layers:
			activations = l.forwardpass(activations)
		return activations







class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes, activation):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		self.activation = activation
		# Stores a quantity that is computed in the forward pass but actually used in the backward pass. Try to identify
		# this quantity to avoid recomputing it in the backward pass and hence, speed up computation
		self.data = None

		# Create np arrays of appropriate sizes for weights and biases and initialise them as you see fit
		###############################################
		# TASK 1a (Marks 0) - YOUR CODE HERE
		# raise NotImplementedError
		self.weights = np.random.rand(in_nodes,out_nodes)
		self.biases = np.random.rand(out_nodes)
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary
		
		# Store the gradients with respect to the weights and biases in these variables during the backward pass
		self.weightsGrad = None
		self.biasesGrad = None

	def relu_of_X(self, X):
		# Input
		# data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
		# Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
		# This will only be called for layers with activation relu
		###############################################
		# TASK 1b (Marks 1) - YOUR CODE HERE
		# raise NotImplementedError
		return np.array(np.maximum(X,0),dtype=float)
		###############################################

	def gradient_relu_of_X(self, X, delta):
		# Input
		# data : Output from next layer/input | shape: batchSize x self.out_nodes
		# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
		# Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
		# This will only be called for layers with activation relu amd during backwardpass
		###############################################
		# TASK 1e (Marks 1) - YOUR CODE HERE
		#raise NotImplementedError
		return np.multiply(delta,np.where(X <= 0, 0, 1))
		###############################################

	def softmax_of_X(self, X):
		# Input
		# data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
		# Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
		# This will only be called for layers with activation softmax
		###############################################
		# TASK 1c (Marks 3) - YOUR CODE HERE
		# raise NotImplementedError
		lst=[]
		for i in range(len(X)):
			e_x = np.exp(X[i] - np.max(X[i]))
			lst.append(e_x / e_x.sum())
		return np.asarray(lst)
		###############################################

	def gradient_softmax_of_X(self, X, delta):
		# Input
		# data : Output from next layer/input | shape: batchSize x self.out_nodes
		# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
		# Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
		# This will only be called for layers with activation softmax amd during backwardpass
		# Hint: You might need to compute Jacobian first
		###############################################
		# TASK 1f (Marks 7) - YOUR CODE HERE
		# raise NotImplementedError
		lst=[]
		for i in range(len(X)):
			lst.append(np.dot(delta[i],np.diag(X[i].flatten()) - np.outer(X[i], X[i])))
		return np.asarray(lst)
		###############################################

	def forwardpass(self, X):
		# Input
		# activations : Activations from previous layer/input | shape: batchSize x self.in_nodes
		# Returns: Activations after one forward pass through this layer | shape: batchSize x self.out_nodes
		# You may need to write different code for different activation layers
		###############################################
		# TASK 1d (Marks 4) - YOUR CODE HERE
		if self.activation == 'relu':
			# raise NotImplementedError
			#print(self.biases)
			#print(np.dot(X,self.weights))
			self.data=np.dot(X,self.weights)+self.biases
			#print(self.data)
			self.data=self.relu_of_X(self.data)
			#print(self.data.shape)
			return self.data

		elif self.activation == 'softmax':
			self.data=np.dot(X,self.weights)+self.biases
			self.data=self.softmax_of_X(self.data)
			return self.data
		
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		###############################################

	def backwardpass(self, activation_prev, delta):
		# Input
		# activation_prev : Output from next layer/input | shape: batchSize x self.out_nodes]
		# delta : del_Error/ del_activation_curr | shape: self.out_nodes
		# Output
		# new_delta : del_Error/ del_activation_prev | shape: self.in_nodes
		# You may need to write different code for different activation layers
		# Just compute and store the gradients here - do not make the actual updates
		###############################################
		# TASK 1g (Marks 6) - YOUR CODE HERE
		if self.activation == 'relu':
			inp_delta = self.gradient_relu_of_X(self.data, delta)
			self.biasesGrad=inp_delta.mean(axis=0)
			self.weightsGrad=np.dot(activation_prev.T,inp_delta)
			return np.dot(inp_delta,self.weights.T)

		elif self.activation == 'softmax':
			inp_delta=self.gradient_softmax_of_X(self.data,delta)
			self.weightsGrad = np.dot(activation_prev.T,inp_delta)
			self.biasesGrad=inp_delta.mean(axis=0)
			return np.dot(inp_delta,self.weights.T)

		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		###############################################
		
	def updateWeights(self, lr):
		# Input
		# lr: Learning rate being used
		# Output: None
		# This function should actually update the weights using the gradients computed in the backwardpass
		###############################################
		# TASK 1h (Marks 2) - YOUR CODE HERE
		# raise NotImplementedError
		#print(self.weights)
		#print(self.biases)
		self.weights=self.weights-lr*self.weightsGrad
		self.biases=self.biases-lr*self.biasesGrad
		###############################################
		