import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing

# Load data from CSV file

training_data = pd.read_csv('numerai_training_data.csv', header=0)
validation_data = pd.read_csv('numerai_tournament_data.csv', header=0)

# Transform the loaded CSV data into numpy arrays
features = [f for f in list(training_data) if "feature" in f]
X = training_data[features]
Y = training_data["target"]
x_validation = validation_data[features]
ids = validation_data["id"]

def svm_sgd_plot(x_train, y_train):
	# Initialize our SVMs weight vector with zeros (3 values)
	w = np.zeros(len(x_train[0]))
	# The learning rate
	eta = 1
	# How many iterations to train for
	epochs = 100000
	# Store misclassifications so we can plot how they change over time
	errors = []

	# Training part, gradient descent part
	for epoch in range(1, epochs):
		error = 0
		for i, x in enumerate(x_train):


		# Misclassification
			if (y_train[i]*np.dot(x_train[i], w)) < 1:

			# Misclassified update for ours weights
				w = w + eta * ((x_train[i] * y_train[i]) + (-2 * (1/epoch)* w))
				error = 1

			else:
			# Correct classification, update our weights
				w = w + eta * (-2 *(1/epoch)* w)

		errors.append(error)

	# Let's plot the rate of classification errors during training for our SVM
	plt.plot(errors, '|')
	plt.ylim(0, 1)
	plt.axes().set_yticklabels([])
	plt.xlabel('Epoch')
	plt.ylabel('Misclassified')
	plt.show()

	return w

w = svm_sgd_plot(X, Y)
# Lets see how it generalizes on the data
# for d, sample in enumerate(X):

# 	# Plot the negative samples
# 	if d < 2:
# 		plt.scatter(sample[0], sample[1], s=120, marker='*', linewidths=2)

# 	# Plot the positive samples
# 	else:
# 		plt.scatter(sample[0],sample[1], s=120, marker='.', linewidths=2)

# # Add our test samples
# plt.scatter(2, 2, s=120, marker='*', linewidths=2, color='yellow')
# plt.scatter(4, 3, s=120, marker='.', linewidths=2, color='blue')

# # Print the hyperplane calculated by svm_sdg()
# x2=[w[0], w[1], -w[1], w[0]]
# x3=[w[0], w[1], w[1],-w[0]]

# x2x3 = np.array([x2, x3])
# X,Y,U,V = zip(*x2x3)
# ax = plt.gca()
# ax.quiver(X,Y,U,V, scale=1, color='blue')