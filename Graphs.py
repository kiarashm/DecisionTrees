import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


gaussian = True
gamma = 0.001
h = 0.1
lamb = 1e-4

def train(K, y, reg=0):
    K += reg*np.eye(K.shape[0])
    K = inv(K)
    return np.dot(K, y)


def predict(K, a, z_index):
	sol = 0
	for i in range(K.shape[0]):
		sol += np.dot(a[i], K[i][z_index])
	return sol

def calc_error(K, a, i, Y):
	error = 0
	for i in range(200):
		val = predict(K, a, i)
		if (val * Y[i] < 0):
			error += 1
	print("ERROR IS: %f" % (float(error)/float(200)))

def generateY():
	Y = np.zeros((200, 1))
	for i in range(200):
		if i > 99:
			Y[i] = 1
		else:
			Y[i] = -1
	return Y

def model(X, z, a):
	values = []
	K = np.zeros((X.shape[0], z.shape[0]))
	for i in range(X.shape[0]):
		for j in range(z.shape[0]):
			val = (1 + np.dot(X[i].T, z[j]))**2
			if gaussian:
				K[i][j] = np.exp(-gamma * np.linalg.norm(X[i] - z[j])**2)
			else:
				K[i][j] = (1 + np.dot(X[i].T, z[j]))**2

	sol = np.sign(np.dot(K.T, a))
	return sol

def plot_points():
	theta = np.random.uniform(0, 2*np.pi, 100)
	w_1 = np.random.normal(0, 1, 100)
	w_2 = np.random.normal(0, 1, 100)
	x_vals = []
	y_vals = []
	for i in range(100):
		x_vals.append( 8*np.cos(theta[i]) + w_1[i] )
		y_vals.append( 8*np.sin(theta[i]) + w_2[i] )
	plt.plot(x_vals, y_vals, 'ro')

	v_1 = np.random.normal(0, 1, 100)
	v_2 = np.random.normal(0, 1, 100)
	plt.plot(v_1, v_2, 'bo')
	x_vals.extend(v_1)
	y_vals.extend(v_2)
	X = np.array((x_vals, y_vals)).T
	return X

def decision_boundary(X):
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# here "model" is your model's prediction (classification) function
	Z = model(X, np.c_[xx.ravel(), yy.ravel()], a) 
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
	plt.axis('off')

	# Plot also the training points
	plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
	plt.show()


X = plot_points()
Y = generateY()
K = np.zeros((X.shape[0], X.shape[0]))
for i in range(X.shape[0]):
	for j in range(X.shape[0]):
		if gaussian:
			K[i][j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j])**2)
		else:
			K[i][j] = (1 + np.dot(X[i].T, X[j]))**2

a = train(K, Y, reg = lamb)
calc_error(K, a, i, Y)
plt.show()
decision_boundary(X)