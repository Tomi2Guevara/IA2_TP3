import sklearn as sk
from sklearn import neural_network
from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the images
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))


lr = 0.001 #learning rate
nn = [20,10,15,10,4,2]

#creamos el objeto del modelo
model = sk.neural_network.MLPClassifier(hidden_layer_sizes=nn, max_iter=1000, learning_rate_init=lr, verbose=True, tol=1e-4, early_stopping=False, n_iter_no_change=15,
                                        activation='relu', solver='sgd')
model.fit(X_train, y_train)
predict = model.predict(X_test)
print("Accuracy: ", model.score(X_test, y_test))
#93.2% en 300 iteraciones

