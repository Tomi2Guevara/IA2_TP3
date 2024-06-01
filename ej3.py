import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur) * 2 - 1  #bias normalizado y centrado
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1  #pesos normalizados y centrados

def create_nn(top, act_f):
    """
        Crea una red neuronal basada en la topología y función de activación dadas.

        Parámetros:
        topology (list): Una lista de enteros donde cada entero representa el número de neuronas en esa capa.
                         La longitud de la lista es el número de capas en la red neuronal.
        act_f (function): La función de activación que se utilizará en la red neuronal.

        Devuelve:
        nn (list): Una lista de objetos neural_layer, cada uno representando una capa en la red neuronal.
    """
    nn = []
    for l in range(len(top) - 1):
        nn.append(neural_layer(top[l], top[l + 1], act_f[l]))
    return nn


def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
    out = [(None, X)]
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        a = neural_net[l].act_f[0](z)
        out.append((z, a))

    if train:
        deltas = []
        for l in reversed(range(0, len(neural_net))):
            z = out[l + 1][0]
            a = out[l + 1][1]

            if l == len(neural_net) - 1:
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].W

            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr

    return out[-1][1]
def dropout(X, drop_probability):
    mask = np.random.rand(*X.shape) > drop_probability
    return X * mask

def dropout_derivative(X, drop_probability):
    mask = np.random.rand(*X.shape) > drop_probability
    return mask

#cargar archivo csv
data = np.loadtxt('data2Rand.csv', delimiter=',')
x_values = data[:, 0]
y_values = data[:, 1]
n = len(x_values)
p = 1

x = np.linspace(-5, 5, 100)

#funcion de activacion
sigm = (lambda x: 1 / (1 + np.exp(-x)),
        lambda x: x * (1 - x))  #derivada de la sigmoide #derivada de la sigmoide
relu = (lambda x: np.maximum(0, x),
        lambda x: np.where(x > 0, 1, 0))  #derivada de la relu
lineal = (lambda x: x,
          lambda x: 1)  #derivada de la lineal




#capa de entrada
topology = [p, 2, 1]

#seleccionamos  un 20% de los datos para el test
# Crear un array de índices y mezclarlo
np.random.seed(42)  # Establecer la semilla del generador de números aleatorios
indices = np.arange(x_values.shape[0])
np.random.shuffle(indices)

neural_net = create_nn(topology, [relu, lineal, lineal])

# Indexar tus arrays con el array de índices mezclados
x_values_mezcla = x_values[indices]
y_values_mezcla = y_values[indices]

# Dividir los datos en entrenamiento y test
x_values_train = x_values_mezcla[:int(n * 0.90)]
y_values_train = y_values_mezcla[:int(n * 0.90)]
x_values_test = x_values_mezcla[int(n * 0.90):]
y_values_test = y_values_mezcla[int(n * 0.90):]

#redimensionar los datos de entrada
x_values_train = x_values_train.reshape(-1, 1)
y_values_train = y_values_train.reshape(-1, 1)
x_values_test = x_values_test.reshape(-1, 1)
y_values_test = y_values_test.reshape(-1, 1)

#normalizar los datos
x_values_train = x_values_train / np.max(x_values_train)
y_values_train = y_values_train / np.max(x_values_train)
x_values_test = x_values_test / np.max(x_values_train)
y_values_test = y_values_test/ np.max(x_values_train)

#funciones de costo
l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: 2 * (Yp - Yr))  #derivada del error cuadratico medio
rmse_cost = (lambda Yp, Yr: np.sqrt(np.mean((Yp - Yr) ** 2)),
             lambda Yp, Yr: (Yp - Yr) / (np.sqrt(np.mean((Yp - Yr) ** 2))))
#entrar la red neuronal
loss = []
for i in range(375):
    pY = train(neural_net, x_values_train, y_values_train, l2_cost, 0.0001)
    if i % 25 == 0:
        loss = np.append(loss, l2_cost[0](pY, y_values_train))
        # print('Epoch: ', i, 'Error: ', l2_cost[0](pY, y_values_train))
        plt.scatter(x_values_train, pY, color='red')
        plt.scatter(x_values_train, y_values_train, color='blue')
        plt.show()


#testear la red neuronal
outTest = train(neural_net, x_values_test, y_values_test, rmse_cost, 0.0001, False)
# plt.plot(loss)
plt.scatter(x_values_test, outTest, color='red')
plt.scatter(x_values_test, y_values_test, color='blue')
plt.show()


xvline = x = 6 * np.random.rand(100, 1) - 3
xvline = xvline.reshape(-1, 1)
xvline_transf = xvline / np.max(x_values_train)

outGraph = train(neural_net, xvline_transf, xvline_transf, l2_cost, 0.01, False)
outGraph = outGraph * np.max(x_values_train)
plt.scatter(xvline, outGraph, color='red')
# plt.scatter(x_values, y_values, color='blue')
plt.show()
