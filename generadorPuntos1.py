import numpy as np

import matplotlib.pyplot as plt
# Definir el tamaño de la matriz y los coeficientes de la función cuadrática
n_rows = 500
a, b, c = 1, 2, 3

# Generar valores aleatorios para x entre -3 y 3
x = 6 * np.random.rand(n_rows, 1) - 3

# Calcular los valores de y usando la función cuadrática y añadir un error aleatorio
y = a * x**2 + b * x + c + np.random.randn(n_rows, 1)

# Concatenar x e y para formar la matriz de datos
data = np.hstack((x, y))


# Extraer los valores de x e y de la matriz de datos
x_values = data[:, 0]
y_values = data[:, 1]

# Crear un gráfico de dispersión
plt.scatter(x_values, y_values)

# Establecer los títulos de los ejes
plt.xlabel('x')
plt.ylabel('y')

# Mostrar el gráfico
plt.show()

# Guardar la matriz de datos en un archivo CSV
np.savetxt('data2Rand.csv', data, delimiter=',')