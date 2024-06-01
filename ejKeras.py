#en la capa de entrada, el parámetro más importante es la cantidad de entradas (features) que tiene cada dato 1()
#en la capa oculta, la cantidad de neuronas
#en la capa de salida, la cantidad de salidas posibles
import keras

## 1
#investigar sequential()
x_input = keras.layers.Input(shape=x.shape[1]) #le pasamos la dimensión del tensor de entrada
# Primera capa oculta
x = keras.layers.Dense(20, activation='relu')(x_input) #20 neuronas en la capa oculta (esto devuelve un tensor)
# Segunda capa oculta
x = keras.layers.Dense(10, activation='relu')(x) #10 neuronas en la capa oculta

x_output = keras.layers.Dense(3, activation='softmax')(x) #3 neurona en la capa de salida, softmax para clasificación

model = keras.models.Model(inputs=x_input, outputs=x_output) # creamos el modelo, se puedn tener varios tensores de entrada y salida

#separar a x en entranamiento y validación

x_train, y_train =

batch_size = 10 #x_train.shape[0] #tamaño del batch

red_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
earlyStop = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy']) #compilamos el modelo

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=100, validation_split=0.2, callbacks=[red_lr,earlyStop]) #entrenamos el modelo

#plotear el accuracy y la loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

#evitar el overfitting
#1. Regularización
#2. Dropout

#1. Regularización (L2)
regular = keras.regularizers.l2(0.01)

#2. Dropout
drop = keras.layers.Dropout(0.5)

#
# Proceso de creación de redes neuronales
# Separación de datos
# acondicionamiento y normalización de datos
# Definimos Entradas, salidas y capas ocultas
# Inicializar W1
# Inicializar b1
# x*W1 + b1
# Función de activación
# Inicializar W2
# Inicializar b2
# x*W2 + b2
# Función de activación
# definir función de error
# back propagation

#Investigar session (para hacer salas de entrenamiento de los parametros de la red)