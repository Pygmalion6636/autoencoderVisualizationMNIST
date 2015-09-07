from keras.layers import containers
from keras.layers.core import AutoEncoder, Dense, Merge, Dropout, Activation
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)


X_train = X_train.astype("float32")
X_test = X_test.astype("float32")


X_train /= 255
X_test /= 255

y_train = X_train
y_test = X_test

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

autoencoder = Sequential()
autoencoder.add(Dense(784, 196))
#autoencoder.add(Dropout(.8))
autoencoder.add(Dense(196, 784))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
RMSprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)

autoencoder.compile(loss='mean_squared_error', optimizer=adam)

autoencoder.fit(X_train, y_train, nb_epoch=5, batch_size=1)

print(len(autoencoder.layers))

for layer in autoencoder.layers:
    config = layer.get_config()
    if config.get('input_dim') == 196:
        weights = layer.get_weights() # list of numpy arrays
        break


from PIL import Image
import scipy

import matplotlib.pyplot as plt
for number, representation in enumerate(weights[0]):
    
    #plt.imshow(np.reshape(representation, (28, 28)), interpolation='nearest', cmap='binary')
    #plt.show()
    learned = np.reshape(representation, (28, 28))
    #result = Image.fromarray((learned * 255).astype(np.uint8))
    #result.save('images/' + str(number) + '.png')
    
    scipy.misc.imsave('images/' + str(number) + '.png', learned)
