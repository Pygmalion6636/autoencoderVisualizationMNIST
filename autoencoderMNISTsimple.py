from keras.models import Sequential
from keras.layers.core import Dense, Activation, AutoEncoder
from keras.optimizers import SGD, Adagrad, Adadelta, Adam, RMSprop
from keras.datasets import mnist
import numpy as np

from ImageWeightsVisualizer import visualize
import math

try:
    xrange
except NameError:
    xrange = range


hiddenUnits = 196
batchSize = 256
epochs = 5


(X_train, junk), (X_test, junk) = mnist.load_data()


layerNumber = 1
neuronShape = (X_train.shape[1], X_train.shape[2])
imagesPerRow = 14


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255
X_test /= 255


model = Sequential()
model.add(Dense(X_train.shape[1], hiddenUnits))
#model.add(Dropout(.8))
model.add(Dense(hiddenUnits, X_train.shape[1]))

opt = RMSprop()
model.compile(loss='mean_squared_error', optimizer=opt)

model.fit(X_train, X_train, nb_epoch=epochs, batch_size=batchSize, validation_data=(X_test, X_test), show_accuracy=True)


(trainscore, trainaccuracy) = model.evaluate(X_train, X_train, batch_size=256, show_accuracy=True)
(testscore, testaccuracy) = model.evaluate(X_test, X_test, batch_size=256, show_accuracy=True)

print("Training Score: " + str(trainscore))
print("Training Accuracy: " + str(trainaccuracy))
print("Test Score: " + str(testscore))
print("Test Accuracy: " + str(testaccuracy))

visualize(model, layerNumber, neuronShape, imagesPerRow)