from keras.models import Sequential
from keras.layers.core import Dense, Activation, AutoEncoder
from keras.optimizers import SGD, Adagrad, Adadelta, Adam, RMSprop
from keras.datasets import mnist
import numpy as np

try:
    xrange
except NameError:
    xrange = range


hiddenUnits = 400
stackSize = 1
batchSize = 256
epochs = 20


(X_train,), (X_test,) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255
X_test /= 255


model = Sequential()

for i in xrange(stackSize):
	encoder = Sequential()
	encoder.add(Dense(X_train.shape[1], hiddenUnits))
	encoder.add(Activation('tanh'))

	decoder = Sequential()
	decoder.add(Dense(hiddenUnits, X_train.shape[1]))
	decoder.add(Activation('tanh'))

	model.add(AutoEncoder(encoder=encoder, decoder=decoder))

#opt = SGD(lr=1)
#opt = Adagrad()
#opt = Adadelta()
#opt = Adam()
opt = RMSprop()
model.compile(loss='mean_squared_error', optimizer=opt)

model.fit(X_train, X_train, nb_epoch=epochs, batch_size=batchSize, validation_data=(X_test, X_test), show_accuracy=True)

(trainscore, trainaccuracy) = model.evaluate(X_train, X_train, batch_size=256, show_accuracy=True)
(testscore, testaccuracy) = model.evaluate(X_test, X_test, batch_size=256, show_accuracy=True)

print("Training Score: " + str(trainscore))
print("Training Accuracy: " + str(trainaccuracy))
print("Test Score: " + str(testscore))
print("Test Accuracy: " + str(testaccuracy))