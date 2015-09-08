# FOR PYTHON 3
# Yoni + Conner

from keras.models import Sequential
from numpy import reshape, amin, amax, vectorize
from PIL import Image
from math import ceil

def visualize(model, layerNumber, neuronShape, imagesPerRow, filename="layerVisualization"):
		if not isinstance(model, Sequential):
			raise Exception("model must be an instance of keras.models.Sequential")
		if not isinstance(layerNumber, int) or layerNumber < 0:
			raise Exception("layerNumber must be an integer that is at least 0")
		if not isinstance(neuronShape, tuple) or len(neuronShape) != 2:
			raise Exception("neuronShape must be a 2-length tuple")
		if not isinstance(imagesPerRow, int) or imagesPerRow <= 0:
			raise Exception("imagesPerRow must be an integer that is at least 1")
		if not isinstance(filename, str):
			raise Exception("filename must be a string")

		weights = model.layers[layerNumber].get_weights() # list of numpy arrays

		# Scale image colors so that the darkest becomes black and the lightest becomes white
		minVal = amin(weights[0])
		maxVal = amax(weights[0])
		scaler = vectorize(lambda i: ((i - minVal) * (255 / (maxVal - minVal))))

		subImages = []
		for representation in weights[0]:
			subImages.append(Image.fromarray(reshape(scaler(representation), neuronShape).astype('float32')).convert(mode='L'))

		imagesPerColumn = int(ceil(len(subImages) / imagesPerRow))
                print("The number of imagesPerColumn is:", imagesPerColumn)

		image = Image.new('L', (imagesPerRow * neuronShape[0] + imagesPerRow - 1, imagesPerColumn * neuronShape[1] + imagesPerColumn - 1))

		xOffset = 0
		yOffset = 0

		for i, subImage in enumerate(subImages):
			(y, x) = divmod(i, imagesPerRow)

			xOffset += 1
			if x == 0:
				xOffset = 0
				yOffset += 1

			x *= neuronShape[0]
			y *= neuronShape[1]
			x += xOffset
			y += yOffset
			# Change brightness of the images, just to emphasise they are unique copies
			subImage.thumbnail(neuronShape)
			# Add the sub-image to the final image
			image.paste(subImage, (x, y))

		image.save(filename + ".png")
		image.show()
