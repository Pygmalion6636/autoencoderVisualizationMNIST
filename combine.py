from PIL import Image
#opens an image:
#creates a new empty image, RGB mode, and size 400 by 400.
new_im = Image.new('RGB', (28 * 60 , 28 * 60 ))

# 168 size
#Here I resize my opened image, so it is no bigger than 100,100
counter = 0

#Iterate through a 4 by 4 grid with 100 spacing, to place my image
for i in xrange(0, 1680, 28):
    for j in xrange(0, 1680, 28):

        im = Image.open('images/' + str(counter) + ".png")
        #I change brightness of the images, just to emphasise they are unique copies.
        im.thumbnail((28, 28))
        #paste the image at location i,j:
        new_im.paste(im, (i, j))
        counter += 1

new_im.save('output.png')
