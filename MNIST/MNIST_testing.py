# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:24:03 2017

@author: hoog
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# Get loading from external dataset ability
import pandas as pd
import gzip, cPickle, struct

# The digits dataset

fileName = "train-images.idx3-ubyte"

with open(fileName, mode='rb') as file: # b is important -> binary
    fileContent = file.read() 

print str(struct.unpack("b", fileContent[2])[0]) 

if ( struct.unpack("b", fileContent[2])[0] == 8):
    # Unsigned byte
    print ("Unsigned byte data type detected")
    a = 1
else:
    print("Could not find data type")



n_dims = struct.unpack("b", fileContent[3])[0]

if ( n_dims == 3):
    # such 3D tensor
    print ("3-tensor detected")
else:
    print("Could not find number of dimensions")
dim = []

for i in xrange(0,n_dims):
    # Big endian data, therefore you need the   >  in front of the i
    dim.append( int(struct.unpack(">i", fileContent[4+4*i:4+4*(i+1)])[0]) )

print (str(dim))
print(str(dim[0]*dim[1]*dim[2]))

cycle_size = 10
cycle_start = 0
#value_size = 1 # size of the input read, will be 1 byte, 0-255
images = []

pream = 4+n_dims*4

for i in xrange(cycle_start*cycle_size,cycle_size*(1+cycle_start)):
    temp =  (cycle_start*cycle_size+i*dim[1]*dim[2])
    temp_image = np.empty((dim[1],dim[2]))
    for j in xrange(0,dim[1]):
        for k in xrange(0,dim[2]):
            temp_image[j,k] = int(struct.unpack("B", fileContent[pream+temp+j*dim[1]+k:pream+(temp+1)+j*dim[1]+k])[0])
    images.append(temp_image)
    #print images[i]

plt.imshow(images[6], cmap=plt.cm.gray_r, interpolation='nearest')
print images[6]
plt.show()

#abc = open("train-images.idx3-ubyte",'rb')

#test_set = cPickle.load(abc)

#print(str(abc[0]))







# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
#images_and_labels = list(zip(digits.images, digits.target))
#for index, (image, label) in enumerate(images_and_labels[:10]):
#    print str(index)
#    plt.subplot(2, 5, index + 1)
#    plt.axis('off')
#    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
#n_samples = len(digits.images)
#data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
#classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
#classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
#print str(n_samples / 2)
#n_training = 10
#n_training = n_samples / 2
#classifier.fit(data[:n_training], digits.target[:n_training])

# Now predict the value of the digit on the second half:
#expected = digits.target[n_samples / 2:]
#predicted = classifier.predict(data[n_samples / 2:])

#print("Classification report for classifier %s:\n%s\n"
#      % (classifier, metrics.classification_report(expected, predicted)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

#images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted, digits.target[n_samples / 2:]))
#for index, (image, prediction, target) in enumerate(images_and_predictions[:10]):
#    plt.subplot(2, 5, index + 1)
#    plt.axis('off')
#    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#    plt.title('Pred: %i, Tar: %i' % (prediction, target))

#plt.show()