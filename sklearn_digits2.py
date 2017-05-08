# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:03:46 2017

@author: hoog
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

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
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
#classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
#print str(n_samples / 2)
#n_training = 10
n_training = n_samples / 2
classifier.fit(data[:n_training], digits.target[:n_training])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted, digits.target[n_samples / 2:]))
for index, (image, prediction, target) in enumerate(images_and_predictions[:10]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Pred: %i, Tar: %i' % (prediction, target))

plt.show()