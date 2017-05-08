# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/hoog/.spyder2/.temp.py
"""

from sklearn import datasets, linear_model, svm
import matplotlib
import numpy as np


digits = datasets.load_digits()

# Set up support vector network
clf = svm.SVC(gamma=0.001, C=100.)

# learn all data in digits except the last one ( :-1 )
clf.fit(digits.data[:-1], digits.target[:-1])

# predict last number and print output
print clf.predict(digits.data[-1:])

# show image of last number
imshow(digits.images[-1],cmap='Greys',interpolation='nearest')



# TESTING AND GRAPHONG STUFFS
#print ( digits.data )
#print ( digits.target )

#print (digits.data[0].size)
#temp_data = np.reshape(digits.data[4], (8, 8))
#print ( temp_data.ndim )

#plot(digits.data[0])
#imshow(temp_data,cmap='Greys',interpolation='nearest')

#imshow(digits.images[0],cmap='Greys',interpolation='nearest')
#show()
