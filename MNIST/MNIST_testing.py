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
#import pandas as pd
#import gzip, cPickle
import struct

# Functions  

# Test print 
def printstuff():
    maxPlot = 5
    for i in xrange(0,maxPlot):
        plt.subplot(1,maxPlot,i)
        plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title( str(labels[i]) )
        plt.show()

def loadImages(fileName,labelFileName,startImage,totalImages):
    with open(fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read() 

    with open(labelFileName, mode='rb') as file: # b is important -> binary
        labelFileContent = file.read() 

    #print str(struct.unpack("b", fileContent[2])[0]) 

    if ( struct.unpack("b", fileContent[2])[0] == 8):
        # Unsigned byte
        print ("Unsigned byte data type detected")
        a = 1
    else:
        print("Could not find data type")
        return
        
    n_dims = struct.unpack("b", fileContent[3])[0]
    dim = []
    for i in xrange(0,n_dims):
        # Big endian data, therefore you need the   >  in front of the i
        dim.append( int(struct.unpack(">i", fileContent[4+4*i:4+4*(i+1)])[0]) )

    #value_size = 1 # size of the input read, will be 1 byte, 0-255
    images = []
    labels = []
    pream = 4+n_dims*4
    
    for i in xrange(startImage*totalImages,totalImages*(1+startImage)):
        print "image # "+str(i)
        temp =  (startImage*totalImages+i*dim[1]*dim[2])
        temp_image = np.empty((dim[1],dim[2]))
        for j in xrange(0,dim[1]):
            for k in xrange(0,dim[2]):
                temp_image[j,k] = int(struct.unpack("B", fileContent[pream+temp+j*dim[1]+k:pream+(temp+1)+j*dim[1]+k])[0])
        images.append(temp_image)
        labels.append(int(struct.unpack("B", labelFileContent[8+i])[0]))
        #print pream+temp+j*dim[1]+k
    
# Test print 
    return [images,labels]








# Load training data

fileName = "train-images.idx3-ubyte"
labelFileName = "train-labels.idx1-ubyte"


#images = []
[images,labels] = loadImages(fileName,labelFileName,0,20)


# Test print 
printstuff()

#abc = open("train-images.idx3-ubyte",'rb')

#test_set = cPickle.load(abc)

#print(str(abc[0]))
