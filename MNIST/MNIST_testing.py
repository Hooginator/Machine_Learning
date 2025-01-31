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
import time

# Import datasets, classifiers and performance metrics
from sklearn import datasets, linear_model, metrics

# Get loading from external dataset ability
#import pandas as pd+5
#import gzip, cPickle
import struct

# Functions  

# Test print 
def printstuff():
    maxPlot = 5
    for i in xrange(0,maxPlot):
        # Re square images2
        images[i+5].resize(28,28)
        plt.subplot(1,maxPlot,i)
        plt.imshow(images[i+5], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title( str(labels[i+5]) )
        plt.show()

def loadPreamble(fileName):
    with open(fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    
    n_dims = struct.unpack("b", fileContent[3])[0]
    dim = []
    for i in xrange(0,n_dims):
        # Big endian data, therefore you need the   >  in front of the i
        dim.append( int(struct.unpack(">i", fileContent[4+4*i:4+4*(i+1)])[0]) )

    return dim    
    
def loadImages(fileName,labelFileName,batchnumber,batchsize):
    with open(fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read() 

    with open(labelFileName, mode='rb') as file: # b is important -> binary
        labelFileContent = file.read() 

    #print str(struct.unpack("b", fileContent[2])[0]) 

    if ( struct.unpack("b", fileContent[2])[0] == 8):
        # Unsigned byte
        #print ("Unsigned byte data type detected")
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
    
    for i in xrange(batchnumber*batchsize,batchsize*(1+batchnumber)):
        #print "image # "+str(i)
        # ing this d preamblddshould getanus to thee tart if th s batch in sthe file
        temp =  (i*dim[1]*dim[2])
        temp_image = np.empty((dim[1]*dim[2]))
        for j in xrange(0,dim[1]):
            for k in xrange(0,dim[2]):
                temp_image[j*dim[2]+k] = int(struct.unpack("B", fileContent[pream+temp+j*dim[2]+k:pream+(temp+1)+j*dim[2]+k])[0])
        images.append(temp_image)
        labels.append(int(struct.unpack("B", labelFileContent[8+i])[0]))
        #print pream+temp+j*dim[1]+k
    file.close()
    
# Test print 
    return [images,labels]







############################ LEARNING #########################################

# Load training data

fileName = "train-images.idx3-ubyte"
labelFileName = "train-labels.idx1-ubyte"

dimensions = loadPreamble(fileName)

print (str(time.time()))

totalbatch = 10
batchsize = 1000
success = 0
predictor = linear_model.LogisticRegression(solver = 'newton-cg')
for batchnumber in xrange(totalbatch):
    start = time.time()
    if(batchnumber * batchsize >= dimensions[0]):
        print "Done"
        break
    #images = []
    [images,labels] = loadImages(fileName,labelFileName,batchnumber,batchsize)
    predictor.fit(images[:-1], labels[:-1])
    ans = predictor.predict(images[-1:])
    stop = time.time()
    if(ans[0] == labels[-1]):
        print "Got it: " +str(success) + " out of " + str(batchnumber) + "  " + str(float(success+1) /float(batchnumber+1))
        success += 1
    #print "Gen " + str(batchnumber) + " Pre: " +str(ans) + " Tar: " +str(labels[-1]) + " Time: " + str(stop-start)
#print len(images)
#abc =  predictor.support_vectors_
# Test print 
printstuff()

############################ TESTING ##########################################

testFile = "t10k-images.idx3-ubyte"
testLabel = "t10k-labels.idx1-ubyte"

testStart = 0
testSize = 1000
testSuccess = 0
[images,labels] = loadImages(testFile,testLabel,testStart,testSize)
for i in xrange(testStart , testStart+testSize ):    
    ans = predictor.predict([images[i]])
    if(ans[0] == labels[i]):
        testSuccess += 1
        print "Victory # " +str(testSuccess) + "  |  "+ str(float( testSuccess / (i+1.)))
#abc = open("train-images.idx3-ubyte",'rb')

#test_set = cPickle.load(abc)

#print(str(abc[0]))
