# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab inline
import numpy as np
import opengm

# <codecell>

# setup
shape=[5,5]
numVar=shape[0]*shape[1]
# img=numpy.random.rand(shape[0],shape[1])*255.0
img = numpy.array([[17.48596584, 230.76685609, 181.72909067, 111.1831536, 176.8002497 ],
                  [104.37091012, 46.74352918, 80.81955776, 122.71055103, 188.70500536],
                  [229.43452333, 248.52203584, 149.5458772, 193.32430752, 218.22665255],
                  [170.84325185, 174.5779461, 241.18400579, 218.1852195, 181.60840245],
                  [119.69538607, 39.63332029, 201.27305304, 217.06898652, 184.71828862]])

# <codecell>

print unaries.shape

# <codecell>

## try using the method in the example  
# unaries  
img1d=img.reshape(numVar)  
lrange=numpy.arange(0,256,1)  
unaries=numpy.repeat(lrange[:,numpy.newaxis], numVar, 1).T  

for l in xrange(256):  
    unaries[:,l]-=img1d  

unaries=numpy.abs(unaries)
unaries=unaries.reshape(shape+[256])
   
# higher order function  
def regularizer(labels):
  val=abs(float(labels[0])-float(labels[1]))
  return val*0.4  

print "generate 2d grid gm"
regularizer=opengm.PythonFunction(function=regularizer,shape=[256,256])  
gm=opengm.grid2d2Order(unaries,regularizer=regularizer)  


icm=opengm.inference.Icm(gm)
icm.infer()  
arg=icm.arg()  
arg=arg.reshape(shape)  

figure(0)  
print imshow(numpy.round(img), interpolation='nearest')  
figure(1)  
print imshow(arg, interpolation='nearest')  
print gm 

# <codecell>

class PyCallback(object):
    """
    callback functor which will be passed to an inference
    visitor.
    In that way, pure python code can be injected into the c++ inference.
    This functor visualizes the labeling as an image during inference.

    Args :
        shape : shape of the image 
        numLabels : number of labels
    """
    def __init__(self,shape,numLabels):
        self.shape=shape
        self.numLabels=numLabels
        matplotlib.interactive(True)
    def begin(self,inference):
        """
        this function is called from c++ when inference is started

        Args : 
            inference : python wrapped c++ solver which is passed from c++
        """
        print "begin"
    def end(self,inference):
        """
        this function is called from c++ when inference ends

        Args : 
            inference : python wrapped c++ solver which is passed from c++
        """
        print "end"
    def visit(self,inference):
        """
        this function is called from c++ each time the visitor is called

        Args : 
            inference : python wrapped c++ solver which is passed from c++
        """
        
        arg = inference.arg()
        gm  = inference.gm()
#         print "energy ",gm.evaluate(arg)

        arg=arg.reshape(self.shape)*255
#         plt.imshow(arg.T, cmap='gray',interpolation="nearest")
#         plt.draw()

# <codecell>

print regularizer

# <codecell>

## try with generic gm

# init gm
numLabels = 2
gm2 = opengm.gm(numpy.ones(numVar,dtype=opengm.label_type)*numLabels)

# deal with unary costs

# unaries  
img1d=img.reshape(numVar)  
lrange=numpy.arange(0,256,1)  
unaries2=numpy.repeat(lrange[:,numpy.newaxis], numVar, 1).T  

for l in xrange(256):  
    unaries2[:,l]-=img1d  

unaries2=numpy.abs(unaries2)

# add functions
fids = gm2.addFunctions(unaries2)
# add first order factors
gm2.addFactors(fids, arange(0, numVar, 1))

# deal with pairwise costs

# higher order function  
def regularizer(labels):
    val=abs(float(labels[0])-float(labels[1]))
#     print labels[0]
    return val#*0.4  

print "generate 2d grid gm"
regularizer=opengm.PythonFunction(function=regularizer,shape=[256,256])
print "la",regularizer
# add function
fid = gm2.addFunction(regularizer)
# add second order factors
vis2Order=opengm.secondOrderGridVis(shape[0],shape[1],True)
gm2.addFactors(fid, vis2Order)


icm2=opengm.inference.Icm(gm2)
# icm2=opengm.inference.BeliefPropagation(gm2,parameter=opengm.InfParam())
icm2.setStartingPoint(icm2.arg())
# set up visitor
# callback=PyCallback(shape,numLabels)
# visitor=icm2.pythonVisitor(callback,visitNth=1)
icm2.infer()#visitor)
# icm2.infer()  
arg2=icm2.arg()  
arg2=arg2.reshape(shape)  

figure(0)  
print imshow(numpy.round(img), interpolation='nearest')  
figure(1)
print imshow(arg2, interpolation='nearest')
figure(2)
print imshow(arg, interpolation='nearest')    
print gm2

# <codecell>

print np.array(vis2Order)

# <codecell>

## try with generic gm and separate function for both unaries and pairwise

# init gm
numLabels = 256
gm3 = opengm.gm(numpy.ones(numVar,dtype=opengm.label_type)*numLabels)

# deal with unary costs

def unaryCost(label):
    val=abs(float(label[0]))
    
    return val*1.0 

unaryCost=opengm.PythonFunction(function=unaryCost,shape=[256])

# unaries  
img1d=img.reshape(numVar)  
lrange=numpy.arange(0,256,1)  
unaries3=numpy.repeat(lrange[:,numpy.newaxis], numVar, 1).T  

for l in xrange(256):  
    unaries3[:,l]-=img1d  

unaries3=numpy.abs(unaries3)

# add function
fid = gm3.addFunction(unaryCost)
# add first order factors
gm3.addFactors(fid, arange(0, numVar, 1))

# deal with pairwise costs

# higher order function  
def regularizer(labels):
    val=abs(float(labels[0])-float(labels[1]))
    print val
    return val*0.4

regularizer=opengm.PythonFunction(function=regularizer,shape=[256,256])

# add function
fid = gm3.addFunction(regularizer)
# add second order factors
vis2Order=opengm.secondOrderGridVis(shape[0],shape[1],True)
gm3.addFactors(fid, vis2Order)


icm3=opengm.inference.Icm(gm3)  
icm3.infer()
arg3=icm3.arg()  
arg3=arg3.reshape(shape)  

figure(0)  
print imshow(numpy.round(img), interpolation='nearest')  
figure(1)
print imshow(arg3, interpolation='nearest') 
print gm3
print img

# <codecell>

print numpy.array(vis2Order).shape

