import numpy as np
import csv
import random
import math
import pandas as pd
from abc import ABC, abstractmethod

class Module(ABC):
    '''
    Base class for all neural network modules.
    Child classes must define abstract forward and backward methods.
    ''' 
    def __init__(self):
        self.input = []
        
    def __call__(self,x):
        self.input = x
        return self.forward(x)
    
    @abstractmethod
    def forward(self,x):
        '''Empty forward pass method to be defined in child class'''
        pass
    
    @abstractmethod
    def backward(self,x):
        '''Empty backprop method to be defined in child class'''
        pass

class MultiLayerPerceptron(Module):
    def __init__(self,
                 input_size,
                 batch_size,
                 num_classes,
                 num_nodes,
                 cost_function,
                 learning_rate=0.001):
        
        # inherit from superclass
        super().__init__()
        
        # learning rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cost_function = cost_function
        
        # define network layers
        self.network = []
        self.network.append(LinearLayer(input_size,num_nodes,batch_size))
        self.network.append(Sigmoid())
        self.network.append(LinearLayer(num_nodes,num_nodes,batch_size))
        self.network.append(Sigmoid())
        self.network.append(LinearLayer(num_nodes,num_classes,batch_size))
        self.network.append(SoftMax())
        
    def forward(self,x):
        
        for layer in self.network:
            x = layer(x)
        
        return x
    
    def backward(self,grad):
        
        # loop through network layers backwards
        for layer in self.network[::-1]:
            
            # compute gradient and pass along
            try:
                if isinstance(layer,LinearLayer):
                    grad = layer.backward(grad,self.learning_rate)
                else:
                    grad = layer.backward(grad)
            except Exception as e:
                print(f'Error computing gradient for {layer} layer')
                break
    
    def train(self,x,y,num_epochs=100,verbose=False):
            
        # loop through training epochs
        loss_by_epoch = []
        for _ in range(num_epochs):
            
            # split data into batches
            np.random.shuffle(x)
            if self.batch_size > 1:
                batch_data = np.array_split(x,x.shape[0]/self.batch_size)

            # loop through batches of data
            for batch in batch_data:

                # forward pass
                preds = self.forward(batch)
                
                # calculate loss
                loss = self.cost_function(y,preds)
                loss_by_epoch.append(loss.mean())
                
                # backward pass
                self.backward(self.cost_function.backward())
                
        return loss_by_epoch
    
    def test(self,x,y,verbose=False):
        
        # forward pass
        preds = self.forward(x)

        # calculate accuracy
        accuracy = np.all(y==np.round(preds),axis=1).sum()/x.shape[0]

        return accuracy
    
    def __str__(self):
        string = ''
        for i,layer in enumerate(self.network):
            
            string += ' '*((30-len(str(layer)))//2)
            string += str(layer) 
            if i < len(self.network)-1:
                string += '\n'+(' '*15)+u'\u2193\n'
        return string
             
    
class LinearLayer(Module):
    '''
    Fully connected linear layer
    '''
    def __init__(self,input_size,output_size,batch_size):
        
        # initialize weight matrix
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(input_size,output_size)
        self.bias = np.random.randn(1,output_size)
        
    def __str__(self):
        return f'Linear {self.input_size}x{self.output_size}'
        
    def forward(self,x):
        return np.matmul(x,self.W)+self.bias
    
    def backward(self,grad,lr):
        grad_out = np.matmul(grad,self.W.T)
        self.update(grad,lr)
        return grad_out
    
    def update(self,grad,lr):
        self.W -= lr * np.matmul(self.input.T,grad)
        self.bias -= lr * grad.sum(axis = 0)
    
class Sigmoid(Module):
    '''
    Sigmoid activation layer
    '''
    def __init__(self):
        super().__init__()
        self.value = []
        
    def __str__(self):
        return 'Sigmoid'
    
    def forward(self,x):
        self.value = 1/(1+np.exp(-x))
        return self.value
    
    def backward(self,grad):
        return grad*(self.value*(1-self.value))
    
class SoftMax(Module):
    '''
    Softmax activation layer
    '''
    def __init__(self):
        super().__init__()
        self.value = []
        
    def __str__(self):
        return 'Softmax'
    
    def forward(self,x):
        x -= np.max(x,axis=1,keepdims=True)
        self.value = np.exp(x) / np.exp(x).sum(axis=1,keepdims=True)
        return self.value
    
    def backward(self,grad):
        return grad*(self.value*(1-self.value))
    
class CrossEntropy(Module):
    '''
    Cross entropy loss function
    '''
    def __init__(self):
        super().__init__()
        self.y = []
        self.y_hat = []
        
    def __str__(self):
        return f'Cross Entropy Loss'
    
    def __call__(self,y,y_hat):
        self.y = y
        self.y_hat = y_hat
        return self.forward(y,y_hat)
    
    def forward(self,y,y_hat):
        return np.sum(-y*np.log(y_hat),axis=1)/self.y.shape[0]
    
    def backward(self):
        return ((self.y_hat-self.y)/(self.y_hat*(1-self.y_hat)))/self.y.shape[0]
            
        
    
data = np.random.randn(8,10)
y = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0]])
    
testnet = MultiLayerPerceptron(input_size=10, 
                               batch_size=8, 
                               num_classes=5, 
                               num_nodes=20,
                               cost_function=CrossEntropy(),
                               learning_rate=0.001)
out = testnet(data)
cost_fn = CrossEntropy()
cost = cost_fn(y,out)
grad = cost_fn.backward()

testnet.train(data,y)


# sig = Sigmoid()
# sm = SoftMax()
# lr = 0.01
# x = np.random.randn(1,3)
# y = np.array([0,1])
# W1 = np.random.randn(3,4)
# b1 = 0.1
# W2 = np.random.randn(4,2)
# b2 = 0.1

# # forward
# o1 = np.matmul(x,W1)+b1
# x2 = sig(o1)
# o2 = np.matmul(x2,W2)+b2
# y_hat = sm(o2)

# # backward
# dLdo2 = y_hat - y
# do2dW2 = x2
# do2dx2 = W2
# dLdW2 = np.matmul(x2.T,dLdo2)
# W2 -= dLdW2*lr
# dLdx2 = dLdo2 do2dx2
# dx2o1 = sig(o1)*(1-sig(o1))
# dLdo1 = 1