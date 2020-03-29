import numpy as np
import csv
import random
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import time
import matplotlib.pyplot as plt

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
        
        # loop through network layers and run forward method
        for layer in self.network:
            x = layer(x)
            
        # return final layer output
        return x
    
    def backward(self,grad):
        
        # loop through network layers backwards
        for layer in self.network[::-1]:
            
            # compute gradient and pass along
            if isinstance(layer,LinearLayer):
                grad = layer.backward(grad,self.learning_rate)
            else:
                grad = layer.backward(grad)
    
    def train(self,x,y,val_split=0.2,num_epochs=100,verbose=False,output_metrics=False):
        
        # one hot encode y
        onehot_y = OneHotEncode(y)
        
        # split data in train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(x, 
                                                          onehot_y, 
                                                          test_size=val_split)
          
        # make sure training data is a numpy array
        if isinstance(x,pd.DataFrame):
            X_train = X_train.to_numpy()
            X_val = X_val.to_numpy()
        
        # loop through training epochs
        start = time.time()
        all_train_loss = []
        all_val_loss = []
        all_train_acc = []
        all_val_acc = []
        for epoch in range(num_epochs):
            
            # create training batches
            train_batch_data = self.create_batches(X_train,y_train)

            # loop through batches of training data
            train_losses = []
            train_accs = []
            for batch in train_batch_data:
                
                # break up x and y
                batch_x = batch[:,:x.shape[1]]
                batch_y = batch[:,x.shape[1]:]

                # forward pass
                preds = self.forward(batch_x)
                
                # calculate loss
                train_loss = self.cost_function(batch_y,preds)
                train_losses.append(train_loss)
                
                # calculate accuracy
                train_acc = self.calculate_accuracy(batch_y,preds)
                train_accs.append(train_acc)
                
                # backward pass
                self.backward(self.cost_function.backward())
           
            # add epoch loss and acc to running list
            all_train_loss.append(np.mean(train_losses))
            all_train_acc.append(np.mean(train_accs))
                
            # create training batches
            val_batch_data = self.create_batches(X_val,y_val)

            # loop through batches of training data
            val_losses = [0]
            val_accs = []
            for batch in val_batch_data:
                
                # break up x and y
                batch_x = batch[:,:x.shape[1]]
                batch_y = batch[:,x.shape[1]:]

                # forward pass
                preds = self.forward(batch_x)
                
                # calculate loss
                val_loss = self.cost_function(batch_y,preds)
                val_losses.append(np.mean(val_loss))
                
                # calculate accuracy
                val_acc = self.calculate_accuracy(batch_y,preds)
                val_accs.append(val_acc)
                
            # add epoch loss and acc to running list  
            all_val_loss.append(np.mean(val_losses))
            all_val_acc.append(np.mean(val_accs))
            
            if verbose and (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}: \
                       \nTrain loss = {np.mean(train_losses)} \
                       \nTrain accuracy = {np.mean(train_accs)} \
                       \nValidation loss = {np.mean(val_losses)} \
                       \nValidation accuracy = {np.mean(val_accs)}\n')
                
         
        train_time = np.round((time.time()-start)/60,decimals=2)
        print(f'Model training for {num_epochs} epochs completed in {train_time} minutes')
        print(f'Final train loss = {all_train_loss[-1]} \
              \nFinal train accuracy = {all_train_acc[-1]} \
              \nFinal validation loss = {all_val_loss[-1]} \
              \nFinal validation accuracy = {all_val_acc[-1]}')
        
        if output_metrics:
            return all_train_loss, all_train_acc, all_val_loss, all_val_acc
    
    def create_batches(self,x,y):
        
        # concatenate x and y
        data = np.hstack((x,y))
        
        # shuffle data
        np.random.shuffle(data)
            
        # split data into batches
        batch_data = np.array_split(data,data.shape[0]/self.batch_size)
        
        return batch_data
              
    def calculate_accuracy(self,y,y_hat):
        
        return np.all(y==np.round(y_hat),axis=1).sum()/y.shape[0]      
    
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
    
def OneHotEncode(y):
    labels = np.unique(y)
    encoded_y = np.zeros((y.shape[0],len(labels)))
    
    lbl_dict = {lbl:i for i,lbl in enumerate(labels)}
    
    for i,label in enumerate(y):
        encoded_y[i,lbl_dict[label]] = 1
        
    return encoded_y
        
    
# command line test for algorithm
if __name__=='__main__':

    # load MNIST train set
    train = pd.read_csv('./MNIST/train.csv')

    # split up x and y
    y_train = train.label
    X_train = train.drop('label',axis=1)

    # initialize model
    testnet = MultiLayerPerceptron(input_size=X_train.shape[1], 
                                   batch_size=64, 
                                   num_classes=len(y_train.unique()), 
                                   num_nodes=500,
                                   cost_function=CrossEntropy(),
                                   learning_rate=0.001)

    # train model
    train_loss, train_acc, val_loss, val_acc = testnet.train(X_train,y_train,
                                                             num_epochs=80,
                                                             verbose=True,
                                                             output_metrics=True) 
   
    # plot training loss
    plt.plot(train_loss,color='blue',label='Train')
    plt.plot(val_loss,color='orange',label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # plot training accuracy
    plt.plot(train_acc,color='blue',label='Train')
    plt.plot(val_acc,color='orange',label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
