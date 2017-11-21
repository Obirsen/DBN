import tensorflow as tf
import numpy as np
from rbm_train import rbm
from dbn_linear import dbn_linear
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
'''
define dbn
'''
class dbn():
    def __init__(self,size,train_step):
        self.size=size
        self.train_step=train_step
        self.rbm_model=[]

    def rbm_layer(self,shape):
        for i in xrange(self.size):
            self.rbm_model.append(rbm(shape[i]))

    def dbn_train(self,data,label):
        for i in xrange(self.size):
            if i==0:
                x,y=self.rbm_model[i].train(data,label,self.train_step,i)
            else:
                x,y=self.rbm_model[i].train(x,label,self.train_step,i)

            if i==self.size-1:
                return x,y


def main():
    data,label = mnist.train.next_batch(100000)#input data
    dbn_model=dbn(2,1000)#design a dbn which include two rbm layer
    shape=np.array([[784,500],[500,100]])#the shape of each rbm layer
    dbn_model.rbm_layer(shape)
    x,y=dbn_model.dbn_train(data,label)#train dbn
    linear_model=dbn_linear([100,10])
    linear_model.train(x,y,1000)#train linear layer

if __name__=='__main__':
    main()



