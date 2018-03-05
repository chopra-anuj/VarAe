import tensorflow as tf
import numpy as np

class DenseLayer:

    def __init__(self, m1, m2, f = tf.nn.relu):

        self.W = tf.Variable(tf.random_normal( shape = (m1, m2))*2/np.sqrt(m1))
        self.b = tf.Variable(tf.zeros(m2, tf.float32))
        self.f = f


    def forward(self,X):
        return self.f(tf.add(tf.matmul(X, self.W), self.b))