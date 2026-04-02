'''
Author: Andrew H. Fagg
'''

# Help link: https://www.tensorflow.org/api_docs/python/tf/keras/Metric
# And used HW2 code to edit off of.

import tensorflow as tf 
import tf_keras as keras


class MeanAbsoluteDeviation(keras.metrics.Metric):
    '''
    MAD = |true_label - y-pred| / count
    
    Our challenge in implementing standard Metrics is that the data to compute the metrics
    is provided incrementally (through multiple calls to update_state())
    
    For MAD, we need to keep track of:
    - Absolute Difference of the predicted and true mean of data points
    - Count of the number of data points
    
    '''
    
    def __init__(self, name='MAD', **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Initialize parameters
        self.mad = self.add_weight(name='mad', initializer='zeros', shape=[], dtype=tf.float64)
        self.count = self.add_weight(name="count", initializer="zeros",  shape=[], dtype=tf.float64)


    #def update_state(self, y_true, dist, sample_weight=None):
    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        @param y_true Expected output (shape: samples y outputs)
        @param y_pred Predicted output (shape: samples x outputs)
        @param sample_weight Weight of each sample in the performance measure (shape: samples)
        
         TODO: don't yet address sample_weight
        
        NOTE: names help with debugging
        '''
        # Ensure that the inputs are in a standard form
        y_true = tf.cast(y_true, dtype=tf.float64)
        y_pred = tf.cast(y_pred, dtype=tf.float64)

        # Do update MAD parameters
        mad = tf.reduce_mean(tf.abs(y_true - tf.reduce_mean(y_pred)))
        self.mad.assign_add(mad)
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float64))


    def result(self):
        '''
        @return MAD pure function.
        '''
        return self.mad / self.count
        

    def reset_state(self):
        '''
        Reset the state of the accumulator variables
        
        This is called between epochs and data sets
        '''
        self.mad.assign(0.0)
        self.count.assign(0.0)
