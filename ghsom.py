import tensorflow as tf
import numpy as np
import datetime
from clustering.som import SOM


class GHSOM(object):

    def __init__(self, m, n, data, tau1, tau2):

        self._m = m
        self._n = n
        self._data = data
        self._tau1 = tau1
        self._tau2 = tau2
        # self._n_iterations = abs(int(n_iterations))

        # self._graph = tf.Graph()

        # with self._graph.as_default():

        self._mqe0 = self._first_neuron(data)

        # INITIALIZE SESSION
        self._sess = tf.Session()
        # INITIALIZE VARIABLES
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)
        # print(self._sess.run(self._mqe0))

    def _first_neuron(self, input_data, weights_array=None):
        rows = len(input_data[:, 0])
        if weights_array is None:
            weights_array = tf.stack([tf.reduce_mean(tf.constant(input_data.astype(float)), 0) for i in range(rows)])
        mqe = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(
            tf.subtract(input_data.astype(float), weights_array), 2), 1)), 0)
        return mqe

    def tau1_check(self, mqe_array):
        array_len = len(mqe_array)
        total_mqe = tf.divide(tf.add_n(mqe_array), array_len)
        with tf.Session() as sess:
            print("MQE = ", sess.run(total_mqe))
            mqe_num = float(sess.run(total_mqe))
            mqe0 = float(sess.run(self._mqe0))
            if mqe_num < self._tau1 * mqe0:
                return True
            else:
                return False

    def tau2_check(self, mqe):
        # tf.cond(total_mqe < self._tau1 * self._mqe0, True, False)
        print(mqe)
        if mqe < self._tau2 * self._mqe0:
            return True
        else:
            return False

    def find_error_unit(self, mqe_array, weight_array, new_input):
        mqe_tmp = tf.Variable(0, dtype=tf.float64)
        dissimilar_weight = tf.Variable(tf.zeros([5], dtype=tf.float64))
        error_unit_location = tf.Variable(0)
        weight_array = np.array(weight_array).astype(float)
        with tf.Session() as sess:
            sess.run(dissimilar_weight.initializer)
            sess.run(mqe_tmp.initializer)
            sess.run(error_unit_location.initializer)
            for idx, (mqe, weight) in enumerate(zip(mqe_array, weight_array)):
                if sess.run(tf.greater(mqe, mqe_tmp)):
                    sess.run(mqe_tmp.assign(mqe))
                    sess.run(dissimilar_weight.assign(weight))
                    sess.run(error_unit_location.assign(idx))
            new_input_size = new_input.shape
            for idx, weight in enumerate(weight_array):
                if idx == sess.run(error_unit_location) + 1:
                    if (sess.run(error_unit_location) + 1) % new_input_size[1] != 0:
                        print("right location = ", idx, sess.run(error_unit_location))
                if idx == sess.run(error_unit_location) - 1:
                    if (sess.run(error_unit_location) - 1) % new_input_size[1] != 1:
                        print("left location = ", idx, sess.run(error_unit_location))
                if idx == sess.run(error_unit_location) + new_input_size[1]:
                    print("down location = ", idx, sess.run(error_unit_location))
                if idx == sess.run(error_unit_location) - new_input_size[1]:
                    print("up location = ", idx, sess.run(error_unit_location))

                dissimilar_tmp = tf.sqrt(tf.reduce_sum(tf.pow(
                    tf.subtract(weight, dissimilar_weight), 2), 0))
            print("diss_tmp = ", sess.run(dissimilar_tmp))

    def clustering_input_data(self, input_data, result, new_weight_array):
        array1 = []
        array2 = []
        array_row1 = []
        array3 = []
        array4 = []
        array_row2 = []
        for data, dimension in zip(input_data, result):
            if np.array_equal(dimension, np.array([0, 0])):
                array1.append(data)
            if np.array_equal(dimension, np.array([0, 1])):
                array2.append(data)
            if np.array_equal(dimension, np.array([1, 0])):
                array3.append(data)
            if np.array_equal(dimension, np.array([1, 1])):
                array4.append(data)
        final_array = []
        array_row1.append(array1)
        array_row1.append(array2)
        array_row2.append(array3)
        array_row2.append(array4)
        final_array.append(array_row1)
        final_array.append(array_row2)
        mqe_array = []
        for row_array, weights in zip(final_array, new_weight_array):
            for array in row_array:
                if len(array) != 0:
                    print("arrlen = ", len(array))
                    weights_array = tf.stack([tf.constant(np.array(weights).astype(float)) for i in range(len(array))])
                    mqe_array.append(self._first_neuron(np.array(array), weights_array))
                else:
                    mqe_array.append(tf.constant(0, dtype=tf.float64))
        print(final_array)
        return np.array(final_array), mqe_array
