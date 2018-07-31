import tensorflow as tf
import tflearn


def init_network(X_inputs, Y_inputs):
    # reset underlying graph data
    tf.reset_default_graph()
    # Build neural network
    net = tflearn.input_data(shape=[None, len(X_inputs[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(Y_inputs[0]), activation='softmax')
    net = tflearn.regression(net)

    return net
