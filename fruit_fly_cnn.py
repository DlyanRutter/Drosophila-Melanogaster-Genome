import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
import fruit_fly_tests, math

def normalize(arr, fourD=True, color=True):
    """
    normalize an array. if color 4D=False array should be of shape (:,:,3). else
    if color and 4D=True, array should be shape (:,:,:,3). if color=False and
    4D=True, array should be shape (:,:,:,1). if color=False and 4D=False, array
    should be shape (:,:,1). return normalized data
    """
    
    if fourD == True:
        i = 0
        count, rows, columns, colors = arr.shape
        array = np.zeros((rows, columns, colors), dtype=np.float32)
        matrix = np.zeros((count, rows, columns,colors), dtype=np.float32)
        
        while i < count:
            array = arr[i][:,:,:]
            array = ((255.0-array)/255.0).astype(np.float32)
            matrix[i] = array[:,:,:]
            i+=1
            
    else:
        matrix = ((255.0-arr)/255.0).astype(np.float32)

    return matrix

def _one_hot_encode(labels):
    """
    function for one hot encoding. checks if labels are numerical or not.
    if not, make them numerical. then one hot encodes labels into a numpy
    array of one hot vectors. columns are in alphabetical order e.g. if
    input list is ['dog', 'ant', 'cat', 'dog'], order will be ant = column 0
    cat = column 1, dog = column2
    """
    number = True
    for e in labels:
        if type(e) not in [np.float128, np.float64, np.float32, np.float16,\
                                np.int16, np.int32, np.int64, np.int8, int,
                           float]:
            number = False
            
    if number == False:
        cols = enumerate(sorted([y for x,y in enumerate(set(labels))]))
        cols = dict((i,c) for (i,c) in cols)
        inverse = dict((c,i) for (i,c) in cols.items())
       
        num_rows, num_cols = len(labels), len(set(labels))
        arr = np.zeros((num_rows, num_cols), dtype=np.float32)
        arr_list = []
        
        for (e,f) in enumerate(labels):
            if f in cols.values():
                arr[e][inverse[f]] = 1
                arr_list.append(list(arr[e]))
        one_hot = np.array(arr_list, dtype=np.float32)
                
    else:
        labels = np.array(labels)
        one_hot = np.zeros((labels.size, labels.max() + 1))
        one_hot[np.arange(labels.size), labels] = 1        
    return one_hot

def one_hot_encode(labelz):
    """
    one hot encodes labels and returns numpy array using _one_hot helper fn.
    """
    return _one_hot_encode(labelz)

def one_hot_sklearn(labels):
    """
    one hot encodes using sklearn. works for numerical and non-numerical labels.
    """
    values = np.array(labels)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def one_hot_sklearn1(labels):
    lb = LabelBinarizer()
    return lb.fit_transform(np.array(labels))
    #return lb.transform(labels)

def neural_net_image_input(img_shape):
    """
    returns a tf placeholder with shape = image shape and batch size = None
    name the placeholder "x" using name parameter in tf.placeholder
    """
    return tf.placeholder(tf.float32, shape=\
                          [None,img_shape[0],img_shape[1],img_shape[2],],\
                           name='x')

def neural_net_label_input(n_classes):
    """
    returns a tf placeholder with shape = n_classes and batch size = None
    name the placeholder "y" using name parameter in tf.placeholder
    """
    return tf.placeholder(tf.float32, shape=[None, n_classes], name='y')

def neural_net_keep_prob_input():
    """
    return a tf placeholder for dropout keep probability. name the placeholder
    as "keep_prob" using name parameter in tf.placeholder
    """
    return tf.placeholder(tf.float32, name="keep_prob")

def conv2d_maxpool(x, conv_outputs, conv_k, conv_strides, pool_k,\
                   pool_strides, padding='SAME', kernel_strides_batch=1,
                   kernel_strides_channel=1, pool_strides_batch=1,
                   pool_strides_channel=1, pool_ksize_batch=1,
                   pool_ksize_channel=1):
    """
    create weight and bias using conv_k, conv_outputs, and shape of x. add bias.
    add nonlinear activation to conv. apply max pooling using pool_k and
    pool_strides. x should have shape [None, img_height, img_width, n_channels].
    
    apply convolution then max pooling to x. params: x=TF tensor,
    conv_outputs=# of outputs (depth) for conv layer. conv_k=kernal size (2-D
    tuple) for conv layer. conv_strides = stride 2D tuple for conv layer.
    pool_k=kernal size 2D tuple for pool. pool_strides=stride 2D tuple for pool.
    returns a tensor that represents conv and max pooling of x tensor.

    kernel_strides_batch is the number of batches used in each kernel stride,
    kernel_strides_channel is the channel used in each kernel stride,
    pool_strides_batch is the number of batches used in each pool stride
    pool_strides_channel is the channel used in each kernel stride.
    pool_ksize_batch is the number of batches per kernel,
    pool_ksize_channel is the channel of the kernel
    padding can be 'SAME' or 'VALID'
    """
    shape = x.get_shape().as_list()
    in_height, in_width = shape[1], shape[2]

    if padding=='SAME':
        out_height = math.ceil(float(in_height)/float(conv_strides[0]))
        out_width = math.ceil(float(in_width)/float(conv_strides[1]))
    else:
        out_height = math.ceil(float(\
            in_height-filter_height+1)/float(conv_strides[1]))
        out_width = math.ceil(float(\
            in_width-filter_width+1)/float(conv_strides[2]))
    in_depth = shape[3]
    out_depth = conv_outputs
        
    filter_weights = tf.Variable(tf.truncated_normal((
        conv_k[0],conv_k[1],in_depth,conv_outputs),dtype=tf.float32))
    filter_bias = tf.Variable(tf.zeros(conv_outputs), dtype=tf.float32)
    conv = tf.nn.conv2d(x, filter_weights,\
                        strides=[kernel_strides_batch, conv_strides[0],
                                 conv_strides[1], kernel_strides_channel],\
                        padding=padding)
    conv_layer = tf.nn.bias_add(conv,filter_bias)
    conv_logits = tf.nn.relu(conv_layer)

    with_pooling = tf.nn.max_pool(
        conv_logits,
        ksize = [pool_ksize_batch, pool_k[0], pool_k[1], pool_ksize_channel],
        strides = [pool_strides_batch, pool_strides[0], pool_strides[1],
                   pool_strides_channel],
        padding = padding)

    return with_pooling

def flatten(x_tensor):
    """
    flatten x_tensor from a 4D tensor to a 2D tensor. e.g. (batch_size,height..) to
    to (batch_size, flattened). flatten comes from height*width*depth
    """
    shape = x_tensor.get_shape().as_list()
    return tf.reshape(x_tensor, [-1, shape[1] * shape[2] * shape[3]])
    
                        

#labelz = ['yes', 'no', 'yes', 'no', 'yes', 'maybe']
#labelz.reverse()
#print one_hot_encode(labelz)
#print type(labelz)
labels = [10, 15, 1, 11, 14]
print _one_hot_encode(labels)
#print one_hot_sklearn(labelz)

fruit_fly_tests.test_normalize(normalize)
#fruit_fly_tests.test_one_hot_encode(one_hot_encode)
fruit_fly_tests.test_nn_image_inputs(neural_net_image_input)
fruit_fly_tests.test_nn_label_inputs(neural_net_label_input)
fruit_fly_tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
fruit_fly_tests.test_con_pool(conv2d_maxpool)
fruit_fly_tests.test_flatten(flatten)
