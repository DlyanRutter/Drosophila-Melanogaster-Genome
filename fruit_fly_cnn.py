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

def load_batch(path, batch_id):
    """
    load a batch of dataset
    """
    with open(path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), n_channels,
                                      pixel_height, pixel_width)).transpose(
                                          0,2,3,1)
    labels = batch['labels']
    return features, labels

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    """
    preprocess data and save it to a file
    """
    features = normalize(features)
    labels = one_hot_encode(labels)
    pickle.dump((features, labels), open(filename, 'wb'))

def preprocess_and_save(path, normalize, one_hot_encode, n_batches=5,
                        pixel_height=32, pixel_width=32, n_channels=3):
    """
    preprocesses training and validation data
    """
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, lables = load_batch(path, batch_i)
        validation_count = int(len(features) * 0.1)

        _preprocess_and_save(
            normalize,
            one_hot_encode,
            features[:-validation_count],
            labels[:-validation_count],
            'preprocess_batch_' + str(batch_i) + '.p')

        valid_features.extend(features[-validation_count:]) ###### extend?
        valid_labels.extend(labels[-validation_count:])

    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation.p')

    with open(path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    test_features = batch['data'].reshape((len(batch['data']), n_channels,
                                           pixel_height,pixel_width)).transpose(
                                               (0, 2, 3, 1))
    test_labels = batch['labels']
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_test.p')
                                               
def make_batches(features, labels, batch_size):
    """
    split features and labels into batches.
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_training_batch(batch_id, batch_size):
    """
    load preprocessed training data and return them in batches of batch_size
    or less.
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    return load_training_batch(features, labels, batch_size)    
    
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
    
def fully_conn(x_tensor, n_outputs, keep_prob=0.5):
    """
    apply a fully connected layer to x_tensor using weight and bias. x_tensor
    is a 2-D tensor where first dim is batch_size, n_outputs is the number of
    output that the new tensor should be. Returns a 2-D tensor where the
    second dim is num_outputs. (Input and output are both flattened). dropout
    is the dropout probability, and size is the batch size.
    """
    weights = tf.Variable(tf.truncated_normal(
        [x_tensor.get_shape().as_list()[1], n_outputs], dtype=tf.float32))
    bias = tf.zeros([n_outputs], dtype=tf.float32)
    fully = tf.add(tf.matmul(x_tensor, weights), bias)
    fully = tf.nn.relu(fully)
    fully = tf.nn.dropout(fully,keep_prob=keep_prob)
    return fully

def output(x_tensor, n_outputs):
    """
    Apply an output layer to x_tensor using weight and bias. x_tensor is
    a 2-D tensor where the first dimension is batch size. n_outputs is the
    number of output the new tensor should be. returns a 2-D tensor where
    the second dimension is n_outputs.
    """
    weights = tf.Variable(tf.truncated_normal(
        [x_tensor.get_shape().as_list()[1], n_outputs], dtype=tf.float32))
    bias = tf.zeros([n_outputs], dtype=tf.float32)
    return tf.add(tf.matmul(x_tensor, weights), bias)

def conv_net(img, keep_prob, num_classes=10):
    """
    create a convolutional neural network model. img is a placeholder tensor
    that holds image data. keep_prob is a placeholder tensor that holds
    dropout keep probability. num_classes is number of possible leabels.
    returns a tensor that represents logits.
    """
    x = conv2d_maxpool(img, 10, (2,2), (4,4), (2,2), (2,2))
    x = conv2d_maxpool(x, 32, (2,2), (2,2), (2,2), (2,2))
    x = conv2d_maxpool(x, 65, (2,2), (3,3), (2,2), (2,2))

    x = flatten(x)
    x = fully_conn(x, 180, keep_prob=0.75)
    x = fully_conn(x, 32, keep_prob=0.5)

    return output(x, num_classes)

def train_neural_network(session, optimizer, keep_probability, feature_batch,
                         label_batch, num_labels=10):
    """
    Optimize a single session on a batch of images and labels. session is the
    current TensorFlow session. optimizer is the TensorFlow optimizer
    function. keep_probability is the keep_probability, feature_batch is
    the batch of numpy image data. label_batch is the batch of numpy
    label data.

    use optimizer to optimize in session with a feed dict having the following:
    x for image input, y for labels, keep_prob for keep probability for
    dropout. function will be called for each batch, so
    tf.global_variables_initializer() has already been called. Nothing returned

    'x' refers to the name factor in neural_net_image_input function
    'y' refers to the name factor in neural_net_label_input function
    'keep_prob' refers to the name factor in neural_net_keep_prob_input function
    """

    session.run(optimizer, feed_dict={
        'x':feature_batch,
        'y':label_batch,
        'keep_prob':keep_probability})
    pass

def print_stats(session, feature_batch, label_batch, cost, accuracy,
                test_valid_size=258):
    """
    print loss and validation accuracy. use global variables valid_features
    and valid_labels to calculate validation accuracy. use keep probability
    of 1.0 to calculate loss and validation accuracy
    """
    loss = session.run(cost, feed_dict={
        x:feature_batch,
        y:label_batch,
        keep_prob:1.0})
    valid_acc = session.run(accuracy, feed_dict={
        x:valid_features[:test_valid_size],
        y:valid_labels[:test_valid_size],
        keep_prob:1.0})
        
    print('Epoch {:>2}, Batch {:>3} -'
        'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
        epoch + 1,
        batch + 1,
        loss,
        valid_acc))      

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

