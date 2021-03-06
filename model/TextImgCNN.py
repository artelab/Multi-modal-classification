import tensorflow as tf
import tflearn


class TextImgCNN(object):

    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, output_image_width, encoding_height,
                 l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_mask = tf.placeholder(tf.float32, [None, output_image_width, output_image_width, 3], name='input_mask')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # resize images
        # size: [new_height, new_width] is The new size for the images.
        resized_images = tf.image.resize_images(self.input_mask, [output_image_width - encoding_height, output_image_width])

        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope('embedding'):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name='W')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope('fc1-text'):
            fc1_size = encoding_height * output_image_width * 3
            fc_w1 = tf.Variable(tf.random_normal([num_filters_total, fc1_size]))
            bd1 = tf.Variable(tf.random_normal([fc1_size]))
            fc1 = tf.add(tf.matmul(self.h_pool_flat, fc_w1), bd1)
            fc1 = tf.nn.sigmoid(fc1)

        self.reshaped_layer = tflearn.reshape(fc1, new_shape=[-1, encoding_height, output_image_width, 3], name='encoded_text')

        self.sum = tf.concat([self.reshaped_layer, resized_images], 1, name='sum')
        self.h_pool_flat_2 = tf.reshape(self.sum, [-1, output_image_width * output_image_width * 3])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat_2, self.dropout_keep_prob)

        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[output_image_width * output_image_width * 3, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

