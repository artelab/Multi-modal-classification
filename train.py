import datetime
import os
import shutil

import cv2
import tensorflow as tf
from tflearn.data_utils import pad_sequences

from BestDataHelperEverCreated import BestDataHelperEverCreated
from TextImgCNN import TextImgCNN
from dataManagement.DataLoader import DataLoader

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("patience", 100, "Stop criteria (default: 100)")
tf.flags.DEFINE_integer("output_image_width", 100, "Size of output Image plus embedding  (default: 100)")
tf.flags.DEFINE_integer("encoding_height", 10, "Height of the output embedding  (default: 10)")

tf.flags.DEFINE_string("train_path", "/home/superior/tmp/test-accuracy/train.csv",
                       "csv file containing text|class|image_path")
tf.flags.DEFINE_string("val_path", "/home/superior/tmp/test-accuracy/test.csv",
                       "csv file containing text|class|image_path")
tf.flags.DEFINE_string("save_model_dir_name", "/home/superior/tmp/test-accuracy/food101-100-10",
                       "dir used to save the model")

tf.flags.DEFINE_string("gpu_id", "0", "ID of the GPU to be used")

FLAGS = tf.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id


def train(x_train, y_train, img_train, x_test, y_test, img_test, words_to_keep, output_image_width, encoding_height,
          patience_init_val):
    def train_step(x_batch, y_batch, images_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.input_mask: images_batch,
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_step_only_accuracy(x_batch, y_batch, images_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.input_mask: images_batch,
            cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
        return accuracy

    best_accuracy = 0
    patience = patience_init_val

    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        with sess.as_default():

            cnn = TextImgCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=words_to_keep,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                output_image_width=output_image_width,
                encoding_height=encoding_height,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, img_train))
            dataset = dataset.batch(FLAGS.batch_size)
            train_iterator = dataset.make_initializable_iterator()
            next_element = train_iterator.get_next()

            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test, img_test))
            test_dataset = test_dataset.batch(FLAGS.batch_size)
            test_iterator = test_dataset.make_initializable_iterator()
            next_test_element = test_iterator.get_next()

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            out_dir = FLAGS.save_model_dir_name
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)

            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and test_accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("test_accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            with open(os.path.join(out_dir, "results.txt"), "a") as resfile:
                resfile.write("Model dir: {}\n".format(out_dir))
                resfile.write("Dataset: {}\n".format(img_train[0]))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            train_length = len(x_train)
            val_length = len(x_test)

            for ep in range(FLAGS.num_epochs):
                print("***** Epoch " + str(ep) + " *****")
                sess.run(train_iterator.initializer)

                for b in range((train_length // FLAGS.batch_size) + 1):
                    images_batch = []
                    element = sess.run(next_element)

                    path_list = [el.decode('UTF-8') for el in element[2]]

                    for path in path_list:
                        # print("image path: " + path)
                        img = cv2.imread(path)
                        img = cv2.resize(img, (output_image_width, output_image_width))
                        img = img / 255
                        images_batch.append(img)

                    train_step(element[0], element[1], images_batch)

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        sess.run(test_iterator.initializer)

                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        # Run one pass over the validation dataset.
                        sess.run(test_iterator.initializer)
                        correct = 0
                        for b in range((val_length // FLAGS.batch_size) + 1):
                            test_img_batch = []
                            test_element = sess.run(next_test_element)

                            test_path_list = [el.decode('UTF-8') for el in test_element[2]]

                            for path in test_path_list:
                                img = cv2.imread(path)
                                img = cv2.resize(img, (output_image_width, output_image_width))
                                img = img / 255
                                test_img_batch.append(img)

                            acc = dev_step_only_accuracy(test_element[0], test_element[1], test_img_batch)
                            correct += acc * len(test_path_list)
                        test_accuracy = correct / val_length
                        print("Test accuracy: " + str(test_accuracy) +
                              ", best accuracy: " + str(best_accuracy) +
                              ", patience: " + str(patience))

                        if test_accuracy > best_accuracy:
                            best_accuracy = test_accuracy
                            patience = patience_init_val
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                        else:
                            patience -= 1

                        print("Test accuracy: " + str(test_accuracy) + ", best accuracy: " + str(
                            best_accuracy) + ", patience: " + str(patience) + "\n")

                        with open(os.path.join(out_dir, "results.txt"), "a") as resfile:
                            resfile.write("epoch: %d, step: %d, test acc: %f, best acc: %f, patience: %d\n" % (
                                ep, current_step, test_accuracy, best_accuracy, patience))

                        if patience == 0:
                            return


def main(argv=None):
    num_words_to_keep = 30000
    num_words_x_doc = 100

    patience = FLAGS.patience

    output_image_width = FLAGS.output_image_width
    encoding_height = FLAGS.encoding_height

    data_helper = BestDataHelperEverCreated(num_words_to_keep, FLAGS.save_model_dir_name)

    train_path = FLAGS.train_path
    val_path = FLAGS.val_path

    data_loader = DataLoader()
    data_loader.load_data(train_path, val_path, delimiter='|', shuffle_data=True)

    train_texts, train_labels, train_images = data_loader.get_training_data()
    val_texts, val_labels, val_images = data_loader.get_val_data()

    data_helper.train_one_hot_encoder(train_labels)
    train_y = data_helper.encode_to_one_hot(train_labels)
    test_y = data_helper.encode_to_one_hot(val_labels)

    data_helper.train_tokenizer(train_texts)

    train_x = data_helper.convert_to_indices(train_texts)
    test_x = data_helper.convert_to_indices(val_texts)

    train_x = pad_sequences(train_x, maxlen=num_words_x_doc, value=0.)
    test_x = pad_sequences(test_x, maxlen=num_words_x_doc, value=0.)

    train(train_x, train_y, train_images, test_x, test_y, val_images, num_words_to_keep, output_image_width,
          encoding_height, patience)

    data_helper.pickle_models_to_disk()


if __name__ == '__main__':
    tf.app.run()
