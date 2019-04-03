import datetime
import os
import shutil

import cv2
import tensorflow as tf

from TextImgCNN import TextImgCNN


class ModelTrainer(object):

    def __init__(self, train_dataset, val_dataset, training_parameters, model_parameters):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_parameters = training_parameters
        self.model_parameters = model_parameters

    def train(self):
        def train_step(x_batch, y_batch, images_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_mask: images_batch,
                cnn.dropout_keep_prob: self.training_parameters.get_dropout_keep_probability()
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
        patience = self.model_parameters.get_patience()

        with tf.Graph().as_default():
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            with sess.as_default():
                output_width = self.training_parameters.get_output_image_width()

                cnn = TextImgCNN(
                    sequence_length=self.train_dataset.get_texts().shape[1],
                    num_classes=self.train_dataset.get_labels().shape[1],
                    vocab_size=self.training_parameters.get_no_of_words_to_keep(),
                    embedding_size=self.training_parameters.get_embedding_dim(),
                    filter_sizes=list(map(int, self.training_parameters.get_filter_sizes().split(","))),
                    num_filters=self.training_parameters.get_num_filters(),
                    output_image_width=output_width,
                    encoding_height=self.training_parameters.get_encoding_height(),
                    l2_reg_lambda=0.0)

                x_train = self.train_dataset.get_texts()
                y_train = self.train_dataset.get_labels()
                img_train = self.train_dataset.get_images()

                x_test = self.val_dataset.get_texts()
                y_test = self.val_dataset.get_labels()
                img_test = self.val_dataset.get_images()

                dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, img_train))
                dataset = dataset.batch(self.training_parameters.get_batch_size())
                train_iterator = dataset.make_initializable_iterator()
                next_element = train_iterator.get_next()

                test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test, img_test))
                test_dataset = test_dataset.batch(self.training_parameters.get_batch_size())
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

                out_dir = self.model_parameters.get_model_directory()
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
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

                with open(os.path.join(out_dir, "results.txt"), "a") as resfile:
                    resfile.write("Model dir: {}\n".format(out_dir))
                    resfile.write("Dataset: {}\n".format(img_train[0]))

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                train_length = len(x_train)
                val_length = len(x_test)

                for ep in range(self.model_parameters.get_no_of_epochs()):
                    print("***** Epoch " + str(ep) + " *****")
                    sess.run(train_iterator.initializer)

                    for b in range((train_length // self.training_parameters.get_batch_size()) + 1):
                        images_batch = []
                        element = sess.run(next_element)

                        path_list = [el.decode('UTF-8') for el in element[2]]

                        for path in path_list:
                            img = cv2.imread(path)
                            img = cv2.resize(img, (output_width, output_width))
                            img = img / 255
                            images_batch.append(img)

                        train_step(element[0], element[1], images_batch)

                        current_step = tf.train.global_step(sess, global_step)
                        if current_step % self.model_parameters.evaluate_every == 0:
                            sess.run(test_iterator.initializer)

                        if current_step % self.model_parameters.evaluate_every == 0:
                            print("\nEvaluation:")
                            # Run one pass over the validation dataset.
                            sess.run(test_iterator.initializer)
                            correct = 0
                            for b in range((val_length // self.training_parameters.get_batch_size()) + 1):
                                test_img_batch = []
                                test_element = sess.run(next_test_element)

                                test_path_list = [el.decode('UTF-8') for el in test_element[2]]

                                for path in test_path_list:
                                    img = cv2.imread(path)
                                    img = cv2.resize(img,
                                                     (output_width,
                                                      output_width))
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
                                patience = self.model_parameters.get_patience()
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
