import os

import tensorflow as tf

from accuracy.Accuracy import Accuracy
from imageManipulation.ImageManipulation import ImageManipulator
from logger.FileLogger import FileLogger
from model.TextImgCNN import TextImgCNN
from patience.Patience import Patience
from result.PartialResult import PartialResult
from result.TrainingResult import TrainingResult
from tensorflowWrapper.CustomIterator import CustomIterator
from tensorflowWrapper.FeedDictCreator import FeedDictCreator
from tensorflowWrapper.ModelTensor import ModelTensor
from view.View import View


class ModelTrainer(object):

    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.view = View()

    def train(self, training_params, model_params):

        patience = Patience(model_params.get_patience())
        best_accuracy = Accuracy(0)

        output_width = training_params.get_output_image_width()

        image_resizer = ImageManipulator(output_width)

        with tf.Graph().as_default():
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            with sess.as_default():

                cnn = TextImgCNN(
                    sequence_length=self.train_dataset.get_texts().shape[1],
                    num_classes=self.train_dataset.get_labels().shape[1],
                    vocab_size=training_params.get_no_of_words_to_keep(),
                    embedding_size=training_params.get_embedding_dim(),
                    filter_sizes=list(map(int, training_params.get_filter_sizes().split(','))),
                    num_filters=training_params.get_num_filters(),
                    output_image_width=output_width,
                    encoding_height=training_params.get_encoding_height(),
                    l2_reg_lambda=0.0)

                train_iterator, next_train_batch = CustomIterator.create_iterator(self.train_dataset,
                                                                                  training_params.get_batch_size())
                test_iterator, next_test_element = CustomIterator.create_iterator(self.val_dataset,
                                                                                  training_params.get_batch_size())

                global_step = tf.Variable(0, name='global_step', trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                        sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                out_dir = model_params.get_model_directory()

                self.view.print_to_screen('Writing to {}\n'.format(out_dir))

                # Summaries for loss and test_accuracy
                loss_summary = tf.summary.scalar('loss', cnn.loss)
                acc_summary = tf.summary.scalar('test_accuracy', cnn.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

                file_logger = FileLogger(os.path.join(out_dir, 'result.txt'))
                file_logger.write_header(out_dir, self.train_dataset.get_images()[0].split('/')[4])

                sess.run(tf.global_variables_initializer())

                train_length = len(self.train_dataset.get_texts())

                no_of_training_batches = (train_length // training_params.get_batch_size()) + 1

                input_tensor = ModelTensor(cnn.input_x, cnn.input_y, cnn.input_mask, cnn.dropout_keep_prob)

                for epoch in range(model_params.get_no_of_epochs()):
                    sess.run(train_iterator.initializer)

                    for i in range(no_of_training_batches):
                        train_batch = sess.run(next_train_batch)

                        train_images_batch = image_resizer.preprocess_images(train_batch[2])

                        feed_dict = FeedDictCreator.create_feed_dict(input_tensor, train_batch, train_images_batch,
                                                                     training_params.get_dropout_keep_probability())

                        _, step, summaries, loss, accuracy = sess.run(
                            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)

                        train_summary_writer.add_summary(summaries, step)
                        current_step = tf.train.global_step(sess, global_step)

                        training_result = TrainingResult(step, loss, accuracy)

                        self.view.print_to_screen(str(training_result))

                        if current_step % model_params.evaluate_every == 0:
                            self.view.print_to_screen('Evaluation:')

                            val_length = len(self.val_dataset.get_texts())
                            no_of_val_batches = (val_length // training_params.get_batch_size()) + 1

                            sess.run(test_iterator.initializer)
                            correct = 0

                            for i in range(no_of_val_batches):
                                test_batch = sess.run(next_test_element)
                                test_images_batch = image_resizer.preprocess_images(test_batch[2])

                                feed_dict = FeedDictCreator.create_feed_dict(input_tensor, test_batch,
                                                                             test_images_batch, 1)

                                step, summaries, loss, accuracy = sess.run(
                                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)

                                correct += accuracy * len(test_images_batch)

                            test_accuracy = Accuracy(correct / val_length)

                            partial_result = PartialResult(epoch, current_step, test_accuracy, best_accuracy, patience)
                            self.view.print_to_screen(str(partial_result))
                            file_logger.write_partial_result_to_file(partial_result)

                            if test_accuracy > best_accuracy:
                                best_accuracy.set_value(test_accuracy.get_value())
                                patience.reset_patience()
                                path = self.store_model(model_params, current_step, sess, saver)
                                self.view.print_to_screen('Saved model checkpoint to {}\n'.format(path))
                            else:
                                patience.decrement_patience()

                        if patience.is_zero():
                            return

    @staticmethod
    def store_model(model_params, current_step, sess, saver):
        checkpoint_dir = os.path.abspath(
            os.path.join(model_params.get_model_directory(), 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        return saver.save(sess, checkpoint_prefix, global_step=current_step)
