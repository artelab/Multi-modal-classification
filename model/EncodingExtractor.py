import os
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf

from imageManipulation.ImageManipulation import ImageManipulator
from tensorflowWrapper.CustomIterator import CustomIterator
from tensorflowWrapper.FeedDictCreator import FeedDictCreator
from tensorflowWrapper.ModelTensor import ModelTensor
from view.View import View


class EncodingExtractor(object):

    def __init__(self, train_dataset, val_dataset, root_dir, model_dir):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.root_dir = root_dir
        self.model_dir = model_dir
        self.view = View()

    def extract(self, extraction_parameters):
        checkpoint_dir = os.path.abspath(os.path.join(self.model_dir, 'checkpoints'))
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        batch_size = extraction_parameters.get_batch_size()

        image_resizer = ImageManipulator(extraction_parameters.get_output_image_width())

        val_length = len(self.val_dataset.get_texts())
        no_of_val_batches = (val_length // batch_size) + 1

        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            with sess.as_default():
                self.view.print_to_screen('Loading latest checkpoint: {}'.format(checkpoint_file))
                saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                train_iterator, train_next_element = CustomIterator.create_iterator(self.train_dataset, batch_size)
                val_iterator, val_next_element = CustomIterator.create_iterator(self.val_dataset, batch_size)

                sess.run(train_iterator.initializer)
                sess.run(val_iterator.initializer)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name('input_x').outputs[0]
                dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
                input_mask = graph.get_operation_by_name('input_mask').outputs[0]

                # Tensors we want to evaluate
                input_y = graph.get_operation_by_name('input_y').outputs[0]
                accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]
                sum = graph.get_operation_by_name('sum').outputs[0]

                input_tensor = ModelTensor(input_x, input_y, input_mask, dropout_keep_prob)

                correct = 0
                for b in range(no_of_val_batches):
                    val_batch = sess.run(val_next_element)
                    path_list = [el.decode('UTF-8') for el in val_batch[2]]

                    test_images_batch = image_resizer.preprocess_images(val_batch[2])
                    feed_dict = FeedDictCreator.create_feed_dict(input_tensor, val_batch, test_images_batch, 1)

                    acc, img_sum = sess.run([accuracy, sum], feed_dict)

                    correct += acc * len(path_list)  # batch_size
                    thread = Thread(target=self.embedding_to_image,
                                    args=(self.root_dir, img_sum, path_list, extraction_parameters))
                    thread.start()

                test_accuracy = correct / val_length
                self.view.print_to_screen('Test accuracy: {} / {} = {}'.format(int(correct), val_length, test_accuracy))

                train_length = len(self.train_dataset.get_texts())
                no_of_train_batches = (train_length // batch_size) + 1

                correct = 0
                for b in range(no_of_train_batches):
                    train_batch = sess.run(train_next_element)
                    path_list = [el.decode('UTF-8') for el in train_batch[2]]

                    train_images_batch = image_resizer.preprocess_images(train_batch[2])
                    feed_dict = FeedDictCreator.create_feed_dict(input_tensor, train_batch, train_images_batch, 1)

                    acc, img_sum = sess.run([accuracy, sum], feed_dict)

                    correct += acc * len(path_list)  # batch_size
                    thread = Thread(target=self.embedding_to_image,
                                    args=(self.root_dir, img_sum, path_list, extraction_parameters))
                    thread.start()

                train_accuracy = correct / train_length
                self.view.print_to_screen('Train accuracy: ' + str(train_accuracy))

    def embedding_to_image(self, root_dir, img_sum, test_img, extraction_parameters):
        x = extraction_parameters.get_separator_size()
        y = extraction_parameters.get_separator_size()
        encoding_height = extraction_parameters.get_encoding_height()
        superpixels_per_row = extraction_parameters.get_superpixel_per_row()
        superpixel_w = extraction_parameters.get_superpixel_w()
        output_image_width = extraction_parameters.get_output_image_width()
        superpixel_h = extraction_parameters.get_superpixel_h()

        for image, path in zip(img_sum, test_img):
            dir_names = path.split('/')[-3:]
            full_path = os.path.join(root_dir,
                                     os.path.join(dir_names[0], dir_names[1], dir_names[2].replace('.jpg', '.png')))
            parent_dir = os.path.abspath(os.path.join(full_path, os.pardir))

            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            # skip empty images
            if img is None:
                continue

            img = cv2.resize(img, (extraction_parameters.get_image_w(), extraction_parameters.get_image_h()))

            assert extraction_parameters.get_separator_size() + superpixels_per_row * superpixel_w \
                   <= extraction_parameters.get_image_w(), 'the image width is smaller than the visual word width'

            text_encoding_crop = image[0:encoding_height, 0:output_image_width, :]
            word_features = np.reshape(text_encoding_crop,
                                       (output_image_width * encoding_height * 3))  # C-like index ordering

            sp_i = 0  # superpixel index

            # write the embedding
            for row in list(
                    range(y, int(y + extraction_parameters.get_superpixel_per_col() * superpixel_h), superpixel_h)):
                spw = 0  # counter for superpixels in row
                for col in list(range(x, int(x + superpixels_per_row * superpixel_w), superpixel_w)):
                    ptl = sp_i * 3
                    ptr = (sp_i + 1) * 3
                    bgr = word_features[ptl:ptr]
                    if len(bgr) == 0:
                        break
                    elif len(bgr) < 3:
                        c = bgr.copy()
                        c.resize(1, 3)
                        bgr = c

                    row_start = row
                    row_end = row + superpixel_w
                    for srow in range(row_start, row_end):
                        for scol in range(col, col + superpixel_h):
                            img[srow, scol] = bgr * 255

                    sp_i += 1
                    spw += 1

            cv2.imwrite(full_path, img)
