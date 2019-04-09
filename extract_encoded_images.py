import os
import shutil
import sys
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf

from dataManagement.DataHelper import DataHelper
from dataManagement.DataLoader import DataLoader
from parameterManager.ExtractionParameters import ExtractionParameters
from tensorflowWrapper.FlagsParser import FlagsParser


def embedding_to_image(root_dir, img_sum, test_img, extraction_parameters):
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
        for row in list(range(y, int(y + extraction_parameters.get_superpixel_per_col() * superpixel_h), superpixel_h)):
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


def main(args):
    num_words_to_keep = 30000
    num_words_x_doc = 100
    flags_parser = FlagsParser(delimiter=';')
    flags_parser.parse_parameter_from_file(args)
    flags = flags_parser.get_flags()

    extraction_parameters = ExtractionParameters(flags.output_image_width, flags.encoding_height, flags.ste_image_w,
                                                 flags.ste_separator_size, flags.ste_superpixel_size)

    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu_id

    batch_size = flags.batch_size
    model_dir = flags.save_model_dir_name

    data_helper = DataHelper(num_words_to_keep, flags.save_model_dir_name)

    train_path = flags.train_path
    val_path = flags.val_path
    root_dir = flags.output_dir

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)

    train_length = len(open(train_path).readlines())
    val_length = len(open(val_path).readlines())

    print('Train: ' + str(train_length))
    print('Val: ' + str(val_length))

    data_loader = DataLoader()
    data_loader.load_data(train_path, val_path, delimiter='|')

    training_data = data_loader.get_training_data()
    val_data = data_loader.get_val_data()

    data_helper.load_from_pickles()
    label_train, label_val = data_helper.preprocess_labels(training_data, val_data)
    text_train, text_val = data_helper.preprocess_texts(training_data, val_data, num_words_x_doc)

    print('\nEvaluating...\n')
    checkpoint_dir = os.path.abspath(os.path.join(model_dir, 'checkpoints'))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    print('Loading latest checkpoint: {}'.format(checkpoint_file))
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            train_dataset = tf.data.Dataset.from_tensor_slices((text_train, label_train, (training_data.get_images())))
            train_dataset = train_dataset.batch(batch_size)
            train_iterator = train_dataset.make_initializable_iterator()
            train_next_element = train_iterator.get_next()

            val_dataset = tf.data.Dataset.from_tensor_slices((text_val, label_val, (val_data.get_images())))
            val_dataset = val_dataset.batch(batch_size)
            val_iterator = val_dataset.make_initializable_iterator()
            val_next_element = val_iterator.get_next()

            sess.run(train_iterator.initializer)
            sess.run(val_iterator.initializer)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name('input_x').outputs[0]
            # input_y = graph.get_operation_by_name('input_y').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            input_mask = graph.get_operation_by_name('input_mask').outputs[0]

            # Tensors we want to evaluate
            input_y = graph.get_operation_by_name('input_y').outputs[0]
            accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]
            sum = graph.get_operation_by_name('sum').outputs[0]

            image_size = extraction_parameters.get_output_image_width()

            correct = 0
            for b in range((val_length // batch_size) + 1):
                images = []
                element = sess.run(val_next_element)
                path_list = [el.decode('UTF-8') for el in element[2]]

                for path in path_list:
                    img = cv2.imread(path)
                    img = cv2.resize(img, (image_size, image_size))
                    img = img / 255
                    images.append(img)

                feed_dict = {
                    input_x: element[0],
                    input_y: element[1],
                    input_mask: images,
                    dropout_keep_prob: 1.0
                }
                acc, img_sum = sess.run([accuracy, sum], feed_dict)

                correct += acc * len(path_list)  # batch_size
                thread = Thread(target=embedding_to_image,
                                args=(root_dir, img_sum, path_list, extraction_parameters))
                thread.start()

            test_accuracy = correct / val_length
            print('Test accuracy: {}/{}={}'.format(int(correct), val_length, test_accuracy))

            correct = 0
            for b in range((train_length // batch_size) + 1):
                images = []
                element = sess.run(train_next_element)
                path_list = [el.decode('UTF-8') for el in element[2]]

                for path in path_list:
                    img = cv2.imread(path)
                    img = cv2.resize(img, (image_size, image_size))
                    img = img / 255
                    images.append(img)

                feed_dict = {
                    input_x: element[0],
                    input_y: element[1],
                    input_mask: images,
                    dropout_keep_prob: 1.0
                }
                acc, img_sum = sess.run([accuracy, sum], feed_dict)

                correct += acc * len(path_list)  # batch_size

                thread = Thread(target=embedding_to_image, args=(root_dir, img_sum, path_list,extraction_parameters))
                thread.start()

            train_accuracy = correct / train_length
            print('Train accuracy: ' + str(train_accuracy))


if __name__ == '__main__':
    main(sys.argv[1])
