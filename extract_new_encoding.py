import math
import os
import shutil
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf
from tflearn.data_utils import pad_sequences

from BestDataHelperEverCreated import BestDataHelperEverCreated
from dataManagement.DataLoader import DataLoader

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("patience", 100, "Stop criteria (default: 100)")
tf.flags.DEFINE_integer("output_image_width", 100, "Size of output Image plus embedding  (default: 100)")
tf.flags.DEFINE_integer("encoding_height", 10, "Height of the output embedding  (default: 10)")

tf.flags.DEFINE_integer("ste_image_w", 256, "width of the output image for embedding and image  (default: 256)")
tf.flags.DEFINE_integer("ste_separator_size", 4, "blank space around the visual embedding (default: 4)")
tf.flags.DEFINE_integer("ste_superpixel_size", 4, "size of the superpixel size (default: 4)")

tf.flags.DEFINE_string("train_path", "/home/superior/tmp/test-accuracy/train.csv",
                       "csv file containing text|class|image_path")
tf.flags.DEFINE_string("val_path", "/home/superior/tmp/test-accuracy/test.csv",
                       "csv file containing text|class|image_path")
tf.flags.DEFINE_string("save_model_dir_name", "/home/superior/tmp/test-accuracy/food101-100-10",
                       "dir used to save the model")
tf.flags.DEFINE_string("output_dir", "/home/super/tmp/new_encoding/100x100-10", "dir used to save the new dataset")

tf.flags.DEFINE_string("gpu_id", "", "ID of the GPU to be used")

FLAGS = tf.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

num_features = FLAGS.output_image_width * FLAGS.encoding_height
image_w = image_h = FLAGS.ste_image_w
separator_size = FLAGS.ste_separator_size
superpixel_w = superpixel_h = FLAGS.ste_superpixel_size
superpixels_per_row = (image_w - 2 * separator_size) / superpixel_w
superpixels_per_col = math.ceil(num_features / superpixels_per_row)


def embedding_to_image(root_dir, img_sum, test_img):
    for image, path in zip(img_sum, test_img):
        # Load a color image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # skip empty images
        if img is None:
            return
        img = cv2.resize(img, (image_w, image_h))
        # create blank_image
        # img = np.zeros((image_h,image_w,3), np.uint8)

        x = separator_size
        y = separator_size

        assert separator_size + superpixels_per_row * superpixel_w <= image_w, 'the image width is smaller than the visual word width'

        text_encoding_crop = image[0:FLAGS.encoding_height, 0:FLAGS.output_image_width, :]
        word_features = np.reshape(text_encoding_crop, (
                FLAGS.output_image_width * FLAGS.encoding_height * 3))  # C-like index ordering

        sp_i = 0  # superpixel index

        # write the embedding
        for row in list(range(y, int(y + superpixels_per_col * superpixel_h), superpixel_h)):
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

        dir_names = path.split('/')[-3:]
        full_path = os.path.join(root_dir,
                                 os.path.join(dir_names[0], dir_names[1], dir_names[2].replace('.jpg', '.png')))
        parent_dir = os.path.abspath(os.path.join(full_path, os.pardir))

        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        cv2.imwrite(full_path, img)


def write_to_disk(root_dir, img_sum, test_img, save_features=False):
    for image, path in zip(img_sum, test_img):
        dir_names = path.split('/')[-3:]
        full_path = os.path.join(root_dir,
                                 os.path.join(dir_names[0], dir_names[1], dir_names[2].replace('.jpg', '.png')))
        parent_dir = os.path.abspath(os.path.join(full_path, os.pardir))

        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        cv2.imwrite(full_path, image * 255)

        if save_features:
            type_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))  # train or test
            type_dir_split = os.path.split(type_dir)
            out_file_name = os.path.join(type_dir_split[0], type_dir_split[1] + ".csv")
            text_encoding_crop = image[0:FLAGS.encoding_height, 0:FLAGS.output_image_width, :]
            text_encoding_crop = np.reshape(text_encoding_crop, (
                    FLAGS.output_image_width * FLAGS.encoding_height * 3))  # C-like index ordering
            with open(out_file_name, "a") as myfile:
                myfile.write("{};{}\n".format(path, np.array2string(text_encoding_crop, threshold=np.inf,
                                                                    max_line_width=np.inf, separator=',').replace('\n',
                                                                                                                  '')))


def main(argv=None):
    num_words_to_keep = 30000
    num_words_x_doc = 100

    image_size = FLAGS.output_image_width

    batch_size = FLAGS.batch_size
    model_dir = FLAGS.save_model_dir_name  # 'runs/1530104868'

    data_helper = BestDataHelperEverCreated(num_words_to_keep, FLAGS.save_model_dir_name)

    train_path = FLAGS.train_path
    val_path = FLAGS.val_path
    root_dir = FLAGS.output_dir

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)

    train_length = len(open(train_path).readlines())
    val_length = len(open(val_path).readlines())

    print("Train: " + str(train_length))
    print("Val: " + str(val_length))

    data_loader = DataLoader()
    data_loader.load_data(train_path, val_path, delimiter='|')

    training_data = data_loader.get_training_data()
    val_data = data_loader.get_val_data()

    data_helper.load_from_pickles()
    label_train = data_helper.labels_to_one_hot(training_data.get_labels())
    label_val = data_helper.labels_to_one_hot(val_data.get_labels())

    text_train = data_helper.texts_to_indices(training_data.get_texts())
    text_val = data_helper.texts_to_indices(val_data.get_texts())

    text_train = pad_sequences(text_train, maxlen=num_words_x_doc, value=0.)
    text_val = pad_sequences(text_val, maxlen=num_words_x_doc, value=0.)

    print("\nEvaluating...\n")
    checkpoint_dir = os.path.abspath(os.path.join(model_dir, "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    print("Loading latest checkpoint: {}".format(checkpoint_file))
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
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
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            input_mask = graph.get_operation_by_name("input_mask").outputs[0]

            # Tensors we want to evaluate
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
            sum = graph.get_operation_by_name("sum").outputs[0]

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
                thread = Thread(target=embedding_to_image, args=(root_dir, img_sum, path_list))

                thread.start()

            test_accuracy = correct / val_length
            print("Test accuracy: {}/{}={}".format(correct, val_length, test_accuracy))

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

                thread = Thread(target=embedding_to_image, args=(root_dir, img_sum, path_list))
                thread.start()

            train_accuracy = correct / train_length
            print("Train accuracy: " + str(train_accuracy))


if __name__ == '__main__':
    main()
