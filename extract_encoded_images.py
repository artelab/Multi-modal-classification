import os
import shutil
import sys

from dataManagement.DataHelper import DataHelper
from dataManagement.DataLoader import DataLoader
from model.EncodingExtractor import EncodingExtractor
from parameterManager.ExtractionParameters import ExtractionParameters
from tensorflowWrapper.FlagsParser import FlagsParser


def main(args):
    num_words_to_keep = 30000
    num_words_x_doc = 100
    flags_parser = FlagsParser(delimiter=';')
    flags_parser.parse_parameter_from_file(args)
    flags = flags_parser.get_flags()

    extraction_parameters = ExtractionParameters(flags.output_image_width, flags.encoding_height, flags.ste_image_w,
                                                 flags.ste_separator_size, flags.ste_superpixel_size, flags.batch_size)

    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu_id

    model_dir = flags.save_model_dir_name

    data_helper = DataHelper(num_words_to_keep, flags.save_model_dir_name)

    train_path = flags.train_path
    val_path = flags.val_path
    root_dir = flags.output_dir

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)

    data_loader = DataLoader()
    data_loader.load_data(train_path, val_path, delimiter='|')

    training_data = data_loader.get_training_data()
    val_data = data_loader.get_val_data()

    data_helper.load_from_pickles()
    text_train, text_val = data_helper.preprocess_texts(training_data, val_data, num_words_x_doc)
    label_train, label_val = data_helper.preprocess_labels(training_data, val_data)

    data_loader.set_training_data(text_train, label_train, training_data.get_images())
    data_loader.set_val_data(text_val, label_val, val_data.get_images())

    encoding_extractor = EncodingExtractor(training_data, val_data, root_dir, model_dir)
    encoding_extractor.extract(extraction_parameters)


if __name__ == '__main__':
    main(sys.argv[1])
