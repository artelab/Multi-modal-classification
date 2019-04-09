import os
import shutil
import sys

from dataManagement.DataHelper import DataHelper
from dataManagement.DataLoader import DataLoader
from parameterManager.ModelParameters import ModelParameters
from parameterManager.TrainingParameters import TrainingParameters
from tensorflowWrapper.FlagsParser import FlagsParser
from trainer.ModelTrainer import ModelTrainer


def main(args):
    flags_parser = FlagsParser(delimiter=';')
    flags_parser.parse_parameter_from_file(args)
    flags = flags_parser.get_flags()

    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu_id

    output_image_width = flags.output_image_width
    encoding_height = flags.encoding_height

    checkpoint_dir = os.path.abspath(os.path.join(flags.save_model_dir_name, 'checkpoints'))

    if os.path.exists(flags.save_model_dir_name):
        shutil.rmtree(flags.save_model_dir_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    training_params = TrainingParameters(flags.num_words_to_keep, output_image_width, encoding_height,
                                         flags.dropout_keep_prob,
                                         flags.embedding_dim, flags.batch_size, flags.filter_sizes, flags.num_filters)
    model_params = ModelParameters(flags.save_model_dir_name, flags.num_epochs, flags.patience, flags.evaluate_every)

    data_loader = DataLoader()
    data_loader.load_data(flags.train_path, flags.val_path, delimiter='|', shuffle_data=True)

    training_data = data_loader.get_training_data()
    val_data = data_loader.get_val_data()

    data_helper = DataHelper(training_params.get_no_of_words_to_keep(), flags.save_model_dir_name)
    train_y, val_y = data_helper.preprocess_labels(training_data, val_data)
    train_x, val_x = data_helper.preprocess_texts(training_data, val_data, flags.num_words_x_doc)

    data_loader.set_training_data(train_x, train_y, training_data.get_images())
    data_loader.set_val_data(val_x, val_y, val_data.get_images())

    data_helper.store_preprocessors_to_disk()

    trainer = ModelTrainer(data_loader.get_training_data(), data_loader.get_val_data())
    trainer.train(training_params, model_params)


if __name__ == '__main__':
    main(sys.argv[1])
