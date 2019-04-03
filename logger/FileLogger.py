class FileLogger(object):

    def __init__(self, filename):
        self.filename = filename

    def write_header(self, model_directory, dataset_path):
        with open(self.filename, 'a') as resfile:
            resfile.write('Model dir: {}\n'.format(model_directory))
            resfile.write('Dataset: {}\n'.format(dataset_path))

    def write_partial_result_to_file(self, partial_result):
        with open(self.filename, 'a') as resfile:
            resfile.write(str(partial_result))
