class ModelParameters(object):

    def __init__(self, save_model_dir_name, no_of_epochs, patience, evaluate_every):
        self.model_directory = save_model_dir_name
        self.patience = patience
        self.no_of_epochs = no_of_epochs
        self.evaluate_every = evaluate_every

    def get_model_directory(self):
        return self.model_directory

    def get_patience(self):
        return self.patience

    def get_no_of_epochs(self):
        return self.no_of_epochs

    def get_evaluate_every(self):
        return self.evaluate_every
