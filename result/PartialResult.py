class PartialResult:

    def __init__(self, epoch, current_step, test_accuracy, best_accuracy, patience):
        self.epoch = epoch
        self.current_step = current_step
        self.test_accuracy = test_accuracy
        self.best_accuracy = best_accuracy
        self.patience = patience

    def __str__(self):
        return 'Epoch: {}, Step: {}, Test acc: {}, Best acc: {}, Patience: {}\n'\
            .format(self.epoch, self.current_step, self.test_accuracy, self.best_accuracy, self.patience)
