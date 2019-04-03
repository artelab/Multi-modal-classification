from result.GenericResult import GenericResult


class PartialResult(GenericResult):

    def __init__(self, epoch, current_step, test_accuracy, best_accuracy, patience):
        super(PartialResult, self).__init__(current_step)
        self.epoch = epoch
        self.test_accuracy = test_accuracy
        self.best_accuracy = best_accuracy
        self.patience = patience

    def __str__(self):
        return super(PartialResult, self).__str__() + 'Epoch: {}, Test acc: {}, Best acc: {}, Patience: {}\n'\
            .format(self.epoch, self.test_accuracy, self.best_accuracy, self.patience)
