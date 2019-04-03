from result.GenericResult import GenericResult


class TrainingResult(GenericResult):

    def __init__(self, step, loss, accuracy):
        super(TrainingResult, self).__init__(step)
        self.loss = loss
        self.accuracy = accuracy

    def __str__(self):
        return super(TrainingResult, self).__str__() + 'Loss: {:g}, Accuracy: {:g}'.format(self.loss, self.accuracy)
