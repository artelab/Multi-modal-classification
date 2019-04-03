class GenericResult(object):

    def __init__(self, current_step):
        self.current_step = current_step

    def __str__(self):
        return 'Step: {} '.format(self.current_step)
