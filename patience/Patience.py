class Patience(object):

    def __init__(self, patience_initial_value):
        self.initial_value = patience_initial_value
        self.value = patience_initial_value

    def get_actual_value(self):
        return self.value

    def reset_patience(self):
        self.value = self.initial_value

    def decrement_patience(self):
        self.value -= 1

    def is_zero(self):
        return self.value == 0

    def __str__(self):
        return str(self.value)
