from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Dropout, Flatten, GaussianNoise

class SoluteModel(BaseModel):
    def __init__(self, config):
        super(SoluteModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        # TODO get network parameters from outside
        # TODO consider using batch normalization
        # TODO make sure that using logcosh is the same as huber loss
        self.model.add(GaussianNoise(0.01))
        self.model.add(Conv1D(29, kernel_size=124, activation='relu'))
        self.model.add(Conv1D(28, kernel_size=119, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(289))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

        self.model.compile(
              loss='logcosh',
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])
