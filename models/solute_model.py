from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Dropout, Flatten, GaussianNoise
from keras import backend as K

class SoluteModel(BaseModel):
    def __init__(self, config):
        super(SoluteModel, self).__init__(config)
        self.build_model()

    def r_2(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    def build_model(self):
        self.model = Sequential()
        # TODO get network parameters from outside
        # TODO hyperparameter optimization
        # TODO think if GaussianNoise layer is needed
        # TODO consider using batch normalization
        # TODO make sure that using logcosh is the same as huber loss
        self.model.add(GaussianNoise(0.01))
        self.model.add(Conv1D(29, kernel_size=124, activation='relu',input_shape=(1, 601)))
        self.model.add(Conv1D(28, kernel_size=119, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(289))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

        self.model.compile(
              loss='logcosh',
              optimizer=self.config.model.optimizer,
              metrics=[self.r_2])
