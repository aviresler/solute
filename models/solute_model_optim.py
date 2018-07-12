from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Dropout, Flatten, GaussianNoise, BatchNormalization
from keras import backend as K
from keras import regularizers
import numpy as np
from keras import optimizers

class SoluteModelOptim(BaseModel):
    def __init__(self, config):
        super(SoluteModelOptim, self).__init__(config)
        self.build_model()

    def r_2(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))


    def build_model(self):
        self.model = Sequential()
        self.choose_hyper_parmaters_random()
        # TODO get network parameters from outside
        # TODO hyperparameter optimization
        # TODO think if GaussianNoise layer is needed
        # TODO consider using batch normalization
        # TODO make sure that using logcosh is the same as huber loss
        self.model.add(GaussianNoise(0.01))
        self.model.add(Conv1D(self.l1_d, kernel_size=self.l1_ks, activation='relu', kernel_regularizer=regularizers.l2(self.reg_l2), input_shape=(1, 601)))
        self.model.add(BatchNormalization())
        self.model.add(Conv1D(self.l2_d, kernel_size=self.l2_ks, activation='relu', kernel_regularizer=regularizers.l2(self.reg_l2)))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(self.dense_num_neuron, kernel_regularizer=regularizers.l2(self.reg_l2)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, kernel_regularizer=regularizers.l2(self.reg_l2)))

        adam1 = optimizers.Adam(lr=self.learning_rate)

        self.model.compile(
              loss='logcosh',
              optimizer=adam1,
              metrics=[self.r_2])

    def choose_hyper_parmaters_random(self):
        self.learning_rate = self.config.model.learning_rate
        self.reg_l2 = self.config.model.l2_reg

        self.l1_d = np.random.randint(low=self.config.param_optim.l1_depth[0], high=self.config.param_optim.l1_depth[1])
        self.l1_ks = np.random.randint(low=self.config.param_optim.l1_filter_size[0], high=self.config.param_optim.l1_filter_size[1])
        if np.mod(self.l1_ks,2) == 0:
            self.l1_ks = self.l1_ks + 1
        self.l2_d = np.random.randint(low=self.config.param_optim.l2_depth[0], high=self.config.param_optim.l2_depth[1])
        self.l2_ks = np.random.randint(low=self.config.param_optim.l2_filter_size[0], high=self.config.param_optim.l2_filter_size[1])
        if np.mod(self.l2_ks,2) == 0:
            self.l2_ks = self.l2_ks + 1
        self.dense_num_neuron = np.random.randint(low=self.config.param_optim.dense_num_nuerons[0], high=self.config.param_optim.dense_num_nuerons[1])


        #self.l1_d = 29
        #self.l1_ks = 124
        #self.l2_d = 28
        #self.l2_ks = 119
        #self.dense_num_neuron = 289
        #self.learning_rate = 10**np.random.uniform(low=self.config.param_optim.learning_rate[0], high=self.config.param_optim.learning_rate[1])
        #self.reg_l2 = 10**np.random.uniform(low=self.config.param_optim.regularization[0], high=self.config.param_optim.regularization[1])


    def get_model_params(self):
        return self.l1_ks, self.l1_d, self.l2_ks, self.l2_d, self.dense_num_neuron, self.learning_rate, self.reg_l2

