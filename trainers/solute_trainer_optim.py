from base.base_trainer import BaseTrain
import os

from keras.callbacks import ModelCheckpoint, TensorBoard


class SoluteModelTrainerOptim(BaseTrain):
    def __init__(self, model, data, config):
        super(SoluteModelTrainerOptim, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        #self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        if hasattr(self.config, "comet_api_key"):
            print(self.config.exp.comet_api_key)
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.exp.comet_api_key, project_name=self.config.exp_name)
            #experiment = Experiment(api_key="AHcyFd2O9nvww9eaAE7cxhuVA")
            experiment.disable_mp()
            experiment.log_multiple_params(self.config)
            self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        history = self.model.fit(
            self.data[0], self.data[1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            #callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['r_2'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_r_2'])

    def get_train_log(self):
        return self.loss, self.acc, self.val_loss, self.val_acc