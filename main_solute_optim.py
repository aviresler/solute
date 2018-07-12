
from data_loader.solute_loader import SoluteDataLoader
from models.solute_model_optim import SoluteModelOptim
from trainers.solute_trainer_optim import SoluteModelTrainerOptim
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import csv


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = SoluteDataLoader(config)

    # iterate over hyperparamters, and log each run for comparison. add hyper parmeters to model function
    lr = []
    reg = []
    val_acc = []
    val_loss = []
    for k in range(config.param_optim.num_of_itr_random):

        print('Create the model.')
        model = SoluteModelOptim(config)


        print('Create the trainer')
        trainer = SoluteModelTrainerOptim(model.model, data_loader.get_train_data(), config)

        print('Start training the model.')
        trainer.train()

        l1_ks, l1_d, l2_ks, l2_d, dense_num_neuron, learning_rate, reg_l2 = model.get_model_params()
        loss, acc, val_loss, val_acc = trainer.get_train_log()

        param_list = [l1_ks, l1_d, l2_ks, l2_d, dense_num_neuron, learning_rate, reg_l2]
        metric_list = [loss, acc, val_loss,val_acc]
        metric_list = zip(*metric_list)

        csvfile = 'experiments/log_{0:.2}.csv'.format(val_acc[-1])

        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['l1_ks', 'l1_d', 'l2_ks', 'l2_d','dense_num_neuron','learning_rate','reg_l2'])
            writer.writerows([param_list])
            writer.writerow(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
            writer.writerows(metric_list)

        print('itr {0}'.format(k))






    #with open("summary1", "w") as text_file:
    #    for k in range(config.param_optim.num_of_itr_random):
    #        str = 'itr {0}, val_acc {1:.2f}, val_loss {2:.2f}, lr = {3:.2E}, reg = {4:.2E}\n'.format(k, val_acc[k],val_loss[k],lr[k],reg[k] )
    #        print(str)
    #        text_file.write(str)




if __name__ == '__main__':
    main()
