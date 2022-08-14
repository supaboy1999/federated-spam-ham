import torch
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import syft as sy
import numpy as np
import time
import warnings
from opacus import PrivacyEngine
import random

from src import Arguments
from src import Preprocessing
from src import DatasetMapper
from src import SpamClassifier
from src import FedDataset
from src import calculate_accuracy
from src import json_output_to_file


def split_dataset_to_users(dataset, num_users):
    """
        Splits the samples of the given dataset over the number of users by creating a dictionary of indexes to each
        data record
        Args:
            dataset (DatasetMapper): input dataset to be split
            num_users (int): the number of users on which to data set to be split
        Returns:
            user_dict (dict) : dictionary with key for every user and values the indexes to the data
    """
    num_samples = int(len(dataset) / num_users)  # getting number of samples pro worker
    users_dict, idxs = {}, [i for i in range(len(dataset))]  # creating dictionaries for indexing
    for i in range(num_users):
        np.random.seed(i)  # random user
        users_dict[i] = set(np.random.choice(idxs, num_samples, replace=False))  # for that user get random element
        idxs = list(set(idxs) - users_dict[i])  # remove the element from the dataset
    return users_dict


def load_dataset_local(num_users):
    """
        loads the raw data into data set and splits it over the users

    :param num_users:
    :return:
    """
    """
        Args:
             num_users (int): the number of users on which to data set to be split 
        Returns:
            train_dataset (
    """
    raw_x_train = data.x_train  # gets the text train data from the dataset without labels
    raw_x_test = data.x_test  # gets the text test data from the dataset without labels

    rand = random.randint(0, len(data.x_train))
    print("Random sample:", data.x_train[rand], "\n The answer is :", data.y_train[rand])

    data.x_train = data.sequence_to_token(raw_x_train)  # converts train data to tokens
    data.x_test = data.sequence_to_token(raw_x_test)  # converts test data to tokens

    print("Random sequenced sample:", data.x_train[rand], "\n The answer is :", data.y_train[rand])  # to print info

    train_dataset = DatasetMapper(data.x_train, data.y_train)  # maps the converted text train data to the labels
    test_dataset = DatasetMapper(data.x_test, data.y_test)  # maps the converted text test data to the labels

    train_grp = split_dataset_to_users(train_dataset, num_users)  # creates dictionary with train data for each worker
    test_grp = split_dataset_to_users(test_dataset, num_users)  # creates dictionary with test data for each worker

    return train_dataset, test_dataset, train_grp, test_grp


def get_data_for_idxs(dataset, idxs, batch_size):
    return DataLoader(FedDataset(dataset, idxs), batch_size=batch_size, shuffle=True)  # spreads the dataset


def train_local_worker(arguments, round_number, worker):
    """
    function for the training of the local workers
    :param arguments: the Arguments class with all the parameter
    :param round_number: number of the current federated round
    :param worker: on which worker is to be trained
    :return:
    """
    print(49 * "X", f'client update\n, on {worker}')  # print info on which worker will be trained
    # print(worker)
    worker['model'].train()  # start training

    for epoch in range(1, arguments.epochs + 1):
        epoch_start_time = time.time()
        predictions = []
        targets = []

        for batch_idx, (txt, target) in enumerate(worker['trainset']):
            x = txt.type(torch.LongTensor).to(arguments.device)  # convert each batch to Long tensor and device
            y = target.type(torch.FloatTensor).to(arguments.device)  # convert each batch to Float tensor and device

            y_pred = worker['model']((x.long()).to(arguments.device))  # get model predictions
            loss = f.binary_cross_entropy(torch.reshape(y_pred, (-1,)), y)  # calculate loss
            worker['optim'].zero_grad()  # zero the gradients
            loss.backward()  # propagate loss backwars
            worker['optim'].step()

            predictions += list(y_pred.squeeze().cpu().detach().numpy())  # add predictions to the list
            targets += list(y.squeeze().cpu().detach().numpy())  # add the target values to the list

            # statistics
            if batch_idx % args.log_interval == 0:
                epsilon, best_alpha = worker['optim'].privacy_engine.get_privacy_spent()

                print('ROUND {}: Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                      'Loss: {:.6f} for (ε = {:.2f}, δ = {}) for α = {}'
                      .format(round_number, worker['hook'].id, epoch,
                              batch_idx * arguments.local_batches, len(worker['trainset']) * arguments.local_batches,
                              100. * batch_idx / len(worker['trainset']), loss.item(),
                              epsilon, arguments.delta, best_alpha, ))

        print("The results of the model on the train set are:")
        train_accuracy, f1_score = calculate_accuracy(targets, predictions)  # calculate train accuracy for each epoch

        epsilon, best_alpha = worker['optim'].privacy_engine.get_privacy_spent(arguments.delta)  # privacy budget
        print("Epoch time elapsed {:.2f}, Accuracy / f1 score ".format(time.time() - epoch_start_time),
              train_accuracy, f1_score, "privacy / alpha", epsilon, best_alpha)

        # for tensorboard
        worker['totalepoch'] += epoch
        writer.add_scalar("Federated_Train_Accuracy/Train {} Round {} ".format(worker['hook'].id, round_number),
                          train_accuracy, epoch)
        writer.add_scalar("Federated_Train_F1/Train {} Round {} ".format(worker['hook'].id, round_number),
                          f1_score, epoch)
        writer.add_scalar("Federated_Privacy/ Client: {}  ".format(worker['hook'].id),
                          epsilon, worker["totalepoch"])


        if f1_score >= arguments.threshold and arguments.threshold != 0:
            break  # break the training in case the model has more than 0.995 f1 score, can spare some time

    return epsilon, best_alpha


def average_models(global_model, workers):
    """
    This function averages the weights of all the workers and updates the global model afterwards
    :param global_model: the global model to be adjusted
    :param workers: list of all the active workers, to provide updated weights
    :return:
    """
    client_models = [workers[i]['model'] for i in range(len(workers))]  # get every model from the workers
    samples = [workers[i]['samples'] for i in range(len(workers))]  # number of samples for each worker as % from all
    global_dict = global_model.state_dict()  # get the global model gradients

    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [client_models[i].state_dict()[k].float() * samples[i] for i in range(len(client_models))], 0).sum(0)
        # each global gradient,is replaced by the sum of workers federated gradients multiplied by portion of samples
    global_model.load_state_dict(global_dict)  # replace the global model with the new one
    return global_model


def test(arguments, model, device, test_loader, name):
    """
    this function tests the global model
    :param arguments: the parameters needed in an Aguments class
    :param model: the model to be tested
    :param device: on which device to be tested
    :param test_loader: which test loaded to be used, which data is loaded
    :param name: model name
    :return: metrics for the accuracy F1 score and loss
    """
    model.eval().to(device)
    test_loss = 0
    correct = 0
    outputs = []
    targets = []

    with torch.no_grad():
        for txt, target in test_loader:
            txt, target = txt.type(torch.LongTensor).to(device), target.type(torch.FloatTensor).to(device)
            output = model(txt)
            outputs += list(output.squeeze().cpu().detach().numpy())
            test_loss += f.binary_cross_entropy(output, target).item()  # sum up batch loss
            targets += target
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy, f1_score = calculate_accuracy(targets, outputs)
    test_loss /= len(test_loader.dataset) * arguments.local_batches
    print('\nTest set: Average loss for {} model: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        name, test_loss, test_accuracy, len(test_loader.dataset),
        ((1000 * test_accuracy) / len(test_loader.dataset))))

    return test_loss, test_accuracy, f1_score


def create_workers(arguments):
    """
    this function creates the workers for the federated training
    :param arguments: copy of the parameters
    :return: a container - list with all workers
    """
    workers = []  # container array for workers

    for i in range(arguments.clients):
        workers.append({'hook': sy.VirtualWorker(hook, id="Worker{}".format(i + 0))})  # create each worker

    for inx, worker in enumerate(workers):
        trainset_ind_list = list(train_group[inx])
        testset_ind_list = list(test_group[inx])# total train set length
        worker['trainset'] = get_data_for_idxs(global_train, trainset_ind_list, arguments.local_batches)  # train set
        worker['testset'] = get_data_for_idxs(global_test, testset_ind_list, arguments.local_batches)  # test set
        worker['samples'] = len(trainset_ind_list) / arguments.num_samples  # portion of samples for the worker
        worker['model'] = SpamClassifier(arguments).to(arguments.device)  # sending the model to the worker
        worker['optim'] = optim.Adam(worker['model'].parameters(), lr=arguments.learning_rate_federated)  # optimizer
        worker['pengine'] = PrivacyEngine(
            worker['model'],
            #  batch_size=arguments.local_batches,
            #  sample_size=len(worker['trainset']),
            sample_rate=arguments.sample_rate,
            #  alphas=range(2, 32),
            noise_multiplier=arguments.noise_multiplier,
            max_grad_norm=1.0, )  # privacy engine
        worker['pengine'].attach(worker['optim'])  # attach privacy engine to optimizer
        worker['totalepoch'] = 0

    return workers


class Execute:
    """
    this class contains the main function, providing the federated rounds for training
    """
    @staticmethod
    def train(arguments):
        global global_train, global_test, train_group, test_group

        global_train, global_test, train_group, test_group = load_dataset_local(arguments.clients)  # create the dataset
        workers = create_workers(arguments)  # create workers

        global_test_dataset = DatasetMapper(data.x_test, data.y_test)  # dataset for testing the global model
        global_test_loader = DataLoader(global_test_dataset, batch_size=arguments.local_batches,
                                        shuffle=True)  # create loader for the global model
        print(f"Number of samples used for training: {len(global_train)}, "
              f"number of samples used for testing: {len(global_test_dataset)}")

        torch.manual_seed(arguments.torch_seed)  # randomization seed
        global_model = SpamClassifier(args)  # define global model
        trainable_layers = [global_model.fc1, global_model.fc2]
        total_params = 0
        trainable_params = 0

        for p in global_model.parameters():
            p.requires_grad = False
            total_params += p.numel()

        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True
                trainable_params += p.numel()

        print(f"Total model parameters count: {total_params}")  # ~125M
        print(f"Trainable model parameters count: {trainable_params}")  # ~0.5M

        for fed_round  in range(1, args.rounds + 1):
            # fed_round += 1
            print(88 * "X", 'New round', fed_round)
            round_start_time = time.time()

            # number of selected clients
            m = int(max(args.C * args.clients, 1))
            print(f"for round {fed_round} are totally active clients {m}")

            # Selected devices
            np.random.seed(fed_round)
            selected_clients_inds = np.random.choice(range(len(workers)), m, replace=False)
            selected_clients = [workers[i] for i in selected_clients_inds]
            print(f"for round {fed_round} are selected randomized clients {selected_clients_inds}")

            # Active devices
            np.random.seed(fed_round)
            active_clients_inds = np.random.choice(selected_clients_inds, int((1 - args.drop_rate) * m), replace=False)
            active_workers = [workers[i] for i in active_clients_inds]
            print(f"the active workers are {active_clients_inds}")

            # Training
            for worker in active_workers:
                epsilon, best_alpha = train_local_worker(arguments, fed_round, worker) # train on each worker

            # Averaging
            global_model = average_models(global_model, active_workers)

            # Testing the average model
            print("The results of the model on the test set are: \n")
            loss, acc, f1 = test(arguments, global_model, arguments.device, global_test_loader, 'Global')

            writer.add_scalar("Global_Loss/Testing_round", loss, fed_round)
            writer.add_scalar("Global_Accuracy/Testing_round", acc, fed_round)
            writer.add_scalar("Global_F1_Score/Testing_round", f1, fed_round)

            # Share the global model with the clients
            for worker in active_workers:
                worker['model'].load_state_dict(global_model.state_dict())

            # info
            print('Round time elapsed:', time.time() - round_start_time, 149 * "X")
            info[fed_round] = {"round_time": time.time() - round_start_time, "round_loss": loss,
                               "round_acc": acc, "round_f1": f1, "round_epsilon": epsilon, "round_alpha": best_alpha}

        if arguments.save_model:
            torch.save(global_model.state_dict(), "Federated-differential-privacy.pt")


if __name__ == "__main__":

    # key parameters for the data processing
    RATIO = 1  # the ratio of messages to be used 0.5 from the whole dataset 16000 from 33000, 1 for the whole set
    MAX_LENGTH = 80  # max length of each sentence to be encoded
    MAX_WORDS = 8000  # max number of words in the dictionary for the word embedding
    TEST_RATIO = 0.1  # the ratio of messages to be used for testing 0.2 from the whole dataset circa 6000 from 33000

    warnings.filterwarnings("ignore")

    # initialising the data processing and encoding messages
    data = Preprocessing(ratio=RATIO, max_len=MAX_LENGTH, max_words=MAX_WORDS, test_size=TEST_RATIO)  # load the data, cleaning, stemming
    data.load_data()  # load data into dataframe
    data.prepare_tokens()  # encode the txt data into tokens

    # initialise arguments for the classification and device to train on
    args = Arguments(samples=data.number_samples, max_length=MAX_LENGTH, max_words=MAX_WORDS)   # initialise parameters
    use_cuda = args.use_cuda and torch.cuda.is_available()  # determine to use CUDA or CPU
    args.device = torch.device("cuda" if use_cuda else "cpu")
    print("The classification will be run on device: ", args.device)

    hook = sy.TorchHook(torch)  # hook for PySyft pointing to the torch library

    # which parameters should be logged for the json output
    info = {}
    info["args"] = {"num_samples": args.num_samples, "num_clients": args.clients, "num_epochs": args.epochs,
                    "num_rounds": args.rounds, "noise": args.noise_multiplier,
                    "batch_size_federated": args.local_batches}
    # print(info["args"])

    # needed for display in tensorboard
    random_int = random.randint(1,100000)
    dir_name = "FED_"+str(random_int)+f"_S{args.num_samples}C{args.clients}E{args.epochs}R{args.rounds}" \
                                      f"N{args.noise_multiplier}Tr{args.threshold}"
    summary_name = "runs/" + dir_name
    print("The current run will be saved as : ", summary_name)
    writer = SummaryWriter(summary_name)

    execute = Execute()     # creating the training class
    execute.train(args)     # starting training with the desired arguments

    writer.flush()
    writer.close()

    json_output_to_file(info, random_int, dir_name, RATIO)  # json output for statistics

