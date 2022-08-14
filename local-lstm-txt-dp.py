import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
import random
from opacus import PrivacyEngine

from src import Preprocessing
from src import SpamClassifier
from src import DatasetMapper
from src import Arguments
from src import calculate_accuracy
from src import json_output_to_file

import warnings


class Execute:
    """
    Class for execution. Initializes the preprocessing for the classification as well as training function

    Args:

    Returns:
        Instance ready to run the classification
    Raises:

    """

    def __init__(self, arguments, model_to_load):
        """
            Function for initialisation. Initializes the arguments and loads the model to the class

            Args:
                arguments (Arguments): parameter class
                model_to_load (SpamClassifier): model class
            Returns:
                None
            Raises:
                None
        """
        self.batch_size = arguments.batch_size
        self.model = model_to_load.to(arguments.device)  # loads model to the device CPU / CUDA
        print("The loaded model for the local classification is: \n", self.model)

    def train(self, train_load, test_load):
        """
            Function for execution. Initializes the optimiser, the privacy engine and creates training loop for the
            the training, inclusive monitoring during the training and summary of the key values at the end of each
            epoch.

            Args:
                train_load (DataLoader ): train loader for the model training
                test_load (DataLoader): test loader for testing the model
            Returns:
                None
            Raises:
                None
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate_local, )  # creating optimiser
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=args.learning_rate_local, )  # creating optimiser
        # alphas_list = gen_uniq_floats(1.05, 1.5, 10)
        print("The loaded optimiser for the local classification is: \n", optimizer)
        privacy_engine = PrivacyEngine(self.model,
                                       # batch_size=args.batch_size,
                                       # sample_size=len(data.x_train),
                                       sample_rate=args.sample_rate,
                                       # alphas=alphas_list,
                                       # alphas=range(2, 32),
                                       noise_multiplier=args.noise_multiplier,
                                       max_grad_norm=1.0,).to(args.device)  # defining the privacy engine
        privacy_engine.attach(optimizer)  # attach the privacy engine to the optimiser
        print("The loaded privacy engine for the local classification is: \n", privacy_engine)


        for epoch in range(args.epochs):
            epoch_start_time = time.time()  # starting time of the epoch
            predictions = []  # empty lists for predictions and targets
            targets = []
            loss = 0.0

            self.model.train().to(args.device)  # enabling model training and sending in on the device
            for batch_idx, (x_batch, y_batch) in enumerate(train_load):  # batch loading from train loader

                x = x_batch.type(torch.LongTensor).to(args.device)  # load encoded text in tensor and on the device
                y = y_batch.type(torch.FloatTensor).to(args.device)  # load spam/ham targets in tensor and on the device

                y_pred = self.model(x.long()).to(args.device)  # getting predictions from the model
                loss = f.binary_cross_entropy(torch.reshape(y_pred, (-1, )), y)  # loss from resized tensor
                optimizer.zero_grad()
                loss.backward()  # back propagation of the results
                optimizer.step()

                predictions += list(y_pred.squeeze().cpu().detach().numpy())  # creating list of predictions
                targets += list(y.squeeze().cpu().detach().numpy())  # creating list of targets

                if batch_idx % args.log_interval == 0:  # monitoring interval and key values output during training
                    #epsilon, best_alpha = 1.0 , "no dp"
                    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent()  # get the privacy info
                    # epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)  # get the privacy info

                    print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} for (ε = {:.2f}, δ = {}) for α = {}.'
                          .format("LSTM", epoch, batch_idx * args.batch_size, len(data.x_train),
                                  100. * batch_idx * args.batch_size / len(data.x_train),
                                  loss.item(), epsilon, args.delta, best_alpha,))

            writer.add_scalar("Loss/Train", loss.item(), epoch)

            test_predictions, test_targets = self.evaluation(test_load)  # testing the model resulting in two lists

            train_accuracy, train_f1 = calculate_accuracy(targets, predictions)  # f1 score and accuracy - training
            test_accuracy, test_f1 = calculate_accuracy(test_targets, test_predictions)  # f1 score and accuracy - test
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)  # get the privacy info

            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f, Time elapsed: %.2f sec."
                  "Epsilon %.2f Alpha %.2f" % (
                epoch + 1, loss.item(), train_accuracy, test_accuracy, (time.time() - epoch_start_time),
                epsilon, best_alpha))  # key values

            writer.add_scalar("Train_Accuracy/Train", train_accuracy, epoch)
            writer.add_scalar("Test_Accuracy/Test", test_accuracy, epoch)

            writer.add_scalar("Train_F1/Train", train_f1, epoch)
            writer.add_scalar("Test_F1/Test", test_f1, epoch)


    def evaluation(self, test_load):
        """
            Function for execution. Initializes the optimiser, the privacy engine and creates training loop for the
            the training, inclusive monitoring during the training and summary of the key values at the end of each
            epoch.

            Args:
                test_load (DataLoader ): test loader for the model testing
            Returns:
                None
            Raises:
                None
        """
        predictions = []
        targets = []
        self.model.eval()  # enabling model testing
        with torch.no_grad():
            for x_batch, y_batch in test_load:
                x = x_batch.type(torch.LongTensor).to(args.device)  # loading encoded text in tensor and on device
                y = y_batch.type(torch.FloatTensor).to(args.device)  # loading spam/ ham targets in tensor and on device
                y_pred = self.model(x).to(args.device)  # getting predictions from the model
                predictions += list(y_pred.squeeze().cpu().detach().numpy())
                targets += list(y.squeeze().cpu().detach().numpy())
        return predictions, targets


if __name__ == "__main__":
    # key parameters for the data processing
    RATIO = 1  # the ratio of messages to be used 0.5 from the whole dataset 16000 from 33000
    MAX_LENGTH = 50  # max length of each sentence to be encoded
    MAX_WORDS = 5000  # max number of words in the dictionary for the word embedding
    TEST_RATIO = 0.4  # the ratio of messages to be used for testing 0.2 from the whole dataset circa 6000 from 33000

    warnings.filterwarnings("ignore")

    # initialising the data processing and encoding messages
    data = Preprocessing(ratio=RATIO, max_len=MAX_LENGTH, max_words=MAX_WORDS)  # load the data, cleaning, stemming
    data.load_data(TEST_RATIO)  # load data into dataframe
    data.prepare_tokens()  # encode the txt data into tokens
    data.x_train = data.sequence_to_token(data.x_train)  # encode txt data to numeric messages
    data.x_test = data.sequence_to_token(data.x_test)  # encode txt data to numeric messages

    # initialise arguments for the classification and device to train on
    args = Arguments(samples=data.number_samples, max_length=MAX_LENGTH, max_words=MAX_WORDS)  # initialise parameters
    use_cuda = args.use_cuda and torch.cuda.is_available()  # determine to use CUDA or CPU
    args.device = torch.device("cuda" if use_cuda else "cpu")
    print("The classification will be run on device: ", args.device)

    info = {}
    info["args"] = {"num_samples": args.num_samples, "num_clients": args.clients, "num_epochs": args.epochs,
                    "num_rounds": args.rounds, "noise": args.noise_multiplier,
                    "batch_size_federated": args.local_batches}
    # print(info["args"])

    # needed for display in tensorboard
    random_int = random.randint(1, 100000)
    dir_name = "LOCAL_" + str(
        random_int) + f"_S{args.num_samples}C{1}E{args.epochs}R{1}N{args.noise_multiplier}"
    summary_name = "runs/" + dir_name
    print("The current run will be saved as : ", summary_name)
    writer = SummaryWriter(summary_name)

    model = SpamClassifier(args)  # loading the model
    execute = Execute(args, model)  # create class for execution

    # attaching batch loaders for test and training with the desired batch size on the execute class
    execute.loader_training = DataLoader((DatasetMapper(data.x_train, data.y_train)), batch_size=args.batch_size)
    execute.loader_test = DataLoader((DatasetMapper(data.x_test, data.y_test)), batch_size=args.batch_size)

    # starting training with the desired loaders
    execute.train(execute.loader_training, execute.loader_test)

    json_output_to_file(info, random_int, dir_name, RATIO)  # json output for statistics

    writer.flush()
    writer.close()