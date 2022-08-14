import time
import re
import json
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import opacus.layers

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words('english'))  # load english stopwords from nltk
st = SnowballStemmer('english')  # load english stemmer from nltk


class DatasetMapper(Dataset):
    """
        Class for mapping the values into a dataset.

        Args:

        Returns:
            Instance of the loaded dataset
        Raises:
            None
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
            Class for mapping the values into a federated dataset.

                Args:

                Returns:
                    Instance of the loaded dataset
                Raises:
                    None
        """
        return self.x[idx], self.y[idx]


class FedDataset(Dataset):
    """
            Class for mapping the values into a federated dataset.

            Args:

            Returns:
                Instance of the loaded dataset
            Raises:
                None
    """
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = [int(i) for i in idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        texts, label = self.dataset[self.idx[item]]
        return torch.tensor(texts).clone().detach(), torch.tensor(label).clone().detach()


class SpamClassifier(nn.ModuleList):
    """
               Class for the LSTM neural network model.

               Args:

               Returns:
                   Instance of the model
               Raises:
                   None
       """
    def __init__(self, args):
        """
            Initialisation class for the neural network

            Args:
                args: Arguments
            Returns:

            Raises:
                None
           """
        super(SpamClassifier, self).__init__()

        self.batch_size = args.batch_size   # size of the batches
        self.hidden_dim = args.hidden_dim   # size of the hidden dimension
        self.LSTM_layers = args.lstm_layers     # number of lstm layers
        self.input_size = args.max_words    # size of the embedding dimension 1
        self.embedding_size = args.embedding_size  # size of the embedding dimension 2
        self.device = args.device   # running device
        self.linear_size = int (self.hidden_dim // 2)  # the size of the second linear layer as the half of the first
        self.features_out = args.num_features_output # number of output classes

        # definition of the layers of the network
        self.embedding = nn.Embedding(self.input_size, self.embedding_size, padding_idx=0)  # word embedding
        self.dropout = nn.Dropout(0.5)
        self.lstm = opacus.layers.DPLSTM(input_size=self.embedding_size, hidden_size=self.hidden_dim,
                                         num_layers=self.LSTM_layers, batch_first=True)  # LSTM
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.linear_size)  # fully connected layer 1
        self.fc2 = nn.Linear(self.linear_size, self.features_out)  # fully connected layer 2

    def forward(self, x):
        """
            Feed forward function for the neural network

            Args:

            Returns:

            Raises:
                None
           """
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).to(self.device)  # hidden state initialise
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).to(self.device)  # internal state initialise

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)
        # propagate input through LSTM
        out = self.embedding(x)
        out, (_, _) = self.lstm(out, (h, c))  # LSTM with input, hidden state and internal state
        out = self.dropout(out)  # first dense
        out = torch.relu_(self.fc1(out[:, -1, :]))  # relu
        out = self.dropout(out)  # second dense
        out = torch.sigmoid(self.fc2(out))  # final output

        return out


class Preprocessing:

    def __init__(self, max_len=20, max_words=1000, test_size=0.2, ratio=0.1):
        self.max_len = max_len
        self.max_words = max_words
        self.test_size = test_size
        self.ratio = ratio
        self.path = './Data/enron/'
        self.number_samples = 0

    def load_data(self):
        import os
        from os import walk
        from pathlib import Path


        if not os.path.isdir(self.path):
            # Raise error if path is invalid.
            # print(self.path)
            raise ValueError('Invalid `path` variable! Needs to be a directory')

        def clean_data(df, col, clean_col):

            start_time = time.time()
            print("cleaning data:")
            # change to lower and remove spaces on either side
            df[clean_col] = df[col].apply(lambda z: z.lower().strip())
            print("cleaning data: all lowercase done")
            # remove extra spaces in between
            df[clean_col] = df[clean_col].apply(lambda z: re.sub(' +', ' ', z))
            print("cleaning data: all extra spaced removed done")
            # remove punctuation
            df[clean_col] = df[clean_col].apply(lambda z: re.sub('[^a-zA-Z]', ' ', z))
            print("cleaning data: all punctuation removed done")
            # remove stopwords and get the stem
            df[clean_col] = df[clean_col].apply(
                lambda z: ' '.join(st.stem(txt) for txt in z.split() if txt not in stop_words))
            print("cleaning data: all stopwords removed and stemmed done in {}".format(time.time() - start_time))
            return df

        path = Path(self.path)
        print("The path where the classification data is stored: ", path)
        path_walk = walk(path)
        texts = []
        labels = []
        answer = "n"
        # answer = input("This is the first time, it will take a while to generate pkl file: y or n:")
        if os.path.exists("./Data/enron.pkl") and answer == "n":
            print("Pickle exists")
            data = pd.read_pickle("./Data/enron.pkl")

        else:
            print("to be created")
            for root, dr, file in path_walk:
                if 'ham' in str(file):
                    for obj in file:
                        with open(root + '/' + obj, encoding='latin1') as ip:
                            text = ' '.join(ip.readlines())
                            texts.append(text)
                            labels.append(0)

                elif 'spam' in str(file):
                    for obj in file:
                        with open(root + '/' + obj, encoding='latin1') as ip:
                            text = ' '.join(ip.readlines())
                            texts.append(text)
                            labels.append(1)

            # Number of examples.
            n_examples = len(labels)
            print('Loaded and cleaned {} from the Dataset located at :{}'.format(n_examples, path))
            data = pd.DataFrame()
            data['text'] = texts
            data['labels'] = labels

            data = data.sample(frac=1).reset_index(drop=True)  # shuffle
            data_frame_cleaned = clean_data(data, 'text', 'cleantext')
            data_frame_cleaned.to_pickle("./Data/enron.pkl")
            print("new pkl generated")

        print("The shape of the dataframe for classification is: ", data.shape)
        print("The first 5 rows of the cleaned dataframe are: \n", data.head(5))
        print("Data set information:", data.info())
        print("Data set label disposition: \n", data['labels'] .value_counts())

        split = int(len(data["text"]) * self.ratio)
        print(f'For this classification {split} number of examples will be used')
        data = data.loc[:split, :]

        # X = data['text'].values
        x = data['cleantext'].values  # cleaned
        y = data['labels'].values

        self.number_samples = len(x)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size)

    def prepare_tokens(self):
        """
               Helper function to prepare tokens. Creates tokenized and apply on the in-class data.

               Args:

               Returns:

               Raises:

        """
        self.tokens = Tokenizer(num_words=self.max_words)
        self.tokens.fit_on_texts(self.x_train)

    def sequence_to_token(self, x):
        """
            Helper function to prepare tokens. Converts the tokens to sentences and trims them to the maximal length

            Args:

            Returns:

            Raises:

        """
        sequences = self.tokens.texts_to_sequences(x)
        return sequence.pad_sequences(sequences, maxlen=self.max_len)


def calculate_accuracy(grand_truth, predictions):
    """
        Helper function to calculate accuracy, f1 score and true & false positives/negatives.

        Args:
            grand_truth (list): List with the target values, which are true, as loaded from the dataset
            predictions (list): List with prediction to be compared with the list of true samples
        Returns:

        Raises:

        """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0  # initialising the variables

    for true, prediction in zip(grand_truth, predictions): # comparing the prediction and target lists:
        if (prediction > 0.5) and (true == 1):
            true_positives += 1
        elif (prediction < 0.5) and (true == 0):
            true_negatives += 1
        elif (prediction < 0.5) and (true == 1):
            false_negatives += 1
        elif (prediction > 0.5) and (true == 0):
            false_positives += 1

    f1_score = true_positives / (true_positives + 0.5 * (false_positives + false_negatives))
    print(f"True positives:{true_positives} / True negatives: {true_negatives} // False positives: {false_positives}, "
          f"False negatives: {false_negatives} and F1 score: {f1_score:.03f}")
    return (true_positives + true_negatives) / len(grand_truth), f1_score

def json_output_to_file(info, random_int, dir_name, ratio):
    json_dict = {}
    for key in info.keys():
        print(key, info[key])
        json_dict[key] = info[key]
    print(json_dict)
    data = json.dumps(json_dict)

    current_id = "Run" + str(random_int) + "ratio" + str(ratio)
    output_name = str(current_id) + ".json"
    # os.mkdir("./runs/" + dir_name)
    file_path = "./runs/" + dir_name + "/" + output_name
    fp = open(file_path, "x")
    fp.write("[\n", )
    fp.write(data)
    fp.write("\n]", )
    fp.close()


class Arguments:
    def __init__(self, samples=1, max_length=20, max_words=1000):  # num of samples, max length of sentence, dictionary
        """
            Container class for all the parameters. Initialised by number of loaded samples, max_length of each sentence
             and max_words for the size of the embedding dictionary).

            Args:
                samples (int): number of samples loaded, needed for calculating split size for each client
                max_length (int): the maximal size of the sentence to be processed, tested with values from 10 to 80
                max_words (int): the maximal size of the embedding dictionary - number of words tested up to 8000
            Returns:

            Raises:

        """
        # common parameters for CUDA and PyTorch
        self.device = ""  # for the CUDA device , leave empty
        self.use_cuda = True  # True when to be run on cuda, False if cuda not available or to be run on CPU
        self.log_interval = 2  # logging interval
        self.torch_seed = 888
        self.seed = 888

        # pre-processing parameters
        # self.test_size = 0.2  # test / train split
        self.max_words = max_words  # the size of dictionary for word embedding
        self.max_len = max_length  # maximal size of each sentence
        self.num_samples = samples  # total number of input samples
        self.save_model = False  # if the model should be saved

        # model parameters
        self.learning_rate_local = 0.01
        self.learning_rate_federated = 0.01  # optimiser learning rate
        self.lstm_layers = 2  # number of LSTM layers
        self.embedding_size = 256
        self.hidden_dim = 128
        self.num_features_output = 1  # Features 1, could be configured to a multi type classifier

        # parameters for local DPLSTM training
        self.epochs = 10
        self.batch_size = 256
        self.threshold = 0.85

        # parameters for federated DPLSTM training
        self.C = 1
        self.drop_rate = 0.0  #
        self.clients = 5  # number of clients for the federated training
        self.rounds = 50  # number of rounds for federated training
        self.split_size = int(self.num_samples / self.clients)  # the size of the split dataset for each client
        self.samples = self.split_size / self.num_samples  # what part of whole samples on each client.
        # self.local_batches = int (self.split_size / 4)  # size of the batch for each local client
        self.local_batches = 256

        # parameters for differential privacy
        self.noise_multiplier = 0.99
        self.delta = 1e-5
        self.sample_rate = self.local_batches / self.split_size
