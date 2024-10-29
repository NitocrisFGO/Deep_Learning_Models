# models.py
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class PrefixEmbeddings:
    def __init__(self, word_embeddings, prefix_length=3):
        self.unk_index = 0
        self.pad_index = 0
        self.prefix_length = prefix_length
        self.prefix_indexer = {}
        self.embeddings = []
        self.init_prefix_embeddings(word_embeddings.word_indexer, word_embeddings)

    def init_prefix_embeddings(self, word_indexer, word_embeddings):
        prefix_to_vectors = {}

        for idx in range(2, len(word_indexer)):
            word = word_indexer.get_object(idx)
            vector = word_embeddings.get_embedding(word)
            prefix = word[:self.prefix_length]
            if prefix not in prefix_to_vectors:
                prefix_to_vectors[prefix] = []
            prefix_to_vectors[prefix].append(vector)

        for prefix, vectors in prefix_to_vectors.items():
            mean_vector = np.mean(vectors, axis=0)
            index = len(self.prefix_indexer)
            self.prefix_indexer[prefix] = index
            self.embeddings.append(mean_vector)

        self.embeddings = np.array(self.embeddings)

        self.pad_index = len(self.prefix_indexer)
        self.unk_index = self.pad_index + 1

        self.prefix_indexer['PAD'] = self.pad_index
        self.prefix_indexer['UNK'] = self.unk_index

        pad_embedding = np.zeros(self.embeddings.shape[1])
        unk_embedding = np.zeros(self.embeddings.shape[1])

        self.embeddings = np.vstack([self.embeddings, pad_embedding, unk_embedding])

    def get_embedding(self, word):
        prefix = word[:self.prefix_length]
        if prefix in self.prefix_indexer:
            return self.embeddings[self.prefix_indexer[prefix]]
        else:
            return self.embeddings[self.unk_index]

    def get_pad_embedding(self):
        return self.embeddings[self.pad_index]

    def __len__(self):
        return len(self.prefix_indexer)

    def get_embedding_length(self):
        return self.embeddings.shape[1]


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier, nn.Module):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, inp, hid, out, embedding_layer):
        super(NeuralSentimentClassifier, self).__init__()

        self.embedding_layer = embedding_layer
        self.firstLayer = nn.Linear(inp, hid)
        self.firstLayerReLU = nn.ReLU()
        self.OutPutLayer = nn.Linear(hid, out)

        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.firstLayer.weight)
        nn.init.xavier_uniform_(self.OutPutLayer.weight)

    def sentence2matirx(self, sentence) -> torch.Tensor:
        output_vector = torch.zeros((1, self.embedding_layer.get_embedding_length()))
        for word in sentence:
            output_vector += self.embedding_layer.get_embedding(word)
        output_vector /= len(sentence)
        return output_vector

    def forward(self, words) -> torch.Tensor:
        a1 = self.firstLayer(words)
        z1 = self.firstLayerReLU(a1)
        a2 = self.OutPutLayer(z1)
        return a2

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        ex_vector = self.sentence2matirx(ex_words)
        ex_vector = ex_vector.to(torch.float32)
        with torch.no_grad():
            output = self.forward(ex_vector)
            prediction = torch.argmax(output, dim=1).item()
        return prediction


class BoundsNeuralSentimentClassifier(SentimentClassifier, nn.Module):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, inp, hid, out, embedding_layer):
        super(BoundsNeuralSentimentClassifier, self).__init__()

        self.embedding_layer = embedding_layer
        self.prefix_embeddings_layer = PrefixEmbeddings(embedding_layer, prefix_length=3)

        self.firstLayer = nn.Linear(inp, hid)
        self.firstLayerReLU = nn.ReLU()
        self.secondLayer = nn.Linear(hid, hid)
        self.secondLayerReLU = nn.ReLU()
        self.thirdLayer = nn.Linear(hid, hid)
        self.thirdLayerReLU = nn.ReLU()
        self.outPutLayer = nn.Linear(hid, out)

        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.firstLayer.weight)
        nn.init.xavier_uniform_(self.secondLayer.weight)
        nn.init.xavier_uniform_(self.thirdLayer.weight)
        nn.init.xavier_uniform_(self.outPutLayer.weight)

    def sentence2matirx(self, sentence) -> torch.Tensor:
        output_vector = torch.zeros((1, self.prefix_embeddings_layer.get_embedding_length()))
        for word in sentence:
            output_vector += self.prefix_embeddings_layer.get_embedding(word)
        output_vector /= len(sentence)
        return output_vector

    def forward(self, words) -> torch.Tensor:
        a1 = self.firstLayer(words)
        z1 = self.firstLayerReLU(a1)
        a2 = self.secondLayer(z1)
        z2 = self.secondLayerReLU(a2)
        a3 = self.thirdLayer(z2)
        z3 = self.thirdLayerReLU(a3)
        a4 = self.outPutLayer(z3)
        return a4

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        ex_vector = self.sentence2matirx(ex_words)
        ex_vector = ex_vector.to(torch.float32)
        with torch.no_grad():
            output = self.forward(ex_vector)
            prediction = torch.argmax(output, dim=1).item()

        return prediction


def get_random_batch(X, y, batch_size):
    num_samples = X.shape[0]
    indices = np.random.choice(num_samples, batch_size, replace=False)
    X_batch = X[indices]
    y_batch = y[indices]

    return X_batch, y_batch


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

    train_texts = [ex.words for ex in train_exs]
    train_labels = [ex.label for ex in train_exs]
    embedding_dim = word_embeddings.get_embedding_length()

    X_train = torch.zeros((len(train_texts), embedding_dim))

    for j in range(len(train_texts)):
        sentence = train_texts[j]
        text_len = len(sentence)
        for word in sentence:
            X_train[j] += word_embeddings.get_embedding(word)
        X_train[j] /= text_len

    y_train = torch.tensor(train_labels)
    if train_model_for_typo_setting:
        # test = PrefixEmbeddings(word_embeddings, prefix_length=3)
        # print(word_embeddings.get_embedding("the"))
        # print(test.get_embedding("the"))
        batch_size = 256
        num_epochs = 20000
        initial_learning_rate = 0.01
        # model = BoundsNeuralSentimentClassifier(embedding_dim, 100, 2, word_embeddings, train_model_for_typo_setting)
        model = BoundsNeuralSentimentClassifier(embedding_dim, 100, 2, word_embeddings)
    else:
        batch_size = 128
        num_epochs = 200
        initial_learning_rate = 0.03
        model = NeuralSentimentClassifier(embedding_dim, 100, 2, word_embeddings)

    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    for epoch in range(0, num_epochs):
        model.train()
        X_epoch, y_epoch = get_random_batch(X_train, y_train, batch_size)
        optimizer.zero_grad()
        log_probs = model.forward(X_epoch)
        loss = nn.CrossEntropyLoss()(log_probs, y_epoch)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model
