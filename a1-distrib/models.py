# models.py
import nltk
import numpy as np
from nltk.corpus import stopwords

from sentiment_data import *
from utils import *

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> np.ndarray:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer, inverse_document_frequency):
        self.indexer = indexer
        self.inverse_document_frequency = inverse_document_frequency

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> np.ndarray:
        feature_vector = np.zeros(self.indexer.__len__())
        for word in sentence:
            if self.indexer.contains(word.lower()):
                word_index = self.indexer.index_of(word.lower())
                idf_value = 1
                feature_vector[word_index] += idf_value

        # with open('output.txt', 'w') as f:
        #     np.set_printoptions(threshold=np.inf)
        #     print(self.indexer, file=f)
        #     print(sentence, file=f)
        #     print(feature_vector, file=f)
        #     print(self.inverse_document_frequency, file=f)
        #
        # raise Exception("done")

        return feature_vector


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> np.ndarray:
        feature_vector = np.zeros(self.indexer.__len__())
        for i in range(0, len(sentence) - 1):
            bigram = sentence[i] + ' ' + sentence[i + 1]
            if self.indexer.contains(bigram.lower()):
                ngram_index = self.indexer.index_of(bigram.lower())
                feature_vector[ngram_index] += 1
        return feature_vector


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer, inverse_document_frequency):
        self.indexer = indexer
        self.inverse_document_frequency = inverse_document_frequency

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> np.ndarray:
        feature_vector = np.zeros(self.indexer.__len__())
        for word in sentence:
            if self.indexer.contains(word.lower()):
                word_index = self.indexer.index_of(word.lower())
                idf_value = self.inverse_document_frequency[word_index]
                feature_vector[word_index] += idf_value

        # with open('output.txt', 'w') as f:
        #     np.set_printoptions(threshold=np.inf)
        #     print(self.indexer, file=f)
        #     print(sentence, file=f)
        #     print(feature_vector, file=f)
        #     print(self.inverse_document_frequency, file=f)
        #
        # raise Exception("done")

        return feature_vector

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weight_vector: np.ndarray, feature_extractor: FeatureExtractor, bias_vector: np.ndarray):
        self.weight_vector = weight_vector
        self.feature_extractor = feature_extractor
        self.bias = bias_vector

    def predict(self, sentence: List[str]) -> int:
        feature_vector = self.feature_extractor.extract_features(sentence)

        z = np.dot(feature_vector, self.weight_vector) + self.bias
        possibility = sigmoid(z)

        return 1 if possibility > 0.5 else 0


def sigmoid(x):
    x = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-x))

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    sentence_number = len(train_exs)
    indexer = feat_extractor.get_indexer()
    W = np.zeros((len(indexer),))
    b = np.zeros(1)
    learning_rate = 0.8
    reg_strength = 0.00
    loop_number = 500
    X = np.zeros((sentence_number, indexer.__len__()))
    Y = np.zeros(sentence_number)

    for i in range(sentence_number):
        X[i] = feat_extractor.extract_features(train_exs[i].words, False)
        Y[i] = train_exs[i].label

    Y = Y.T

    for j in range(loop_number):
        Z = np.dot(X, W) + b
        A = sigmoid(Z)
        # loss = (-1 / sentence_number) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        dW = 1 / sentence_number * X.T.dot(A - Y) + reg_strength * W
        db = 1 / sentence_number * np.sum(A - Y)
        W = W - learning_rate * dW
        b = b - learning_rate * db

    return LogisticRegressionClassifier(W, feat_extractor, b)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor

    nltk.download('stopwords')
    indexer = Indexer()
    stop_words = set(stopwords.words('english'))
    punkt = (',', '.', '...', '?', '\'', '\'\'', '!', ':', ';')
    # rare_words = ('the')
    document_frequency = {}
    min_word_freq = 1

    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here

        for sentence in train_exs:
            filtered_words = [word for word in sentence.words]
            filtered_words = list(dict.fromkeys(filtered_words))
            for word in filtered_words:
                document_frequency[word.lower()] = document_frequency.get(word.lower(), 0) + 1

        for sentence in train_exs:
            filtered_words = [word for word in sentence.words]
            filtered_words = list(dict.fromkeys(filtered_words))
            for word in filtered_words:
                if document_frequency[word.lower()] >= min_word_freq:
                    indexer.add_and_get_index(word.lower())

        document_frequency_array = np.array(list(document_frequency.values()))
        inverse_document_frequency = np.log(len(train_exs) / (1 + document_frequency_array))

        # with open('document_frequency.txt', 'w') as f:
        #     print(document_frequency_array, file=f)
        #     print(inverse_document_frequency, file=f)
        #
        # inverse_document_frequency = np.log(train_exs.__len__() / np.array(list(document_frequency.values())))

        feat_extractor = UnigramFeatureExtractor(indexer, inverse_document_frequency)

        # print(indexer)

    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here

        for sentence in train_exs:
            filtered_words = [word for word in sentence.words]
            for n in range(0, len(filtered_words) - 1):
                bigram = filtered_words[n] + ' ' + filtered_words[n + 1]
                indexer.add_and_get_index(bigram.lower())

        feat_extractor = BigramFeatureExtractor(indexer)

    elif args.feats == "BETTER":

        # Add additional preprocessing code here
        for sentence in train_exs:
            filtered_words = [word for word in sentence.words]
            filtered_words = list(dict.fromkeys(filtered_words))
            for word in filtered_words:
                document_frequency[word.lower()] = document_frequency.get(word.lower(), 0) + 1

        for sentence in train_exs:
            filtered_words = [word for word in sentence.words]
            filtered_words = list(dict.fromkeys(filtered_words))
            for word in filtered_words:
                if document_frequency[word.lower()] >= min_word_freq:
                    indexer.add_and_get_index(word.lower())

        document_frequency_array = np.array(list(document_frequency.values()))
        inverse_document_frequency = np.log(len(train_exs) / (1 + document_frequency_array))

        feat_extractor = BetterFeatureExtractor(indexer, inverse_document_frequency)

    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL or LR to run the appropriate system")
    return model