# This file is part of lascar
#
# lascar is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
# Copyright 2018 Manuel San Pedro, Victor Servant, Charles Guillemet, Ledger SAS - manuel.sanpedro@ledger.fr, victor.servant@ledger.fr, charles@ledger.fr

import pickle
from tensorflow.keras.utils import Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from . import GuessEngine
from . import PartitionerEngine


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, X, Y, batch_size=1000):
        """Initialization
        :param X: input data
        :param Y: output data
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.X = X
        self.Y = Y
        self.batch_size = batch_size

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and Y
        """

        # Generate data
        X = self.X[index * self.batch_size: (index + 1) * self.batch_size]
        Y = self.Y[index * self.batch_size: (index + 1) * self.batch_size]
        return X, Y

class ProfileEngine(PartitionerEngine):
    """
    ProfileEngine is a PartitionerEngine used to mount Profiled Side-Channel Attacks.

    A classifier has to be instanciated first, and will be updated by traces batches, and a partition function to separate them.
    The classifier can be either a sklearn classifier or a keras model (neural network).

    After the partition is done, the finalize() method output the classifier updated.
    It can hence be used by MatchEngine.

    see examples/attacks/classifier.py for an example.

    """

    def __init__(
        self,
        name,
        classifier,
        partition_function,
        partition_range,
        epochs=10,
        test_size=0.1,
        verbose=1,
        batch_size=128,
        callbacks=[],
    ):
        """

        :param name:
        :param classifier: instantieted sklearn classifier or compiled keras model
        :param partition_function:
        :param partition_range:
        :param epochs: only used when using keras model, will be passed to the keras .fit() method
        :param test_size: only used when using keras model, will be passed to the keras .fit() method
        :param verbose: only used when using keras model, will be passed to the keras .fit() method
        :param batch_size: only used when using keras model, will be passed to the keras .fit() method
        :param callbacks: only used when using keras model, will be passed to the keras .fit() method
        """

        import tensorflow.keras as keras
        import sklearn

        if not isinstance(classifier, sklearn.base.ClassifierMixin) and not (
            isinstance(classifier, keras.Model) and classifier._is_compiled
        ):
            raise ValueError(
                "Classifier should be a sklearn classifier or a compiled keras model."
            )

        PartitionerEngine.__init__(self, name, partition_function, partition_range, 1)
        self._classifier = classifier
        self.classifier_type = (
            "keras" if isinstance(classifier, keras.Model) else "sklearn"
        )

        self.output_parser_mode = None

        # Used only for keras model:
        self.test_size = test_size
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.callbacks = callbacks

    def _initialize(self):

        self._session._batch_size = self._session.container.number_of_traces
        self._session._thread_on_update = False
        if self.classifier_type == "keras":
            self._session._progressbar = None

        self._partition_count = np.zeros((self._partition_size,), dtype=np.double)

    def _update(self, batch):
        if self.classifier_type == "sklearn":
            self._update_sklearn_classifier(batch)
        else:
            self._update_keras_model(batch)

    def _update_sklearn_classifier(self, batch):
        partition_values = list(map(self._partition_function, batch.values))
        for v in partition_values:
            self._partition_count[self._partition_range_to_index[v]] += 1
        self._classifier.fit(batch.leakages, partition_values)

    def _update_keras_model(self, batch):
        from tensorflow.keras.utils import to_categorical

        Y = to_categorical(
            list(map(self._partition_function, batch.values)), self._partition_size
        )
        X_train, X_test, Y_train, Y_test = train_test_split(
            batch.leakages, Y, test_size=self.test_size
        )
        
        train_generator = DataGenerator(X_train, Y_train, batch_size=self.batch_size)
        test_generator = DataGenerator(X_test, Y_test, batch_size=self.batch_size)

        self.history = self._classifier.fit(
            train_generator,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_data=test_generator,
            callbacks=self.callbacks
        )

        self.score_train = self._classifier.evaluate(train_generator)
        self.score_test = self._classifier.evaluate(test_generator)

    def _finalize(self):
        return self._classifier


class MatchEngine(GuessEngine):
    """
    MatchEngine is a GuessEngine allowing to perform the matching phase of a Profiled Side-Channel Attack.

    Compute for each guess, the log_proba accumulated by the classifier along traces.

    see examples/attacks/classifier.py for an example.
    """

    def __init__(
        self, name, classifier, selection_function, guess_range, solution=None
    ):
        """

        :param name:
        :param classifier:
        :param selection_function:
        :param guess_range:
        :param solution:
        """

        GuessEngine.__init__(
            self, name, selection_function, guess_range, solution)
        self._classifier = classifier
        self.output_parser_mode = "max"

    def _initialize(self):
        self._log_probas = np.zeros((self._number_of_guesses,))
        self._session._batch_size = self._session.container.number_of_traces
        self._session._thread_on_update = False

    def _update(self, batch):

        y = np.array(
            [
                [self._function(d, guess) for guess in self._guess_range]
                for d in batch.values
            ]
        )

        if hasattr(self._classifier, "predict_log_proba"):
            log_probas = self._classifier.predict_log_proba(batch.leakages)
        # Has been deprecated in keras
        # elif hasattr(self._classifier, "predict_proba"):
        #    log_probas = np.log2(self._classifier.predict_proba(batch.leakages))
        elif hasattr(self._classifier, "predict"):
            log_probas = np.log2(self._classifier.predict(batch.leakages))
        else:
            raise ValueError(
                "the classifier should have either .predict_proba() or .predict_log_proba() or .predict() method"
            )

        log_probas = np.nan_to_num(log_probas, False)

        for i in range(len(batch)):
            self._log_probas += log_probas[i, y[i]]
            self._log_probas = np.nan_to_num(self._log_probas, False)

    def _finalize(self):
        return self._log_probas


def save_classifier(classifier, filename):
    if hasattr(classifier, "save"):  # keras save
        classifier.save(filename)
        return
    try:
        with open(filename, "wb") as f:  # sklearn save
            pickle.dump(classifier, f)
        return
    except e:
        pass

    raise ValueError("Classifier cant be saved %s %s." %
                     (classifier, filename))


def load_classifier(filename):
    try:  # sklearn load
        with open(filename, "rb") as f:
            return pickle.load(f)
    except:  # keras load
        from keras.models import load_model

        return load_model(filename)
