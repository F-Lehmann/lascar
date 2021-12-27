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

"""
THis code is taken from ASCAD:

https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py

"""
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop


#### MLP Best model (6 layers of 200 units)
def mlp_best(
    input_shape, number_of_classes=256, nodes=None, layer_nb=5, dropout=0.2, lr=0.01
):

    if nodes == None:
        nodes = [200] * (layer_nb)
    else:
        layer_nb = len(nodes)

    model = Sequential()
    model.add(Dense(nodes[0], input_shape=input_shape, activation="relu"))
    if dropout:
        model.add(Dropout(dropout))

    for i in range(1, layer_nb):
        model.add(Dense(nodes[i], activation="relu"))
        if dropout:
            model.add(Dropout(dropout))

    model.add(Dense(number_of_classes, activation="softmax"))
    optimizer = RMSprop(lr=lr)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model


### CNN Best model
def cnn_best(input_shape=(700, 1), number_of_classes=256, lr=0.01):
    model=Sequential()
    # Block 1
    model.add(Conv1D(64, 11, activation="relu",input_shape=input_shape, padding="same", name="block1_conv1"))
    model.add(AveragePooling1D(2, strides=2, name="block1_pool"))
    # Block 2
    model.add(Conv1D(128, 11, activation="relu", padding="same", name="block2_conv1"))
    model.add(AveragePooling1D(2, strides=2, name="block2_pool"))
    # Block 3
    model.add(Conv1D(256, 11, activation="relu", padding="same", name="block3_conv1"))
    model.add(AveragePooling1D(2, strides=2, name="block3_pool"))
    # Block 4
    model.add(Conv1D(512, 11, activation="relu", padding="same", name="block4_conv1"))
    model.add(AveragePooling1D(2, strides=2, name="block4_pool"))
    # Block 5
    model.add(Conv1D(512, 11, activation="relu", padding="same", name="block5_conv1"))
    model.add(AveragePooling1D(2, strides=2, name="block5_pool"))
    # Classification block
    model.add(Flatten(name="flatten"))
    model.add(Dense(4096, activation="relu", name="fc1"))
    model.add(Dense(4096, activation="relu", name="fc2"))
    model.add(Dense(number_of_classes, activation="softmax", name="predictions"))

    optimizer = RMSprop(lr=lr)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model
