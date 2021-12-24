# -*-coding:utf-8-*-
# create by zhaoliang19960421@outlook.com on 2021/12/22

from tensorflow.keras.initializers import Zeros, glorot_normal
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.regularizers import l2

from config import Config


class MLP(Config, Layer):
    def __init__(self, *args, **kwargs):
        Config.__init__(self, *args, **kwargs)
        Layer.__init__(self, *args, **kwargs)
        self.kernel = []
        self.bacthnorm = []
        self.dropout = []
        for i in range(len(self.hidden_layer_units)):
            self.kernel.append(Dense(units=self.hidden_layer_units[i],
                                     kernel_initializer=glorot_normal(self.seed),
                                     kernel_regularizer=l2(self.l2_reg_embedding),
                                     trainable=True,
                                     use_bias=True,
                                     bias_initializer=Zeros(),
                                     name=f"{i}_hidden_layer",
                                     activation=self.activation,
                                     ))
            self.bacthnorm.append(BatchNormalization())
            self.dropout.append(Dropout(rate=self.dropout_rate,
                                        seed=self.hidden_layer_units[i],
                                        trainable=True,
                                        ))
        self.output_layer = Dense(units=1,
                                  kernel_initializer=glorot_normal(self.seed),
                                  kernel_regularizer=l2(self.l2_reg_embedding),
                                  trainable=True,
                                  use_bias=True,
                                  bias_initializer=Zeros(),
                                  name=f"MLP_last_hidden_layer",
                                  activation=self.output_activation,
                                  )

    def call(self, inputs, **kwargs):
        output = inputs
        for i in range(len(self.hidden_layer_units)):
            w = self.kernel[i]
            bn = self.bacthnorm[i]
            dropout = self.dropout[i]
            output = bn(output)
            output = w(output)
            output = dropout(output)
        output = self.output_layer(output)
        return output

    def compute_output_shape(self, input_shape):
        return None, input_shape[1], 1


class AttentionUnit4DIN(Layer, Config):
    def __init__(self, *args, **kwargs):
        Layer.__init__(self, *args, **kwargs)
        Config.__init__(self, *args, **kwargs)
        self.kernel = []
        self.bacthnorm = []
        self.dropout = []

    def build(self, input_shape):
        super(AttentionUnit4DIN, self).build(input_shape)
        for i in range(len(self.hidden_layer_units)):
            self.kernel.append(Dense(units=self.hidden_layer_units[i],
                                     kernel_initializer=glorot_normal(self.seed),
                                     kernel_regularizer=l2(self.l2_reg_embedding),
                                     trainable=True,
                                     use_bias=True,
                                     bias_initializer=Zeros(),
                                     name=f"{i}_hidden_layer",
                                     activation=self.activation,
                                     )
                               )
            self.bacthnorm.append(BatchNormalization())
            self.dropout.append(Dropout(rate=self.dropout_rate,
                                        seed=self.hidden_layer_units[i],
                                        trainable=True,
                                        )
                                )

    def call(self, inputs, **kwargs):
        t1, t2 = inputs
        output = Concatenate(axis=-1)([t1 - t2, t1, t2])
        output = output
        for i in range(len(self.hidden_layer_units)):
            w = self.kernel[i]
            bn = self.bacthnorm[i]
            dropout = self.dropout[i]
            output = bn(output)
            output = w(output)
            output = dropout(output)
        output = Dense(units=1,
                       kernel_initializer=glorot_normal(self.seed),
                       kernel_regularizer=l2(self.l2_reg_embedding),
                       trainable=True,
                       use_bias=True,
                       bias_initializer=Zeros(),
                       name=f"AttentionUnit4DIN_last_hidden_layer",
                       activation="sigmoid",
                       )(output)
        return output

    def compute_output_shape(self, input_shape):
        return None, input_shape[1], 1


if __name__ == '__main__':
    atu = AttentionUnit4DIN()
    from data_progress import DataProgress

    data_progress = DataProgress()
    data_progress.tfrecord2tensor()
    for line in data_progress.train_tensor:
        print(atu([line[0]["vgg16_pca"], line[0]["vgg16_pca"]]))
