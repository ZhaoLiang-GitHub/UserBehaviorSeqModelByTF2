# -*-coding:utf-8-*-
# create by zhaoliang19960421@outlook.com on 2021/12/21

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, Reshape

from config import *
from .layer import *


class UserBehaviorSeqBaseModel(Config):
    def __init__(self, *args, **kwargs):
        super(UserBehaviorSeqBaseModel, self).__init__(*args, **kwargs)
        self.embedding_layer = {}
        self.input_for_other = {}
        self.input_for_seq = {}
        self.input_for_candidate = {}
        self.all_input = {}
        self.seq_feature_length = -1

        self.bulid_layer()
        self.bulid_input()

    def bulid_layer(self):
        for feature in self.feature:
            name = feature.name
            feature_type = feature.feature_type
            data_type = feature.data_type
            length = feature.length
            progress = feature.progress
            embedding_size = feature.embedding_size
            category = feature.category
            if feature_type == "sequence":
                continue
            if data_type == "dense":
                if progress == "embedding":  # 浮点特征通过一个dense层，输出一个向量
                    self.embedding_layer[f"dense_{name}"] = Dense(units=embedding_size,
                                                                  activation=None,
                                                                  use_bias=False,
                                                                  kernel_initializer=glorot_normal(self.seed),
                                                                  kernel_regularizer=l2(self.l2_reg_embedding),
                                                                  name=f"dense_embedding_{name}")
            elif data_type == "sparse":
                self.embedding_layer[f"sparse_{name}"] = Embedding(input_dim=category + 1,
                                                                   input_length=1,
                                                                   output_dim=embedding_size,
                                                                   embeddings_initializer=glorot_normal(
                                                                       self.seed),
                                                                   embeddings_regularizer=l2(
                                                                       self.l2_reg_embedding),
                                                                   name=f"sparse_embedding_{name}")
            elif data_type in ["embedding"]:
                continue

    def feature2tensor(self, feature: Feature):
        name = feature.name
        feature_type = feature.feature_type
        data_type = feature.data_type
        progress = feature.progress
        embedding_size = feature.embedding_size
        length = feature.length
        if feature_type == "sequence":
            if data_type in ["dense", "sparse"]:
                return self.input_for_seq[name]

        temp_dict = {}
        if feature_type == "item":
            temp_dict = self.input_for_candidate
        elif feature_type == "other":
            temp_dict = self.input_for_other
        if data_type == "dense":
            if progress == "org":  # 浮点数直接计算
                return Reshape((1, embedding_size))(temp_dict[name])
            elif progress == "embedding":
                return Reshape((1, embedding_size))(self.embedding_layer[f"dense_{name}"](temp_dict[name]))
        elif data_type == "sparse":
            return Reshape((1, embedding_size))(self.embedding_layer[f"sparse_{name}"](temp_dict[name]))
        elif data_type == "embedding":
            return Reshape((1, length))(temp_dict[name])

    def get_tensor(self):
        tensor_for_candidate_list = []
        tensor_for_seq_list = []
        tensor_for_other_list = []
        seq_length = -1
        for feature in self.feature:
            feature_type = feature.feature_type
            if feature_type == "item":
                tensor_for_candidate_list.append(self.feature2tensor(feature))
            elif feature_type == "sequence":
                tensor_for_seq_list.append(self.feature2tensor(feature))
                seq_length = feature.length
            elif feature_type == "other":
                tensor_for_other_list.append(self.feature2tensor(feature))
        return tensor_for_candidate_list, tensor_for_seq_list, tensor_for_other_list, seq_length

    def bulid_input(self):
        for feature in self.feature:
            name = feature.name
            feature_type = feature.feature_type
            data_type = feature.data_type
            length = feature.length
            if feature_type == "sequence":
                if data_type in ["dense", "sparse"]:
                    self.input_for_seq[name] = Input((length,), name=name)
            elif feature_type == "other":
                if data_type in ["dense", "sparse"]:
                    self.input_for_other[name] = Input((1,), name=name)
                elif data_type == "embedding":
                    self.input_for_other[name] = Input((length,), name=name)
            elif feature_type == "item":
                if data_type in ["dense", "sparse"]:
                    self.input_for_candidate[name] = Input((1,), name=name)
                elif data_type == "embedding":
                    self.input_for_candidate[name] = Input((length,), name=name)
        self.all_input.update(self.input_for_seq)
        self.all_input.update(self.input_for_other)
        self.all_input.update(self.input_for_candidate)


class DIN(UserBehaviorSeqBaseModel):
    def __init__(self, *args, **kwargs):
        super(DIN, self).__init__(*args, **kwargs)

    def model(self):
        tensor_for_candidate_list, tensor_for_seq_list, tensor_for_other_list, seq_length = self.get_tensor()

        tensor_for_other = Concatenate(axis=2)(tensor_for_other_list)
        tensor_for_candidate = Concatenate(axis=2)(tensor_for_candidate_list)

        # attentionUnit4Din
        tensor_seq_org = Reshape((seq_length, len(tensor_for_seq_list)))(Concatenate(axis=-1)(tensor_for_seq_list))
        tensor_seq_atu = Concatenate(axis=2)([tensor_seq_org - tensor_for_candidate,
                                              tensor_seq_org,
                                              Concatenate(axis=1)(
                                                  [tensor_for_candidate for _ in range(seq_length)])])
        tensor_seq_alpha = Dense(units=1,
                                 kernel_initializer=glorot_normal(self.seed),
                                 kernel_regularizer=l2(self.l2_reg_embedding),
                                 trainable=True,
                                 use_bias=True,
                                 bias_initializer=Zeros(),
                                 name=f"AttentionUnit4DIN_hidden_2_layer",
                                 activation="linear",
                                 )(Dense(units=tensor_seq_org.shape[1] // 2,
                                         kernel_initializer=glorot_normal(self.seed),
                                         kernel_regularizer=l2(self.l2_reg_embedding),
                                         trainable=True,
                                         use_bias=True,
                                         bias_initializer=Zeros(),
                                         name=f"AttentionUnit4DIN_hidden_1_layer",
                                         activation="relu")(tensor_seq_atu))

        tensor_for_seq = tf.reduce_sum(tensor_seq_org * tensor_seq_alpha, axis=1, keepdims=True)

        mlp = MLP()
        all_input = Concatenate(axis=2)([tensor_for_other, tensor_for_candidate, tensor_for_seq])
        all_output = mlp(all_input)
        model = Model(inputs=self.all_input, outputs=all_output)
        return model


if __name__ == '__main__':
    din = DIN()
    din = din.model()
