# -*-coding:utf-8-*-
# create by zhaoliang19960421@outlook.com on 2021/12/17
import tensorflow as tf

from config import Config


class DataProgress(Config):
    def __init__(self, *args, **kwargs):
        super(DataProgress, self).__init__(*args, **kwargs)
        self.train_tensor = None
        self.test_tensor = None
        self.valid_tensor = None

    def tfrecord2tensor(self):
        feature_description = {self.label: tf.io.FixedLenFeature([1], tf.float32)}
        for feature in self.feature:
            name = feature.name
            feature_type = feature.feature_type
            data_type = feature.data_type
            length = feature.length
            if feature_type == "sequence":
                if data_type in ["dense", "sparse"]:
                    feature_description[name] = tf.io.FixedLenFeature([length], tf.float32,
                                                                      tf.constant(0.0, tf.float32, [length]))
            else:
                if data_type == "dense":
                    feature_description[name] = tf.io.FixedLenFeature([1], tf.float32, 0.0)
                elif data_type == "sparse":
                    feature_description[name] = tf.io.FixedLenFeature([1], tf.float32, 0.0)
                elif data_type == "embedding":
                    feature_description[name] = tf.io.FixedLenFeature([length], tf.float32,
                                                                      tf.constant(0.0, tf.float32, [length]))

        def _parse_function(exam_proto):  # 映射函数，用于解析一条example
            feature = tf.io.parse_single_example(exam_proto, feature_description)
            label = float(feature[self.label])
            feature.pop(self.label)
            return feature, label

        for name, file_name in [("train", self.train_path), ("test", self.test_path), ("valid", self.valid_path)]:
            ds_fs = tf.data.Dataset.list_files(file_pattern=file_name)
            fs = sorted(ds_fs.as_numpy_iterator())
            ds = tf.data.TFRecordDataset(filenames=fs) \
                .map(_parse_function) \
                .repeat(1) \
                .shuffle(10) \
                .batch(self.batch_size, drop_remainder=True)
            if name == "train":
                self.train_tensor = ds
            elif name == "test":
                self.test_tensor = ds
            elif name == "valid":
                self.valid_tensor = ds


if __name__ == '__main__':
    data_progress = DataProgress()
    data_progress.tfrecord2tensor()
    for line in data_progress.train_tensor.take(1):
        print(line[0])
