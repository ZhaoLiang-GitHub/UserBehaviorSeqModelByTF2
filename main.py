import os

import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.data_progress import DataProgress
from config import Config
from models.model import *


def run():
    config = Config()
    data = DataProgress()
    data.tfrecord2tensor()
    din = DIN()
    model = din.model()
    model.summary()
    early_stop = EarlyStopping(monitor='val_auc', min_delta=0.0005, patience=3, mode='max', restore_best_weights=True)
    model_ckp = ModelCheckpoint(filepath=os.path.join(config.ckpt_path, 'cp-{epoch:04d}.ckpt'), monitor='val_auc',
                                mode='max', save_best_only=True)
    tb = keras.callbacks.TensorBoard(log_dir=config.log_path)

    callbacks = [model_ckp, early_stop, tb]
    opt = tf.optimizers.Adam(0.001)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  experimental_run_tf_function=False,
                  metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model.fit(data.train_tensor, validation_data=data.valid_tensor, epochs=config.epochs, callbacks=callbacks,
              verbose=1)
    print('training finished')


if __name__ == '__main__':
    run()
