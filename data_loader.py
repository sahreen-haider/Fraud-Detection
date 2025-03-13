import tensorflow as tf
import tf.data as tfd



datasetb = tfd.experimental.make_csv_dataset(
    "data/fraudTrain.csv", batch_size=25, label_name="label", num_epochs=1
)

