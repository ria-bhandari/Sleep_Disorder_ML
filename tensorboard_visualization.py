# In this file, we will use tensorboard to visualize the tabular data

import tensorflow as tf
from tensorboard.plugins import projector 
from tensorboard.plugins.hparams import api as hp
import pandas as pd

sleep_data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
dataset = tf.data.Dataset.from_tensor_slices(dict(sleep_data)) # converting from pandas dataframe to tensorflow dataset

# Summary writer - to match tensorboard 
log_directory = "log"
summary_writer = tf.summary.create_file_writer(log_directory)

# log the data to TensorBoard
with summary_writer.as_default():
    for index, row in enumerate(dataset):
        tf.summary.scalar("Age", row["Age"], step=index)
        tf.summary.scalar("Sleep-Duration", row["Sleep Duration"], step=index)
        tf.summary.text("Gender", row["Gender"], step=index)
        tf.summary.text("Occupation", row["Occupation"], step=index)

summary_writer.close()



