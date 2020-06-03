import os.path

window = 28
nb_items = 30490
nb_agg_items = 42840
gru_size = 10
batch_size = 16
days = 1913
logdir = os.path.expanduser("~/tf_log")
nb_epochs = 2
evaluation_period = 100

training_days = days - 28
validation_range = (training_days, training_days+28)
submission_range = (training_days+28, training_days+56)
