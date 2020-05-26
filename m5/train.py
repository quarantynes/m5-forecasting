from tqdm import tqdm
import tensorflow as tf

# Env setup
import m5.setup

# Training parameters
from m5.params import (batch_size, model_name, logdir, nb_epochs,
                       evaluation_period)

# Model definition
from m5.models import ModelV1

model = ModelV1(
    gru_units=8,
    dnn_units=32,
    nonCuda=False,
    name=model_name,
)

# Data loading
from m5.data_loading import batch_generator


# IO
def write_output(H, dir="evaluation"):
    import pandas as pd
    df = pd.DataFrame(H.numpy(), columns=[f"F{i}" for i in range(1, 1 + 28)])
    from m5.feature import item_id
    df.insert(0, "id", item_id())
    step = tf.summary.experimental.get_step()
    df.to_csv(f"data/{dir}/output_on_step{step}.csv", index=False)


# Training
def increment_step():
    step = tf.summary.experimental.get_step()
    if step is None:
        tf.summary.experimental.set_step(0)
    else:
        tf.summary.experimental.set_step(step + 1)


def train_batch(model, X, Y, w):
    with tf.GradientTape() as tape:
        H = model(X)
        loss = tf.losses.mean_squared_error(Y, H)
        loss = loss**0.5
        loss = loss * w
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


def evaluate(model, eval_slice=slice(-28, None)):
    loss_list = []
    Hlist = []
    for (X, Y, w) in (batch_generator(eval_data=True, batch_size=10)):
        H = model(X)
        H = H[:, eval_slice]
        Y = Y[:, eval_slice]
        loss_i = tf.losses.mean_squared_error(Y, H)
        loss_i = loss_i**0.5
        loss_i = loss_i * w
        loss_list.append(loss_i)
        Hlist.append(H)
    loss = tf.concat(loss_list, axis=0)
    H = tf.concat(Hlist, axis=0)
    return H, loss


def evaluate_now(evaluation_period=evaluation_period) -> bool:
    step = tf.summary.experimental.get_step()
    return (step % evaluation_period) == 0


# Training Loop
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

with tf.summary.create_file_writer(logdir).as_default():
    for epoch in range(nb_epochs):
        print(f"Epoch: {epoch}")

        for (X, Y, w) in tqdm(batch_generator(batch_size=batch_size)):
            increment_step()
            batch_loss = train_batch(model, X, Y, w)
            tf.summary.scalar(f"{model.name}_batch_loss",
                              tf.reduce_mean(batch_loss))

            if evaluate_now():
                H, loss = evaluate(model)
                tf.summary.scalar(f"{model.name}_eval_loss",
                                  tf.reduce_mean(loss))
                write_output(H, "evaluation")

# Save Model
# TODO: save model here and implement prediction/submission
X, Y, w = batch_generator(eval_data=True, batch_size=batch_size)
model.save(f"data/saved_models/{model.name}.tf")
