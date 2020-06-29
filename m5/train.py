from m5.debug import *
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from m5.feature import (
    item_category,
    item_dept,
    item_state,
    reduced_calendar,
    unit_sales_per_item_over_time,
    item_weight,
    item_store,
    item_kind,
    prices_per_item_over_time,
    events_calendar,
)


from m5.params import (
    batch_size,
    logdir,
    nb_epochs,
    training_days,
    training_range,
    evaluation_range,
    submission_range,
    nb_items,
    steps_per_epoch,
)  # Training parameters

#  cache functions output in module global variables to improve performance
unit_sales_per_item_over_time = unit_sales_per_item_over_time()
item_category, _ = item_category()
item_dept, _ = item_dept()
item_kind, _ = item_kind()
reduced_calendar = reduced_calendar()
item_weight = item_weight()
# load item_state as pandas series to use look_up function
# TODO: perform lookup using only numpy
state_series = pd.Categorical.from_codes(*item_state())
item_state, _ = item_state()
item_store, _ = item_store()
prices_per_item_over_time = prices_per_item_over_time()
events, _ = events_calendar()


def make_batch(items_index, days_index):
    """Computes one batch for given items and days.
    Supported Features:
        category : categorical #3
        dept     : categorical #7
        kind     : categorical #3049
        weekday  : int 0..6
        month    : int 0..11
        year     : int ?
        snap     : boolean
        state    : int 0..2
    """
    assert len(items_index) == len(days_index)
    feat_item_category = item_category.take(items_index)
    feat_item_dept = item_dept.take(items_index)
    feat_item_kind = item_kind.take(items_index)
    feat_calendar = reduced_calendar.take(days_index)
    feat_item_state = item_state.take(items_index)
    feat_item_store = item_store.take(items_index)
    # split feat_calendar into feat_calendar and snap_df
    # then, rename snap_df columns to match the values in state_series
    snap_df = feat_calendar[["snap_CA", "snap_TX", "snap_WI"]]
    feat_calendar = feat_calendar.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    snap_df = snap_df.rename(
        columns={"snap_CA": "CA", "snap_TX": "TX", "snap_WI": "WI"}
    )

    # remove duplicated items in snap_df
    # note that drop_duplicates applies only to columns
    snap_df = snap_df.reset_index().drop_duplicates().set_index("index")

    # bi-dimensional lookup on snap_df to get snap information for
    # given items and given days
    feat_snap = snap_df.lookup(days_index, state_series[items_index])

    features = dict(
        category=feat_item_category,
        dept=feat_item_dept,
        kind=feat_item_kind,
        weekday=feat_calendar.wday.values,
        month=feat_calendar.month.values,
        year=feat_calendar.year.values,
        snap=feat_snap,
        state=feat_item_state,
        store=feat_item_store,
        days_index=np.asarray(days_index),
        items_index=np.asarray(items_index),
    )

    try:
        target = unit_sales_per_item_over_time[items_index, days_index]
    except IndexError:
        target = None

    weight = item_weight[items_index]
    return features, target, weight


def index_generator(days_range, batch_size):
    """ Generates args for make_batch function so that each call of
    make_batch returns a batch of given size, and successive calls cover the
    whole day_range for all items. Samples are ordered in two axis: items and
    days. In the resulting sequence, equal items are contiguous.
    """
    # TODO: change index order to generate day wise batch
    from more_itertools import take

    full_sequence_items = (
        items for items in range(nb_items) for _ in range(*days_range)
    )
    full_sequence_days = (day for _ in range(nb_items) for day in range(*days_range))
    while True:
        items_index = take(batch_size, full_sequence_items)
        days_index = take(batch_size, full_sequence_days)
        assert len(items_index) == len(days_index)
        if not items_index:
            return  # end of iteration
        yield items_index, days_index


def batch_generator(mode, batch_size=None):
    """ This generator returns either random samples from the training
    set or ordered samples from the validation dataset. Each batch is
    a tuple (x,y,w) If mode is:
      - 'train', it generates random samples from the training set.
      - 'evaluation', it generates ordered samples from the validation
        set.
      - 'submission', it generates ordered samples from the submission
        set.
    When samples are ordered, each batch corresponds to the ordered
    days of a unique item, which respects the natural order needed to
    write the final csv.
    UPDATE: in this branch, each batch corresponds to all items of a given day.
    days are generated in order
    """
    items_index = list(range(30490))

    if mode == "train":
        for days in range(*training_range):
            days_index = [days] * len(items_index)
            yield make_batch(items_index, days_index)
    elif mode == "evaluation":
        for days in range(*evaluation_range):
            days_index = [days] * len(items_index)
            yield make_batch(items_index, days_index)
    elif mode == "submission":
        for days in range(*evaluation_range):
            days_index = [days] * len(items_index)
            yield make_batch(items_index, days_index)
        # TODO: implement submission after evaluation is OK
        # for index_tuple in index_generator(submission_range, batch_size):
        #     yield make_batch(*index_tuple)


import m5.layers


class StModel(tf.keras.models.Model):
    """
    This model predicts the unit sale for a given product_id, in a
    given day.  There are 30490 different product_id while the time
    span in this dataset cover almost 2000 days.

    When submitting the prediction, this model should be evaluated 28
    times for each product_id, one for each day in the submission
    range. The inputs of this model are defined in function make_batch
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.category = layers.Embedding(
            input_dim=3, output_dim=3, input_length=1, name="category",
        )
        self.dept = layers.Embedding(
            input_dim=7, output_dim=3, input_length=1, name="dept",
        )
        self.kind = layers.Embedding(
            input_dim=3049, output_dim=5, input_length=1, name="kind",
        )
        self.weekday = layers.Embedding(
            input_dim=8, output_dim=3, input_length=1, name="weekday",
        )
        self.month = layers.Embedding(
            input_dim=13, output_dim=3, input_length=1, name="month",
        )
        self.year = layers.Embedding(input_dim=8, output_dim=3, input_length=1,)
        self.snap = layers.Embedding(
            input_dim=2, output_dim=1, input_length=1, name="snap",
        )
        self.state = layers.Embedding(
            input_dim=3, output_dim=3, input_length=1, name="state"
        )
        self.store = layers.Embedding(
            input_dim=10, output_dim=3, input_length=1, name="store"
        )
        self.price = m5.layers.Price(prices_per_item_over_time)
        self.events = m5.layers.Events(events)
        self.all_together = layers.Concatenate(
            axis=1
        )  # axis=1 because axis 0 is batch dimension
        # self.DNN = tf.keras.Sequential(
        #     [
        #         layers.Dense(units=units, activation=tf.keras.activations.linear),
        #         layers.Dense(units=units, activation=tf.keras.activations.relu),
        #         layers.Dense(units=units, activation=tf.keras.activations.linear),
        #         layers.Dense(units=units, activation=tf.keras.activations.relu),
        #         layers.Dense(units=units, activation=tf.keras.activations.linear),
        #         layers.Dense(1, activation=tf.keras.activations.relu),
        #     ],
        #     name="DNN",
        # )
        self.rnn = layers.GRU(
            units=units, activation=tf.keras.activations.sigmoid, stateful=True
        )
        self.dense = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """
        In this model, inputs is a dict. The available features are those
        returned by function make_batch (check its docstring).
        """
        # make_batch
        category = self.category(inputs["category"])
        dept = self.dept(inputs["category"])
        kind = self.kind(inputs["kind"])
        weekday = self.weekday(inputs["weekday"])
        month = self.month(inputs["month"])
        snap = self.snap(inputs["snap"])
        year = self.year(tf.math.add(inputs["year"], -2011))
        state = self.state(inputs["state"])
        store = self.store(inputs["store"])
        events = self.events(inputs)
        prices, mean_price, relative_price = self.price(inputs)
        # mean_price = tf.reshape(mean_price, [-1, 1])
        relative_price = tf.expand_dims(relative_price, axis=-1)
        all_together = self.all_together(
            [
                category,
                dept,
                kind,
                weekday,
                month,
                year,
                snap,
                state,
                store,
                # mean_price,
                events,
                relative_price,
            ]
        )

        all_together = tf.expand_dims(all_together, axis=1)

        rnn = self.rnn(all_together)
        output = self.dense(rnn)
        return output


print("building model")
model = StModel(units=256, name="RecurrentModel",)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
print("testing model with one single batch")
x, y, w = next(batch_generator(mode="train", batch_size=128))
model(x)
model.summary()
# TODO: implement alternative summary method
print("model summary may appear incomplete if using subclassed model definition.")

# Training


def increment_step():
    step = tf.summary.experimental.get_step()
    if step is None:
        tf.summary.experimental.set_step(1)
    else:
        tf.summary.experimental.set_step(step + 1)
    return tf.summary.experimental.get_step()


# @tf.function()
def train_batch(model, X, Y, w):

    with tf.GradientTape() as tape:
        H = model(X)
        H = tf.squeeze(H)
        loss = tf.math.squared_difference(Y, H)
        # loss = H - Y * tf.math.log(H)
        # loss = tf.math.log(tf.cosh(H - Y))
        loss *= w
        # loss = tf.reduce_mean(loss)
        loss = tf.reduce_sum(loss)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


def evaluate_now(evaluation_period=100) -> bool:
    """
    This method should be used to set evaluation independently of the epoch. It
    returns True if the current iteration is a multiple of evaluation_period.
    """
    step = tf.summary.experimental.get_step()
    return (step % evaluation_period) == 0


def evaluate(model):
    """ Computes the prediction H and the loss for the whole evaluation range.
    Returns:
    H: is a TF array in which the number of rows corresponds to the number of
    items, whereas the number of columns corresponds to the number of days in
    the evaluation range.
    loss: is a TF scalar. The mean of the loss over all evaluation points is
    computed inside this function.
    """
    # TODO: there is a bug in evaluation, but the training seems ok
    print("\tEntering in evaluate function.")
    model.reset_states()
    for (X, Y, w) in batch_generator(mode="training"):
        H = model(X)
    Hlist = []
    Ylist = []
    for (X, Y, w) in batch_generator(mode="evaluation"):
        H = model(X)

        H = tf.squeeze(H)
        Y = tf.squeeze(Y)
        w = tf.squeeze(w)
        Hlist.append(H)
        Ylist.append(Y)
    H = tf.stack(Hlist)
    Y = tf.stack(Ylist)
    H = tf.transpose(H)
    Y = tf.transpose(Y)

    tf.debugging.assert_shapes(
        [(H, (30490, 28,)), (Y, (30490, 28,)), (w, (30490,)),]
    )
    # H = tf.reshape(H, (30490, 28))
    # Y = tf.reshape(Y, (30490, 28))
    # w = tf.reshape(w, (30490, 28))
    # w = w[:, 0]

    # plt.plot(tf.reduce_mean(H, axis=1))
    sns.jointplot(H, Y)

    loss = tf.reduce_mean(tf.square(Y - H), axis=1)
    loss = tf.sqrt(loss)
    loss = loss * w
    loss = tf.reduce_sum(loss)
    tf.debugging.assert_scalar(loss)
    return H, loss, Y


def write_output(H, mode="evaluation"):
    """ Writes the TF array H to csv using a pandas DataFrame
    """
    H = H.numpy()
    assert H.shape == (30490, 28)
    # using pandas fo IO
    import pandas as pd

    # create columns
    df = pd.DataFrame(H, columns=[f"F{i}" for i in range(1, 1 + 28)])
    # create index
    from m5.feature import item_id

    df.insert(0, "id", item_id())
    # write to file
    step = tf.summary.experimental.get_step()
    df.to_csv(f"data/{mode}/output_on_step{step}.csv", index=False)


def submit(model):
    print("writing submission file")
    Hlist = []
    for (X, _, w) in batch_generator(mode="submission", batch_size=30490 * 28):
        H = model(X)
        H = tf.squeeze(H)
        Hlist.append(H)
    H = tf.concat(Hlist, axis=0)
    assert H.shape == (30490 * 28,)
    H = tf.reshape(H, (30490, 28))
    write_output(H, mode="submission")


def train_model():
    with tf.summary.create_file_writer(logdir).as_default():
        for epoch in range(nb_epochs):
            print(f"Epoch: {epoch}")

            for (X, Y, w) in tqdm(batch_generator(mode="train", batch_size=batch_size)):
                model.reset_states()
                increment_step()

                batch_loss = train_batch(model, X, Y, w)
                # tf.summary.scalar(
                #     f"{model.name}_batch_loss", tf.reduce_mean(batch_loss)
                # )

            # Evaluate: record loss on tensorboard and write predictions for
            # evaluation period in csv format
            H, loss, _ = evaluate(model)
            tf.summary.scalar(f"{model.name}_eval_loss", tf.reduce_mean(loss))
            # write_output(H, "evaluation")
            print(f"loss: {loss}")

    print("finish train_model")


nb_epochs = 10
batch_size = 1024 * 16
partial_set_factor = 1  # set to higher values for fast training
steps_per_epoch = (30000 * 2000) // (batch_size * partial_set_factor)
train_model()
submit(model)
