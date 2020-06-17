import pandas as pd
from typing import Type
import numpy as np
from joblib import Memory
from nptyping import NDArray
from typing import Any, Tuple, Dict

memory = Memory("./joblib_cache")

Series = Type[pd.Series]
DataFrame = Type[pd.DataFrame]

SALES_TRAIN_VALIDATION = "data/sales_train_validation.csv"
CALENDAR = "data/calendar.csv"
SELL_PRICES = "data/sell_prices.csv"

ItemVector = NDArray[(30490), np.float32]
ItemArray = NDArray[(30490, Any), np.float32]
MultipleSeries = NDArray[(Any, Any), np.float32]


@memory.cache
def item_id() -> pd.Series:
    id = pd.read_csv(SALES_TRAIN_VALIDATION, usecols=["id"])
    id = id.squeeze()
    return id


@memory.cache
def item_store() -> Tuple[ItemArray, list]:
    store = pd.read_csv(SALES_TRAIN_VALIDATION,
                        usecols=["store_id"],
                        dtype='category')
    store = store.squeeze()
    return store.cat.codes.values, store.cat.categories


@memory.cache
def item_dept() -> Tuple[ItemArray, list]:
    dept = pd.read_csv(SALES_TRAIN_VALIDATION,
                       usecols=["dept_id"],
                       dtype='category')
    dept = dept.squeeze()
    return dept.cat.codes.values, dept.cat.categories


@memory.cache
def item_state() -> Tuple[ItemArray, list]:
    state = pd.read_csv(SALES_TRAIN_VALIDATION,
                        usecols=["state_id"],
                        dtype='category')
    state = state.squeeze()
    return state.cat.codes.values, state.cat.categories


@memory.cache
def item_category() -> Tuple[ItemArray, list]:
    c = pd.read_csv(SALES_TRAIN_VALIDATION,
                    usecols=["cat_id"],
                    dtype='category')
    c = c.squeeze()
    return c.cat.codes.values, c.cat.categories


@memory.cache
def item_kind() -> Tuple[ItemArray, list]:
    kind = pd.read_csv(SALES_TRAIN_VALIDATION,
                       usecols=["item_id"],
                       dtype='category')
    kind = kind.squeeze()
    return kind.cat.codes.values, kind.cat.categories


@memory.cache
def open_items_sale_data():
    path = SALES_TRAIN_VALIDATION
    columns = pd.read_csv(path, nrows=0).columns
    d_columns = [col for col in columns if col.startswith("d_")]
    return pd.read_csv(path, dtype={col: np.float16 for col in d_columns})


@memory.cache
def prices_per_item_over_time() -> ItemArray:
    week_day = pd.read_csv(CALENDAR, usecols=['wm_yr_wk', 'd'])
    items = pd.read_csv(SALES_TRAIN_VALIDATION,
                        usecols=['id', 'store_id', 'item_id'])
    p = pd.read_csv(SELL_PRICES)  # ,dtype={"sell_price":np.float16})
    p = p.merge(week_day, on="wm_yr_wk")
    p = p.merge(items, on=['store_id', 'item_id'])
    p = p.loc[:, ['id', 'd', 'sell_price']]
    p = p.pivot(index="id", columns="d", values="sell_price")
    items = open_items_sale_data()
    items = items.set_index("id")
    items = items.select_dtypes(np.float16)
    days = week_day.d.unique()
    for d in days:
        if d not in items.columns:
            items.loc[:, d] = 0.
    # assert items.shape == (30490, 1913)
    # assert p.shape ==  (30490, 1913)
    p = p.reindex_like(items)
    p = p.fillna(method="backfill", axis=1)
    return p.values.astype(np.float32)


@memory.cache
def unit_sales_per_item_over_time() -> ItemArray:
    s = open_items_sale_data()
    s = s.set_index("id")
    s = s.select_dtypes(np.float16)

    # week_day = pd.read_csv(CALENDAR, usecols=['wm_yr_wk', 'd'])
    # days = week_day.d.unique()
    # for d in days:
    #     if d not in items.columns:
    #         items.loc[:,d] = np.float16(0)

    return s.values.astype(np.float32)


@memory.cache
def item_weight() -> ItemVector:
    """
    The weights for each item are computed based on the difference of
    unit sales in the training data set.  This is used in the
    computation of RMSSE. It should not be confounded with the
    aggregation weights of WRMSSE.  This weight is introduced to make
    scale indifferent cost across the different time series.
    """
    s = unit_sales_per_item_over_time()
    s = s[:, 0 : -28 * 1].astype(np.float32)
    item_w = np.sqrt(np.mean(np.diff(s, axis=1) ** 2, axis=1))
    assert isinstance(item_w, ItemVector), ItemVector.type_of(item_w)
    return item_w


@memory.cache
def sales_weight() -> ItemVector:
    """
    These weights are the sum of dollar sales of the last 28 days in
    the training dataset.  They are used in the aggregation
    coefficients of WRMSSE.
    """
    s = unit_sales_per_item_over_time()
    p = prices_per_item_over_time()
    s = s[:, -28 * 1:]
    p = p[:, -28 * 3:-28 * 2]
    assert p.shape == s.shape, f"{p.shape} {s.shape}"
    ps = p.astype(np.float32) * s
    w = np.sum(ps, axis=1)
    # next, we can normalize the weights here instead of normalizing
    # inside WRMSSE, since the every product appears once and only
    # once in each aggregation
    w = w / sum(w)
    assert isinstance(w, ItemVector), ItemVector.type_of(w)
    return w


import tensorflow as tf


def tf_groupby(time_series: ItemArray, grouping_matrix: ItemVector):
    from tensorflow import constant, unique, size, SparseTensor
    time_series = constant(time_series, tf.dtypes.float32)
    grouping = constant(grouping_matrix)
    elements, _ = unique(grouping)
    grouping_lookup = SparseTensor(
        indices=[[v, i] for i, v in enumerate(grouping)],
        values=tf.ones(size(grouping)),
        dense_shape=[size(elements), size(grouping)],
    )

    # print(grouping_lookup)
    output = tf.sparse.sparse_dense_matmul(
        sp_a=grouping_lookup,
        b=time_series,
    )
    return output


NB_ITEMS, NB_DAYS = 30490, 1913


def compose_aggregation_matrices(A: ItemVector, B: ItemVector) -> ItemVector:
    """
    A and B are vectors of indices representing two different
    aggregations. If A has m different indices and B has n different
    indices. The aggregation composition will have m*n different
    indices. The length of A, B and output is the same.
    """
    output = np.zeros_like(A)
    nunique_A = len(np.unique(A))
    nunique_B = len(np.unique(B))
    output = A + B * nunique_A
    assert max(output) == nunique_A * nunique_B - 1
    return output


@memory.cache
def get_aggregation_matrices():
    """
    Returns the 12 aggregation matrices in vector format.
    """
    from itertools import product
    it_state, it_state_idx = item_state()
    it_store, it_store_idx = item_store()
    it_category, it_category_idx = item_category()
    it_dept, it_dept_idx = item_dept()
    it_kind, it_kind_idx = item_kind()

    output = []
    output.append(np.zeros((NB_ITEMS), dtype=int))
    for grouping in [it_state, it_store, it_category, it_dept]:
        output.append(grouping)
    for (gr_a, gr_b) in product([it_state, it_store], [it_category, it_dept]):
        grouping = compose_aggregation_matrices(gr_a, gr_b)
        output.append(grouping)
    output.append(it_kind)
    output.append(compose_aggregation_matrices(it_kind, it_state))
    output.append(np.arange(NB_ITEMS))

    return output


@memory.cache
def unit_sales_aggregation() -> Dict[int, MultipleSeries]:
    """
    Here, we compute the aggregations used in the M5 evaluation. Each
    aggregation corresponds to the operation groupby-sum. The
    aggregations are split in 12 different levels. For each level, we
    compute an array, in which rows are the elements of the
    aggregation, columns are time, and the value is the sum of unit
    sales of the given element on that day. Arrays in each level have
    different number of rows, but the same number of columns.

    | id | Aggregation Level                                      | Series |
    | -- | -----------------                                      | ------ |
    | 1  | all products, aggregated for all stores/states         | 1      |
    | 2  | all products, aggregated for each State                | 3      |
    | 3  | all products, aggregated for each store                | 10     |
    | 4  | all products, aggregated for each category             | 3      |
    | 5  | all products, aggregated for each department           | 7      |
    | 6  | all products, aggregated for each State and category   | 9      |
    | 7  | all products, aggregated for each State and department | 21     |
    | 8  | all products, aggregated for each store and category   | 30     |
    | 9  | all products, aggregated for each store and department | 70     |
    | 10 | product x, aggregated for all stores/states            | 3,049  |
    | 11 | product x, aggregated for each State                   | 9,147  |
    | 12 | product x, aggregated for each store                   | 30,490 |
    |    | total                                                  | 42,840 |
    """
    # TODO: make tf version
    # TODO: include weights
    all_sales = unit_sales_per_item_over_time()
    it_state, it_state_idx = item_state()
    it_store, it_store_idx = item_store()
    it_category, it_category_idx = item_category()
    it_dept, it_dept_idx = item_dept()
    it_kind, it_kind_idx = item_kind()

    aggregations = {}

    # level 1: aggregate all
    agg = np.sum(all_sales, axis=0, keepdims=True, dtype=np.float32)
    assert agg.shape == (1, NB_DAYS), agg.shape
    aggregations[1] = agg

    # level 2,3,4,5: groupby state, store, category, department
    for i, grouping in enumerate([it_state, it_store, it_category, it_dept]):
        i += 2
        agg = tf_groupby(all_sales, grouping)
        assert agg.shape == (len(np.unique(grouping)), NB_DAYS), agg.shape
        aggregations[i] = agg
    # level 6,7,8,9: state.category, state.dept, store.category,store.dept
    from itertools import product
    for i, (gr_a, gr_b) in enumerate(
            product([it_state, it_store], [it_category, it_dept])):
        i += 6
        grouping = compose_aggregation_matrices(gr_a, gr_b)
        agg = tf_groupby(all_sales, grouping)
        assert agg.shape == (len(np.unique(grouping)), NB_DAYS), agg.shape
        aggregations[i] = agg

    # level 10
    grouping = it_kind
    agg = tf_groupby(all_sales, grouping)
    assert agg.shape == (len(np.unique(grouping)), NB_DAYS), agg.shape
    aggregations[10] = agg

    # level 11
    grouping = compose_aggregation_matrices(it_kind, it_state)
    agg = tf_groupby(all_sales, grouping)
    assert agg.shape == (
        len(np.unique(grouping)),
        NB_DAYS), f"{agg.shape} == ({len(np.unique(grouping))}, {NB_DAYS})"
    aggregations[11] = agg

    # level 12
    assert all_sales.shape == (NB_ITEMS, NB_DAYS), all_sales.shape
    aggregations[12] = all_sales

    return aggregations


@memory.cache
def reduced_calendar():
    c = pd.read_csv(
        CALENDAR,
        usecols=['d', 'wday', 'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI'],
    )
    return c
