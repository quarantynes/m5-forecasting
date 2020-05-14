import pandas as pd
from typing import Type
import numpy as np
from joblib import Memory

memory = Memory("./joblib_cache")

Series = Type[pd.Series]
DataFrame = Type[pd.DataFrame]

SALES_TRAIN_VALIDATION = "~/projects/kaggle/m5-forecasting-accuracy/data/sales_train_validation.csv"
CALENDAR = "~/projects/kaggle/m5-forecasting-accuracy/data/calendar.csv"
SELL_PRICES = "~/projects/kaggle/m5-forecasting-accuracy/data/sell_prices.csv"


def open_items_sale_data():
    path = SALES_TRAIN_VALIDATION
    columns = pd.read_csv(path, nrows=0).columns
    d_columns = [col for col in columns if col.startswith("d_")]
    return pd.read_csv(path, dtype={col: np.float16 for col in d_columns})


def open_calendar_data():
    return pd.read_csv(CALENDAR, )


def open_prices_data():
    return pd.read_csv(SELL_PRICES, dtype={"sell_price": np.float16})


def inspect_cat(df: DataFrame):
    print(f"\nTotal number of elements: {len(df)}")
    for name1, col1 in df.iteritems():
        print(f"\nSection {name1}")
        print(f"\tUnique elements in {name1} : {col1.nunique()} ")
        for name2, col2 in df.iteritems():
            if name1 == name2:
                continue
            print(f"\tUnique elements in [{name1} {name2}] : {(col1 + col2).nunique()} ")
            print(f"\tEntropy in [{name1} {name2}] : {(col1 + col2).nunique() - max(col1.nunique(), col2.nunique())} ")


def extract_item_sales(df: DataFrame):
    items_sales = df.iloc[:, 6:].T
    items_sales.columns = df.id
    return items_sales.astype(np.float16)


def extract_item_groups(df: DataFrame):
    items_groups = df.iloc[:, 0:6]
    return items_groups


@memory.cache
def add_day_column_in_price_table(price: DataFrame, calendar: DataFrame) -> DataFrame:
    week_day = calendar.loc[:, ['wm_yr_wk', 'd']]
    return price.merge(week_day, on='wm_yr_wk', )


@memory.cache
def add_id_column_in_price_table(price: DataFrame, item_sales: DataFrame) -> DataFrame:
    item_sales = item_sales.loc[:, ['id', 'store_id', 'item_id']]
    return price.merge(item_sales, on=['store_id', 'item_id'], )
