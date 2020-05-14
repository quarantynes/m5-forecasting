##
from m5.feature import series_weight,unit_sales_aggregation
import numpy as np

unit_sales_dict = unit_sales_aggregation()
for name, series_collection in unit_sales_dict.items():
    w = series_weight(series_collection)
    print(name,series_collection.shape,np.nanmax(w),w.shape)

##

