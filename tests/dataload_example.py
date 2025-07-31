"""
This is a supplementary file and NOT a core part of the UAVTrafficPy project.

This file acts as an example and provides a function that makes appropriate transformations to the format of
the pNEUMA dataset, such that it is compatible with UAVTrafficPy.
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

def get_info_pneuma(file_path: str, sampling_period:float, time_decimals:int, position_smoothing:float) -> dict:

    """
    description
    -----------
    this function converts any pNEUMA dataset into the correct format for UAVTrafficPy

    arguments
    ---------
    1) the file path
    2) sampling period; how frequently to sample data, smallest possible value
        for pNEUMA is 0.04s
    3) time decimals for time axis; e.g., if sampling period is 0.04, decimals should be 2
        if sampling period is 0.2, decimals should be 1
    4) smoothing amount on positions with gaussian filter

    output
    ------
    data directory with the correct format
    """
    data_frame = pd.read_csv(file_path)
    file_rows,inner_rows=[],[]
    for _index,row in data_frame.iterrows():
        file_rows.append(row.tolist())

    for row in file_rows:
        inner_rows.append(row[0].split(';')[:-1])

    iter_num = int(round(sampling_period*6/0.04,ndigits=0))

    vehicle_id,vehicle_type=[],[]
    y,x,u,t=[],[],[],[]
    for row in inner_rows:
        vehicle_id.append(int(row[0]) - 1)

        temp=str(row[1][1:])
        try:
            vehicle_type.append(temp[:temp.index(' ')])
        except ValueError:
            vehicle_type.append(str(row[1][1:]))

        y.append((gaussian_filter(list(map(float,row[4::iter_num])),sigma=position_smoothing)).tolist())
        x.append((gaussian_filter(list(map(float,row[5::iter_num])),sigma=position_smoothing)).tolist())
        u.append(list(map(float,row[6::iter_num])))
        t.append(np.round(list(map(float,row[9::iter_num])),decimals=time_decimals).tolist())
    raw_data = {'id':vehicle_id,'vtype':vehicle_type,'x':x,'y':y,'time':t,'speed':u}
    return raw_data
