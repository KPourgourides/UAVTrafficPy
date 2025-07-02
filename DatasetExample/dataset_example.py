"""
ok
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

def get_info_pneuma(file_path: str, sampling_period:float, time_decimals:int, position_smoothing:float) -> dict:

    """
    ok
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
    raw_vd = {'id':vehicle_id,'vtype':vehicle_type,'x':x,'y':y,'time':t,'speed':u}
    return raw_vd
