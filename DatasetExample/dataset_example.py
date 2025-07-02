import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

def get_info_pNEUMA(file_path: str, sampling_period:float, time_decimals:int, position_smoothing:float) -> dict:

    df = pd.read_csv(file_path)
    file_rows,inner_rows=[],[]
    
    for index,row in df.iterrows():
        file_rows.append(row.tolist())

    for row in file_rows:
        inner_rows.append(row[0].split(';')[:-1])

    iter_num = int(round(sampling_period*6/0.04,ndigits=0))

    vehicle_id,vtype=[],[]
    y,x,u,t=[],[],[],[]
    for i,row in enumerate(inner_rows):
        vehicle_id.append(int(row[0]) - 1)

        temp=str(row[1][1:])
        try:
            vtype.append(temp[:temp.index(' ')])
        except ValueError:
            vtype.append(str(row[1][1:]))

        y.append((gaussian_filter(list(map(float,row[4::iter_num])),sigma=position_smoothing)).tolist())
        x.append((gaussian_filter(list(map(float,row[5::iter_num])),sigma=position_smoothing)).tolist())
        u.append(list(map(float,row[6::iter_num])))
        t.append(np.round(list(map(float,row[9::iter_num])),decimals=time_decimals).tolist())
    
    Raw_VD = {'id':vehicle_id,'vtype':vtype,'x':x,'y':y,'time':t,'speed':u}
    
    return Raw_VD
