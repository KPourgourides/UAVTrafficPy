"""
This file is supplementary and NOT a core file of the UAVTrafficPy project.

Users can run this test file to perform some analysis and visualization tasks and check if it runs as expected.
The dataset used is in tests/dataset_example and is a small fraction from a pNEUMA dataset just to run the tests.
"""
import random
import sys
sys.path.append('C:/Users/kpourg01/Desktop/Work/Code/UAVTrafficPy')
from tool import uavtrafficpy
import dataload_example
tool =  uavtrafficpy.Master()
import numpy as np

raw_data = dataload_example.get_info_pneuma(file_path=r'tests/dataset_example.csv',sampling_period=0.2,time_decimals=1,position_smoothing=2)

"""necessary information that needs to be provided by the user to proceed with the tool"""
#=================================================================
start=min(min(set) for set in raw_data.get('time'))
end=max(max(set) for set in raw_data.get('time'))
#-----------------------------------------------------------------
time_axis = np.round(np.arange(start,end+1,1),decimals=1).tolist()
#=================================================================
ll_lat,ll_lon = 37.97811671602297, 23.733975874806358
lr_lat,lr_lon = 37.97876143771719, 23.735210640421347
ur_lat,ur_lon = 37.97938790250702, 23.734674127747873
ul_lat,ul_lon = 37.97874587304865, 23.73345028699124
#-----------------------------------------------------------------
intersection_center = (37.97866950849114, 23.734362398006162)
#-----------------------------------------------------------------
bbox = [(ll_lat,ll_lon),
        (lr_lat,lr_lon),
        (ur_lat,ur_lon),
        (ul_lat,ul_lon)]
#-----------------------------------------------------------------
spatio_temporal_info  =  {'bbox':bbox,'intersection center':intersection_center,'time axis':time_axis}
#=================================================================

"""loading tool features with filtered data"""
dataloader = tool.dataloader(raw_data,spatio_temporal_info)
data = dataloader.get_filtered_data()
analysis = tool.analysis(data,spatio_temporal_info)
visualization = tool.visualization(data,spatio_temporal_info)

vehicle_id = random.choice(data.get('id'))

"""testing some visualization functions"""
visualization.draw_trajectories()
visualization.draw_speed(vehicle_id)
visualization.draw_acceleration(vehicle_id)
visualization.draw_distance_travelled(vehicle_id)

"""testing some analysis functions"""
speed = analysis.get_speed()[data.get('id').index(vehicle_id)]
acceleration = analysis.get_acceleration()[data.get('id').index(vehicle_id)]
distance_travelled = analysis.get_distance_travelled()[data.get('id').index(vehicle_id)]

print('-'*200)
print(f'The vehicle with id {vehicle_id} had an average speed of {np.mean(speed):.2f}km/h, maximum acceleration {max(acceleration):.2f}m/s^2, and travelled a total of {max(distance_travelled):.2f} meters while in the recording. Thanks for testing this software!')
print('-'*200)
