"""
This file is supplementary and NOT a core file of the UAVTrafficPy project.

Users can run this test file to perform some analysis and visualization tasks and check if it runs as expected.
"""

from UAVTrafficPy.tool import uavtrafficpy
tool =  uavtrafficpy.Master()
import dataload_example
import numpy as np


raw_data = dataload_example.get_info_pneuma(file_path=r'tests/dataset_example.csv',sampling_period=0.2,time_decimals=1,position_smoothing=2)

"""necessary information that needs to be provided by the user to proceed with the tool"""

start=min(min(set) for set in raw_data.get('time'))
end=max(max(set) for set in raw_data.get('time'))

#=================================================================
time_axis = np.round(np.arange(start,end+1,1),decimals=1).tolist()

ll_lat,ll_lon = 37.97811671602297, 23.733975874806358
lr_lat,lr_lon = 37.97876143771719, 23.735210640421347
ur_lat,ur_lon = 37.97938790250702, 23.734674127747873
ul_lat,ul_lon = 37.97874587304865, 23.73345028699124

intersection_center = (37.97866950849114, 23.734362398006162)
bbox = [(ll_lat,ll_lon),
        (lr_lat,lr_lon),
        (ur_lat,ur_lon),
        (ul_lat,ul_lon)]
#=================================================================

spatio_temporal_information  =  {'wgs':True,'bbox':bbox,'intersection center':intersection_center,'time axis':time_axis}

"""loading tool features with filtered data"""
dataloader = tool.dataloader(raw_data,spatio_temporal_information)
filtered_data = dataloader.get_filtered_data()
analysis = tool.analysis(filtered_data,spatio_temporal_information)
visualization = tool.visualization(filtered_data,spatio_temporal_information)

"""testing some analysis functions"""

speed = analysis.get_speed()
acceleration = analysis.get_acceleration()
distance_travelled = analysis.get_distance_travelled()
od_pairs = analysis.get_od_pairs()

"""testing some visualization functions"""
visualization.draw_trajectories()
visualization.draw_speed(vehicle_id=filtered_data.get('id')[0])
visualization.draw_acceleration(vehicle_id=filtered_data.get('id')[0])
visualization.draw_distance_travelled(vehicle_id=filtered_data.get('id')[0])
