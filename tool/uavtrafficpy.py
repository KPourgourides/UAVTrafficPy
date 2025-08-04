"""
Copyright 2025, Konstantinos Pourgourides (kotsiosp2001@gmail.com)

This file is the core part of UAVTrafficPy.

UAVTrafficPy is free/open-source software and you can distriubute it and/or 
contribute to it / modify it under the terms of the European Union Public License
Version 1.2 (referred to as "the Licence"). You may not use this work except in
compliance with the Licence. You may obtain a copy of the Licence at:

https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12

UAVTrafficPy is distributed under the Licence on an "AS IS" basis,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND. See the Licence for the
specific language governing permissions and limitations under the Licence.
"""

from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from shapely import Point,Polygon
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

class Master:
    """
    description
    -----------
    this is the master class that includes all other classes of the tool
    """

    def __init__(self):
        """
        description
        -----------
        Initialization
        """

    def distances(self, initial_coordinates:tuple, final_coordinates:tuple):
        """
        description
        -----------
        class the distances class

        arguments
        ---------
        1) initial set of coordinates (y,x)
        2) final set of coordinates (y,x)

        output
        ------
        returns the distances class 
        """
        return self._distances(self, initial_coordinates, final_coordinates)


    def dataloader(self, raw_data:dict, spatio_temporal_info:dict):
        """
        description
        -----------
        calls the dataloader class

        arguments
        ---------
        1) initial data dictionary
        2) spatiotemporal info dictionary

        output
        ------
        returns the dataloader class 
        """
        return self._dataloader(self, raw_data, spatio_temporal_info)


    def analysis(self, data:dict, spatio_temporal_info:dict):
        """
        description
        -----------
        calls the analysis class

        arguments
        ---------
        1) data dictionary
        2) spatiotemporal info dictionary

        output
        ------
        returns the analysis class 
        """
        return self._analysis(self, data, spatio_temporal_info)


    def visualization(self, data:dict, spatio_temporal_info:dict):
        """
        description
        -----------
        calls the visualization class

        arguments
        ---------
        1) data dictionary
        2) spatiotemporal info dictionary

        output
        ------
        returns the visualization class
        """
        return self._visualization(self, data, spatio_temporal_info)

    class _distances:
        """
        description
        -----------
        this class is dedicated to tasks that invlove calculation of distances
        """

        def __init__(self, master:'Master', initial_coordinates:tuple, final_coordinates:tuple):
            """
            description
            -----------
            initialization of distances class

            arguments
            ---------
            1) master class
            2) initial set of coordinates (y,x)
            3) final set of coordinates (y,x)
            """
            self.master = master
            self.y_i,self.x_i = initial_coordinates
            self.y_f,self.x_f = final_coordinates
            self.factor = 2*np.pi*6371000/360

        def get_dx(self) -> float:
            """
            description
            -----------
            calculation of longitudinal distance between two points on the x axis

            output
            ------
            longitudinal distance between two points on the x axis in meters
            """
            return (self.factor*float(np.cos(np.deg2rad(self.y_i)))*(self.x_f - self.x_i))

        def get_dy(self) -> float:
            """
            description
            -----------
            calculation of latitudinal distance between two points on the y axis

            output
            ------
            latitudinal distance between two points on the y axis in meters
            """
            return (self.factor*(self.y_f - self.y_i))

        def get_distance(self) -> float:
            """
            description
            -----------
            calculation of distance between two points (Euclidean approximation for small distances)

            output
            ------
            distance between two points in meters
            """
            dx = self.get_dx()
            dy = self.get_dy()
            return pow((pow(dx,2) + pow(dy,2)),0.5)

    class _dataloader:
        """
        description
        -----------
        this class is dedicated to loading and filtering data
        """

        def __init__(self, master:'Master', raw_data:dict, spatio_temporal_info:dict):
            """
            description
            -----------
            initialization of dataloader class

            arguments
            ---------
            1) master class
            2) initial data dictionary
            3) spatiotemporal info dictionary

            notes
            -----
            Detailed info on the correct format of the input dictionaries can be found at https://github.com/KPourgourides/UAVTrafficPy/blob/main/usage%20example/intersection_pipeline_walkthrough.pdf
            """
            self.master = master
            needed_keys_data = ['id','vtype','x','y','time']
            needed_keys_info = ['bbox','intersection center','time axis']

            if any(key not in raw_data.keys() for key in needed_keys_data):
                raise KeyError(f'data dictionary needs keys {needed_keys_data} to work!')
            if any(key not in spatio_temporal_info.keys() for key in needed_keys_info):
                raise KeyError(f'spatio temporal info dictionary needs keys {needed_keys_info} to work!')

            self.spatio_temporal_info = spatio_temporal_info
            self.vehicle_id,self.vehicle_type,self.x,self.y,self.t  = itemgetter('id','vtype','x','y','time')(raw_data)
            self.bbox,self.intersection_center,self.time_axis = itemgetter('bbox','intersection center','time axis')(spatio_temporal_info)
            raw_data['speed']=[]
            self.u = self.master.analysis(raw_data,spatio_temporal_info).get_speed()
            raw_data['speed']=self.u
            self.raw_data = raw_data

        def get_data(self) -> dict:
            """
            output
            ------
            returns the original dataset plus speeds in case
            the user does not want to apply a spatial filter
            """
            return self.raw_data

        def get_bounded_data(self) -> dict:
            """
            description
            -----------
            function that trims the initial data dictionary only to include data from the area of interest

            output
            ------
            trimmed data dictionary
            """
            box = Polygon(self.bbox)
            id_,vtype_,x_,y_,t_,u_=[],[],[],[],[],[]
            for i,vec in enumerate(self.x):
                flag=False

                for j,_ in enumerate(vec):
                    if box.contains(Point(self.y[i][j], self.x[i][j])):
                        flag=True
                        index_start = j
                        break

                if flag:
                    for k,_ in enumerate(vec[index_start+1:]):
                        index_end=k+index_start+1
                        if box.contains(Point(self.y[i][index_end], self.x[i][index_end])):
                            continue
                        break

                    if len(self.t[i][index_start:index_end])>1:
                        id_.append(self.vehicle_id[i])
                        vtype_.append(self.vehicle_type[i])
                        x_.append(self.x[i][index_start:index_end])
                        y_.append(self.y[i][index_start:index_end])
                        t_.append(self.t[i][index_start:index_end])
                        u_.append(self.u[i][index_start:index_end])

            bounded_data ={'id':id_,'vtype':vtype_,'x':x_,'y':y_,'time':t_,'speed':u_}
            return bounded_data

        def get_filtered_data(self, cursed_ids=None) -> dict:
            """
            description
            -----------
            function that filters the intersection data dictionary to remove parked or unwanted vehicles

            arguments
            ---------
            1) (optional) list of ids of unwanted vehicles

            output
            ------
            filtered data dictionary

            notes
            -----
            this function can be skipped if there are no vehicles you want to remove from the dataset
            """
            if cursed_ids is None:
                cursed_ids=[]

            intersection_data  = self.get_bounded_data()
            vehicle_id,vehicle_type,x,y,t,u = itemgetter('id','vtype','x','y','time','speed')(intersection_data)
            id_,vtype_,x_,y_,t_,u_=[],[],[],[],[],[]

            for i,vec in enumerate(x):
                immobility = sum(1 if self.master.distances(initial_coordinates=(y[i][j-1],x[i][j-1]),final_coordinates=(y[i][j],x[i][j])).get_distance() <1e-4 else 0 for j,element in enumerate(vec[1:]))
                if vehicle_id[i] not in cursed_ids and immobility<=0.95*len(vec):
                    id_.append(vehicle_id[i])
                    vtype_.append(vehicle_type[i])
                    x_.append(x[i])
                    y_.append(y[i])
                    t_.append(t[i])
                    u_.append(u[i])

            filtered_data = {'id':id_,'vtype':vtype_,'x':x_,'y':y_,'time':t_,'speed':u_}
            return filtered_data

    class _analysis:
        """
        description
        -----------
        this class is dedicated to carrying out analysis tasks
        """

        def __init__(self, master:'Master', data:dict, spatio_temporal_info:dict):
            """
            description
            -----------
            initialization of analysis class

            arguments
            ---------
            1) master class
            2) trimmed/filtered data dictionary
            3) spatiotemporal info dictionary
            """
            self.master = master
            needed_keys_data = ['id','vtype','x','y','time','speed']
            needed_keys_info = ['bbox','intersection center','time axis']

            if any(key not in data.keys() for key in needed_keys_data):
                raise KeyError(f'data dictionary needs keys {needed_keys_data} to work!')
            if any(key not in spatio_temporal_info.keys() for key in needed_keys_info):
                raise KeyError(f'spatio temporal info dictionary needs keys {needed_keys_info} to work!')

            self.data = data
            self.spatio_temporal_info = spatio_temporal_info
            self.vehicle_id,self.vehicle_type,self.x, self.y, self.t, self.u = itemgetter('id','vtype','x','y','time','speed')(data)
            self.bbox,self.center,self.time_axis = itemgetter('bbox','intersection center','time axis')(spatio_temporal_info)
            self.y_center,self.x_center=self.center
            self.detector_positions=None
            self.flow_info=None
            self.normalized_flow=None
            self.trafficlightphases=None
            self.lane_info=None
            self.sorted_id=None
            self.gaps=None
            self.d_from_edge=None
            self.flow_direction=None
            self.cycles=None
            self.traffic_light_phases=None

        def get_distance_travelled(self) ->list:
            """
            description
            -----------
            function that calculates the cumulative distance travelled per vehicle in meters per time step of each vehicle

            output
            ------
            list of lists; each nested list corresponds to a different vehicle, and includes the information written in the description
            """
            distance_travelled=[]

            for i,vec in enumerate(self.x):
                temp_distance_travelled=[]
                temp_sum=0
                for j,_ in enumerate(vec):
                    if j==0:
                        temp_distance_travelled.append(0)
                    else:
                        temp_sum+= self.master.distances(initial_coordinates=(self.y[i][j-1],self.x[i][j-1]),final_coordinates=(self.y[i][j],self.x[i][j])).get_distance()
                        temp_distance_travelled.append(temp_sum)
                distance_travelled.append(temp_distance_travelled)

            return distance_travelled

        def get_speed(self,km_per_h=True) -> list:
            """
            description
            -----------
            function that calculates speed per vehicle per time step of each vehicle, based on cumulative distance travelled

            arguments
            ---------
            1) (optional) whether the speed will be calculated in km per h or m/s

            output
            ------
            list of lists; each nested list corresponds to a different vehicle, and includes the information written in the description
            """
            distance_travelled = self.get_distance_travelled()
            multiplier = 3.6*(km_per_h) + 1.0*(not km_per_h)
            u = [[float(value) for value in np.gradient(_,self.t[i])*multiplier] for i,_ in enumerate(distance_travelled)]
            smoothing_factor=2
            u_smooth = [gaussian_filter(vec,sigma=smoothing_factor).tolist() for vec in u]
            return u_smooth

        def get_acceleration(self) -> list:
            """
            description
            -----------
            function that calculates acceleration in m/s2 per vehicle per time step of each vehicle, based on speed

            output
            ------
            list of lists; each nested list corresponds to a different vehicle, and includes the information written in the description
            """
            u = self.get_speed(km_per_h=False)
            multiplier  = 1000/3600
            acc = [[float(value)*multiplier for value in np.gradient(_,self.t[i])] for i,_ in enumerate(u)]
            smoothing_factor=2
            a_smooth = [gaussian_filter(vec,sigma=smoothing_factor).tolist() for vec in acc]
            return a_smooth

        def get_od_pairs(self) -> list:
            """
            description
            -----------
            function that calculates origin-destination pairs for each vehicle based on their entry and exit points in the intersection

            output
            ------
            list of tuples; each tuple corresponds to a different vehicle, and includes the information written in the description
            """
            def is_inside_triangle(a,b,c,p):
                """
                description
                -----------
                function that calculates if a point p is inside a triangle defined by vertices a,b,c

                arguments
                ---------
                1) a,b,c are the triangle vertices
                2) p is the point of interest

                output
                ------
                boolean; True if the point is on or inside the triangle, otherwise False
                """
                triangle = Polygon([a,b,c])
                point = Point(p)
                return triangle.contains(point) or triangle.touches(point)

            ll_y,ll_x=self.bbox[0]
            lr_y,lr_x=self.bbox[1]
            ur_y,ur_x=self.bbox[2]
            ul_y,ul_x=self.bbox[3]

            triangle_1 = [(ll_x,ll_y),(self.x_center,self.y_center),(lr_x,lr_y)]
            triangle_2 = [(ul_x,ul_y),(self.x_center,self.y_center),(ll_x,ll_y)]
            triangle_3 = [(ur_x,ur_y),(self.x_center,self.y_center),(ul_x,ul_y)]
            triangle_4 = [(lr_x,lr_y),(self.x_center,self.y_center),(ur_x,ur_y)]

            od_pairs=[]

            for v,_ in enumerate(self.x):
                origin=0
                destination=0
                o = (self.x[v][0],self.y[v][0])
                d = (self.x[v][-1],self.y[v][-1])
                for t,triangle in enumerate([triangle_1,triangle_2,triangle_3,triangle_4]):
                    if is_inside_triangle(*triangle,o) is True:
                        origin=t+1
                        break
                for t,triangle in enumerate([triangle_1,triangle_2,triangle_3,triangle_4]):
                    if is_inside_triangle(*triangle,d) is True:
                        destination=t+1
                        break
                od_pairs.append((origin,destination))

            return od_pairs

        def get_triangle(self) -> list:
            """
            description
            -----------
            function that calculates the virtual triange in which a vehicle belongs, per vehicle, per time step of each vehicle

            output
            ------
            list of lists; each nested list corresponds to a different vehicle, and includes the information written in the description
            """
            def is_inside_triangle(a,b,c,p):
                """check previous function"""
                triangle = Polygon([a,b,c])
                point = Point(p)
                return triangle.contains(point) or triangle.touches(point)

            ll_y,ll_x=self.bbox[0]
            lr_y,lr_x=self.bbox[1]
            ur_y,ur_x=self.bbox[2]
            ul_y,ul_x=self.bbox[3]

            triangle_1 = [(ll_x,ll_y),(self.x_center,self.y_center),(lr_x,lr_y)]
            triangle_2 = [(ul_x,ul_y),(self.x_center,self.y_center),(ll_x,ll_y)]
            triangle_3 = [(ur_x,ur_y),(self.x_center,self.y_center),(ul_x,ul_y)]
            triangle_4 = [(lr_x,lr_y),(self.x_center,self.y_center),(ur_x,ur_y)]

            in_triangle=[]

            for v,vec in enumerate(self.x):
                temp_in_triangle=[]
                for e,_ in enumerate(vec):
                    pos = (self.x[v][e],self.y[v][e])
                    for t,triangle in enumerate([triangle_1,triangle_2,triangle_3,triangle_4]):
                        if is_inside_triangle(*triangle,pos):
                            temp_in_triangle.append(t+1)
                            break
                in_triangle.append(temp_in_triangle)

            return in_triangle

        def get_od_data(self, desirable_pairs:list) -> list:
            """
            description
            -----------
            function that makes a data dictionary only for vehicles that have a certain desirable od pair

            arguments
            ---------
            1) list of tuples; each tuple is a desired od pair

            output
            ------
            dictionary with the information written in the description
            """
            id_od,vtype_od,x_od,y_od,t_od,u_od,=[],[],[],[],[],[]
            od_pairs = self.get_od_pairs()

            for p,pair in enumerate(od_pairs):
                if pair in desirable_pairs:
                    id_od.append(self.vehicle_id[p])
                    vtype_od.append(self.vehicle_type[p])
                    x_od.append(self.x[p])
                    y_od.append(self.y[p])
                    t_od.append(self.t[p])
                    u_od.append(self.u[p])

            od_data = {'id':id_od,'vtype':vtype_od,'x':x_od,'y':y_od,'time':t_od,'speed':u_od,}
            return od_data

        def get_d_from_road_edge(self, flow_direction:str) -> list:
            """
            description
            -----------
            function that calculates distance from bbox edge per vehicle per time step of each vehicle
            in meters; the bbox edge is selected to be parallel to the direction of the vehicles.

            arguments
            ---------
            1) string denoting the direction of flow in the street of interest

            output
            ------
            list of lists; each nested list corresponds to a different vehicle, and includes the information written in the description
            """
            if flow_direction not in ['up','down','right','left']:
                raise ValueError(f'Invalid flow direction, must be one of {['up','down','left','right']}')

            y1,x1 = self.bbox[0]
            y2,x2 = self.bbox[1]*(flow_direction in ['right','left']) + self.bbox[3]*(flow_direction in ['up','down'])
            b_x = self.master.distances(initial_coordinates=(y1,x1),final_coordinates=(y2,x2)).get_dx()
            b_y = self.master.distances(initial_coordinates=(y1,x1),final_coordinates=(y2,x2)).get_dy()
            b = pow((pow(b_x,2)+pow(b_y,2)),0.5)

            d_from_bbox_edge=[]

            for i,vec in enumerate(self.y):
                temp_dfromedge=[]
                for j,_ in enumerate(vec):
                    a_x = self.master.distances(initial_coordinates=(y1,x1),final_coordinates=(self.y[i][j],self.x[i][j])).get_dx()
                    a_y = self.master.distances(initial_coordinates=(y1,x1),final_coordinates=(self.y[i][j],self.x[i][j])).get_dy()
                    temp_dfromedge.append(abs(a_x*b_y - a_y*b_x)/b)
                d_from_bbox_edge.append(temp_dfromedge)

            p=np.percentile([i for j in d_from_bbox_edge for i in j],1)
            effective_min = min(min(_) for _ in d_from_bbox_edge if min(_)>p)
            d_from_road_edge = [(np.array(_)-effective_min).tolist() for _ in d_from_bbox_edge]

            self.d_from_road_edge = d_from_road_edge
            return d_from_road_edge

        def get_lane_info(self, flow_direction:str, valid_od_pairs:list, nbins=100, custom_boundaries=False) -> dict:
            """
            description
            -----------
            function that calculates lane-wise information for a certain street, such as the number of lanes,
            the physical boundaries of each lane (in meters), and the lane in which the vehicles belong to, per vehicle,
            per time step of each vehicle

            arguments
            ---------
            1) string denoting the direction of flow in the street of interest
            2) (optional) number of bins for histograms
            3) (optional) list of tuples; each tuple is an od pair; make the histograms using info only from valid od pairs
            4) (optional) bool; plot the average d_from_bbox_edge per vehicle if True, plot the entire list per vehicle otherwise
            5) (optional) bool; user inserts their own lane boundaries if True, clustering algorithm selected lane boundaries otherwise

            output
            ------
            dictionary with the information written in the description
            """
            self.flow_direction = flow_direction
            d_from_edge = self.get_d_from_road_edge(flow_direction)
            od_pairs = self.get_od_pairs()
            avg_d_from_edge  = [np.mean(_) for i,_ in enumerate(d_from_edge) if od_pairs[i] in valid_od_pairs]

            plt.figure(figsize=(10,3))
            plt.hist(avg_d_from_edge, color='blue', bins=nbins)
            plt.xlabel('Distance from  road edge (m)')
            plt.ylabel('Vehicle Count')
            plt.xticks(np.arange(-1, int(max(avg_d_from_edge))+1,1))
            plt.xlim(-1, int(max(avg_d_from_edge)+1))
            plt.title("Distribution of vehicles in the street")
            plt.tight_layout()
            plt.show()
            plt.close()

            n_clusters = int(input('How many lanes?'))
            low_lim = float(input('low limit?'))
            high_lim = float(input('high limit?'))
            bounded_d_from_edge = [element for element in avg_d_from_edge if low_lim<element<high_lim]

            if custom_boundaries:
                boundaries=[]
                for j in range(1,n_clusters):
                    if j!=n_clusters:
                        tempboundary = float(input(f'Boundary between lanes {j} and {j+1}'))
                    else:
                        tempboundary = float(input('Last boundary'))
                    boundaries.append(tempboundary)
                lane_boundaries = [low_lim]+boundaries+[high_lim]
            else:
                data_for_clustering = np.sort(np.array(bounded_d_from_edge))
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_for_clustering.reshape(-1, 1))
                centers = np.sort(kmeans.cluster_centers_.flatten())
                boundaries_ = (centers[:-1] + centers[1:]) / 2
                boundaries =[low_lim]+boundaries_.tolist()+[high_lim]
                lane_boundaries = [round(value,ndigits=2) for value in boundaries]

            plt.figure(figsize=(10,3))
            counts,bin_edges,_patches=plt.hist(bounded_d_from_edge,bins=int(nbins/2),color='red')
            max_idx = np.argmax(counts)
            peak_of_highest_bin = bin_edges[max_idx]
            for boundary in lane_boundaries:
                plt.axvline(boundary, color='black', linestyle='--',linewidth=2)
            plt.title("Distribution of vehicles in the street")
            plt.xlabel('Distance from road edge (m)')
            plt.ylabel('Vehicle Count')
            plt.ylim(peak_of_highest_bin*2)
            plt.xlim(-1,max(lane_boundaries)+min(bounded_d_from_edge))
            plt.xticks(np.arange(-5,max(lane_boundaries)+5,1))
            plt.tight_layout()
            plt.show()
            plt.close()

            lane_number = len(lane_boundaries)-1
            lane_distribution=[]

            for i,vec in enumerate(d_from_edge):
                temp_lanes=[]
                for j,element in enumerate(vec):
                    if element>max(lane_boundaries) or element<min(lane_boundaries):
                        temp_lanes.append(None)
                    else:
                        temp_lanes.append(max(b for b,boundary in enumerate(lane_boundaries) if element>=boundary))
                lane_distribution.append(temp_lanes)

            lane_info = {
                        'number':lane_number,
                        'boundaries':lane_boundaries,
                        'distribution':lane_distribution
                        }

            self.lane_info = lane_info
            return lane_info

        def get_flow_info(self, detector_position:tuple) -> list:
            """
            description
            -----------
            function that measures counts per time step of the time axis and which vehicles (ids) passed from a virtual detector

            arguments
            ---------
            1) the position coordinates (y,x) of the detector

            output
            ------
            dictionary with the information written in the description
            """
            qoi = self.x*(self.flow_direction in ['left','right']) + self.y*(self.flow_direction in ['up','down'])
            detector_index = 1*(qoi == self.x) + 0*(qoi == self.y)
            before_detector = 'lower'*(self.flow_direction in ['up','right']) + 'higher'*(self.flow_direction in ['left','down'])
            counted_ids=[]
            flow_info=[]

            for m,moment in enumerate(self.time_axis):
                if m==0:
                    tempdict={'time stamp':moment, 'flow':0, 'id':[]}
                    flow_info.append(tempdict)
                    continue
                tempdict={}
                tempid=[]
                for v,_ in enumerate(self.x):

                    if self.vehicle_id[v] in counted_ids:
                        continue
                    if self.time_axis[m] in self.t[v] and self.time_axis[m-1] in self.t[v]:
                        qoi_now = qoi[v][self.t[v].index(self.time_axis[m])]
                        qoi_before =  qoi[v][self.t[v].index(self.time_axis[m-1])]
                        if before_detector=='lower':
                            if qoi_before<= detector_position[detector_index] <= qoi_now:
                                tempid.append(self.vehicle_id[v])
                                counted_ids.append(self.vehicle_id[v])
                        elif before_detector=='higher':
                            if qoi_now<= detector_position[detector_index] <= qoi_before:
                                tempid.append(self.vehicle_id[v])
                                counted_ids.append(self.vehicle_id[v])

                tempdict['time stamp'] = moment
                tempdict['flow'] = len(tempid)
                tempdict['id'] = tempid
                flow_info.append(tempdict)

            self.flow_info = flow_info
            self.detector_positions = detector_position
            return flow_info

        def get_normalized_flow(self, threshold:int) -> tuple[list,list]:
            """
            description
            -----------
            function that groups the counts from the virtual detectors if their distance in time is
            below a threshold; this will later help with traffic light phases

            arguments
            ---------
            1) the groupping threshold written in the description, in seconds

            output
            ------
            1) list with real counts per step of the time axis
            2) list with grouped and normalized (1 if count>0, 0 otherwise) counts per step of the time axis

            notes
            -----
            user needs to have already ran the get_flow_info() function in order for this one to work properly
            """
            flow_info = self.flow_info
            flow=[_.get('flow') for i,_ in enumerate(flow_info)]
            pre_normalized_flow = [1 if value>0 else 0 for value in flow]
            detector_signal = np.array(pre_normalized_flow)
            max_gap = int(threshold/(self.time_axis[1]-self.time_axis[0]))
            on_indices = np.where(detector_signal == 1)[0]
            grouped_signal = np.zeros_like(detector_signal)

            if len(on_indices) > 0:
                start = on_indices[0]
                for i in range(1, len(on_indices)):
                    if on_indices[i] - on_indices[i - 1] > max_gap:
                        grouped_signal[start:on_indices[i - 1] + 1] = 1
                        start = on_indices[i]
                grouped_signal[start:on_indices[-1] + 1] = 1

            normalized_flow = grouped_signal.tolist()
            self.normalized_flow = normalized_flow
            return flow, normalized_flow

        def get_cursed_id(self, low_lim:float, high_lim:float) -> None:
            """
            description
            -----------
            function that prints the ids of vehicles that passed from the detector within a time range

            arguments
            ---------
            1) low limit of time range in seconds
            2) high limit of time range in seconds
            """
            sampling_period = round(self.time_axis[1]-self.time_axis[0])
            for _ in self.flow_info:
                if _.get('time stamp') in  np.arange(low_lim,high_lim+sampling_period,sampling_period):
                    if len(_.get('id'))==0:
                        continue
                    print(f"ids {_.get('id')} passed at time {_.get('time stamp')}")

        def get_traffic_light_phases(self):
            """
            description
            -----------
            function that calculates the traffic light phases for a certain flow direction;
            this info includes: time of green,red lights and their durations;
            entire phase duration (green + red) in seconds

            output
            ------
            list of dictionaries; each nested dictionary corresponds to a different traffic light cycle and has the phase
            information written in the description

            notes
            -----
            user needs to have already ran the get_normalized_flow() function in order for this one to work properly
            """
            normalized_flow = self.normalized_flow
            trafficlights = []
            green,red=0,0

            for v,value in enumerate(normalized_flow):
                tempdic={}
                if value==1 and normalized_flow[v-1]==0:
                    green=self.time_axis[v]
                elif value==0 and normalized_flow[v-1]==1:
                    red=self.time_axis[v]
                if red>0 and green>0:
                    tempdic['Green']=green
                    tempdic['Duration ON']= round(red-green,ndigits=1)
                    tempdic['Red']=red
                    trafficlights.append(tempdic)
                    green,red=0,0
                else:
                    continue

            trafficlightphases=[]
            for d,_dict in enumerate(trafficlights):
                if d+1<len(trafficlights):
                    _dict['Duration OFF'] = round(trafficlights[d+1].get('Green') - trafficlights[d].get('Red'),ndigits=1)
                    _dict['Phase Duration'] = round(trafficlights[d].get('Duration ON') + trafficlights[d].get('Duration OFF'),ndigits=1)
                else:
                    try:
                        _dict['Duration OFF'] = round(trafficlights[d+1].get('Green') - trafficlights[d].get('Red'),ndigits=1)
                        _dict['Phase Duration'] = round(trafficlights[d].get('Duration ON') + trafficlights[d].get('Duration OFF'),ndigits=1)
                    except (IndexError, KeyError):
                        _dict['Duration OFF'] = None
                        _dict['Phase Duration'] = None
                trafficlightphases.append(_dict)

            self.traffic_light_phases = trafficlightphases
            return trafficlightphases

        def get_traffic_light_cycles(self, *traffic_lights_phases:dict) -> dict:
            """
            description
            -----------
            function that calculates the traffic light cycles for two competing traffic phases,
            this info includes: time of beginning,ending of the individual phases and of the cycle in seconds;

            arguments
            ---------
            1) the dictionaries for the two traffic light phases

            output
            ------
            list of directories; each nested dictionary corresponds to a different traffic light cycle, and includes the information written in the description
            """
            traffic_lights_one,traffic_lights_two=traffic_lights_phases

            cycles=[]

            for p,_ in enumerate(traffic_lights_one):
                temp={}
                this_phase_one=traffic_lights_one[p]
                this_phase_two=traffic_lights_two[p]
                temp['Start'] = min(this_phase_one.get('Green'),this_phase_two.get('Green'))
                temp['Break'] = min(this_phase_one.get('Red'),this_phase_two.get('Red'))
                temp['Continue']  = max(this_phase_one.get('Green'),this_phase_two.get('Green'))
                temp['Stop'] = max(this_phase_one.get('Red'),this_phase_two.get('Red'))
                if p<len(traffic_lights_one)-1:
                    next_phase_one = traffic_lights_one[p+1]
                    next_phase_two = traffic_lights_two[p+1]
                    temp['End'] =   min(next_phase_one.get('Green'),next_phase_two.get('Green'))
                else:
                    temp['End'] = temp['Stop']
                cycles.append(temp)

            self.cycles = cycles
            return cycles

        def get_sorted_id(self):
            """
            description
            -----------
            function that sorts vehicle ids according to their lane and direction of motion per step of the time axis

            output
            ------
            list of dictionaries; each nested dictionary includes the information written in the description

            notes
            -----
            the user needs to have already ran the get_lane_info() function in order for this one to work properly
            """
            lane_number,lane_distriubtion = itemgetter('number','distribution')(self.lane_info)

            if self.flow_direction not in ['up','down','left','right']:
                raise ValueError(f'Invalid flow direction, must be one of {['up','down','left','right']}')

            qoi = self.y*(self.flow_direction in ['up','down']) + self.x*(self.flow_direction in ['left','right'])
            sorted_id=[]

            for moment in self.time_axis:
                temp_dict = {'time stamp':moment }
                for l in range(lane_number):
                    temp_lane_ids=[]
                    temp_lane_qoi=[]
                    for v,_ in enumerate(qoi):
                        if moment in self.t[v] and lane_distriubtion[v][self.t[v].index(moment)]==l:
                            temp_lane_ids.append(self.vehicle_id[v])
                            temp_lane_qoi.append(qoi[v][self.t[v].index(moment)])
                    if len(temp_lane_ids)==0:
                        temp_sorted_id=None
                    else:
                        temp_sorted_id = [temp_lane_ids for _, temp_lane_ids in sorted(zip(temp_lane_qoi, temp_lane_ids),reverse=(self.flow_direction in ['left','down']))]
                    temp_dict[f'lane {l}']= temp_sorted_id
                sorted_id.append(temp_dict)

            self.sorted_id = sorted_id
            return sorted_id

        def get_gaps(self) -> list:
            """
            description
            -----------
            function that calculates gaps in meters (front to rear bumper) between  sorted vehicles according to their lane and direction of motion per step of the time axis

            output
            ------
            list of dictionaries; each nested dictionary includes the information written in the description

            notes
            -----
            the user needs to have already ran the get_sorted_id() function in order for this one to work properly
            """
            sorted_id = self.sorted_id

            gaps=[]

            for _dict in sorted_id:
                moment = _dict.get('time stamp')
                temp_dict = {'time stamp': moment}
                for k,key in enumerate(_dict): #each key for k>0 corresponds to the different lanes
                    if k>0: #skip the time stamp
                        sorted_id_in_this_lane = _dict.get(key)
                        if sorted_id_in_this_lane is None:
                            temp_dict[f'lane {k-1}']=None
                        elif len(sorted_id_in_this_lane)==1:
                            temp_dict[f'lane {k-1}']=[-1.0]
                        else:
                            refx = [self.x[self.vehicle_id.index(value)][self.t[self.vehicle_id.index(value)].index(moment)] for v,value in enumerate(sorted_id_in_this_lane)]
                            refy = [self.y[self.vehicle_id.index(value)][self.t[self.vehicle_id.index(value)].index(moment)] for v,value in enumerate(sorted_id_in_this_lane)]
                            nxtx = [self.x[self.vehicle_id.index(sorted_id_in_this_lane[v+1])][self.t[self.vehicle_id.index(sorted_id_in_this_lane[v+1])].index(moment)] for v,value in enumerate(sorted_id_in_this_lane[:-1])]
                            nxty = [self.y[self.vehicle_id.index(sorted_id_in_this_lane[v+1])][self.t[self.vehicle_id.index(sorted_id_in_this_lane[v+1])].index(moment)] for v,value in enumerate(sorted_id_in_this_lane[:-1])]
                            vtypes = [self.vehicle_type[self.vehicle_id.index(value)] for v,value in enumerate(sorted_id_in_this_lane)]
                            vlengths = [5*(vt in ('Car','Taxi')) + 5.83*(vt=='Medium')+12.5*(vt in ('Heavy','Bus'))+2.5*(vt=='Motorcycle') for vt in vtypes]
                            intralane_gaps = [self.master.distances(initial_coordinates=(refy[i],refx[i]),final_coordinates=(nxty[i],_)).get_distance() - 0.5*(vlengths[i]+vlengths[i+1]) for i,_ in enumerate(nxtx)]
                            final_gaps = [round(value,ndigits=1) if value>0 else 1 for value in intralane_gaps]
                            if len(final_gaps)>0:
                                final_gaps.append(-1.0) #for the leader
                            temp_dict[f'lane {k-1}']=final_gaps
                gaps.append(temp_dict)

            self.gaps = gaps
            return gaps

        def get_queue_info(self, speed_threshold:float, gap_threshold:float) -> list:
            """
            description
            -----------
            function that calculates queue-wise information per traffic light phase
            per lane; this info includes: number of queued vehicles, queue length in meters,
            queued vehicle ids, queue dissipation time in seconds

            arguments
            ---------
            1) speed threshold for vehicle to be considered in queue
            2) gap threshold for vehicle to be considered in queue

            output
            ------
            list of lists; each nested list corresponds to a different traffic light cycle
            and has nested dictionaries; each nested dictionary corresponds to a different lane
            and includes the information written in the description

            notes
            -----
            the user needs to have already ran the get_lane_info(), get_gaps(), get_sorted_id() function in order for this one to work properly
            """
            qoi = self.x*(self.flow_direction in ['left','right']) + self.y*(self.flow_direction in ['up','down'])
            detector_index = 1*(qoi==self.x) + 0*(qoi==self.y)
            before_detector = 'lower'*(self.flow_direction in ['up','right']) + 'higher'*(self.flow_direction in ['down','left'])
            queue_info=[]

            for phase in self.traffic_light_phases:

                phase_info=[]
                offset_digits=0
                green = phase.get('Green')-offset_digits
                red = phase.get('Red')

                for l in range(self.lane_info.get('number')):

                    temp_info={'Lane':l}
                    id_at_green = self.sorted_id[self.time_axis.index(green)].get(f'lane {l}')
                    if id_at_green is None:
                        temp_info['Queued Vehicles']=None
                        temp_info['Queue Length']=None
                        temp_info['Queued IDs']=None
                        temp_info['Dissipation Duration']=None
                        phase_info.append(temp_info)
                        continue

                    id_before_trafficlight=[]
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    for _i,identity in enumerate(id_at_green):
                        vehicle_index = self.vehicle_id.index(identity)
                        green_index  = self.t[self.vehicle_id.index(identity)].index(green)
                        if self.u[vehicle_index][green_index]>=speed_threshold:
                            continue
                        #---------------------------
                        if before_detector=='lower':
                            if qoi[vehicle_index][green_index]<=self.detector_positions[detector_index]:
                                id_before_trafficlight.append(identity)
                        #---------------------------
                        elif before_detector=='higher':
                            #-----------------------------------------------------------------------------
                            if qoi[vehicle_index][green_index]>=self.detector_positions[detector_index]:
                                id_before_trafficlight.append(identity)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if len(id_before_trafficlight)==0:
                        temp_info['Queued Vehicles']=None
                        temp_info['Queue Length']=None
                        temp_info['Queued IDs']=None
                        temp_info['Dissipation Duration']=None
                        phase_info.append(temp_info)
                        continue

                    id_gaps_before_trafficlight = [(_identity,self.gaps[self.time_axis.index(green)].get(f'lane {l}')[j]) for j,_identity in enumerate(id_at_green) if _identity in id_before_trafficlight]
                    revised_id_before_trafficlight = [_identity for _identity,gap in id_gaps_before_trafficlight if gap<=gap_threshold]
                    revised_gaps_before_trafficlight = [gap for _identity,gap in id_gaps_before_trafficlight if gap<=gap_threshold]
                    types_before_trafficlight = [self.vehicle_type[self.vehicle_id.index(_identity)] for _j,_identity in enumerate(revised_id_before_trafficlight)]
                    lengths_before_trafficlight = [5*(vt in ('Car','Taxi'))+ 5.83*(vt in ('Medium')) + 12.5*(vt in ('Heavy','Bus')) + 2.5*(vt in ('Motorcycle')) for vt in types_before_trafficlight]

                    if len(revised_id_before_trafficlight)>0:
                        temp_info['Queued Vehicles'] = len(revised_id_before_trafficlight)
                        temp_info['Queue Length'] = round(sum(revised_gaps_before_trafficlight[:-1])+sum(lengths_before_trafficlight),ndigits=0)
                        temp_info['Queued IDs'] = revised_id_before_trafficlight

                        flag=False
                        for moment in self.time_axis[self.time_axis.index(green):]:
                            last_vehicle = id_before_trafficlight[0] #leading vehicle is last in list, last vehicle always first in list
                            if moment in self.t[self.vehicle_id.index(last_vehicle)]:

                                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                if before_detector=='lower':
                                    if qoi[self.vehicle_id.index(last_vehicle)][self.t[self.vehicle_id.index(last_vehicle)].index(moment)]>=self.detector_positions[detector_index]:
                                        temp_info['Dissipation Duration'] = round(moment-green+offset_digits,ndigits=1)
                                        flag=True
                                        break
                                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                elif before_detector=='higher':
                                    if qoi[self.vehicle_id.index(last_vehicle)][self.t[self.vehicle_id.index(last_vehicle)].index(moment)]<=self.detector_positions[detector_index]:
                                        temp_info['Dissipation Duration'] = round(moment-green+offset_digits,ndigits=1)
                                        flag=True
                                        break
                                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                        if flag is False:
                            temp_info['Dissipation Time'] = round(red-green+offset_digits,ndigits=1)
                    else:
                        temp_info['Queued Vehicles'] = None
                        temp_info['Queued IDs'] = None
                        temp_info['Queue Length'] = None
                        temp_info['Dissipation Duration'] = None

                    phase_info.append(temp_info)
                queue_info.append(phase_info)
            return queue_info

    class _visualization:
        """
        description
        -----------
        this class is dedicated to carrying out visualization tasks
        """

        def __init__(self, master:'Master', data:dict, spatio_temporal_info:dict):
            """
            description
            -----------
            this class is dedicated to carrying out analysis tasks

            arguments
            ---------
            1) master class
            2) trimmed/filtered data dictionary
            3) spatiotemporal info dictionary
            """
            self.master = master
            needed_keys_data = ['id','vtype','x','y','time','speed']
            needed_keys_info = ['bbox','intersection center','time axis']

            if any(key not in data.keys() for key in needed_keys_data):
                raise KeyError(f'data dictionary needs keys {needed_keys_data} to work!')
            if any(key not in spatio_temporal_info.keys() for key in needed_keys_info):
                raise KeyError(f'spatio temporal info dictionary needs keys {needed_keys_info} to work!')

            self.data = data
            self.spatio_temporal_info = spatio_temporal_info
            self.vehicle_id,self.vehicle_type,self.x, self.y, self.t, self.u = itemgetter('id','vtype','x','y','time','speed')(data)
            self.bbox,self.center,self.time_axis = itemgetter('bbox','intersection center','time axis')(spatio_temporal_info)
            self.y_center,self.x_center = self.center
            self.ll_y,self.ll_x=self.bbox[0]
            self.lr_y,self.lr_x=self.bbox[1]
            self.ur_y,self.ur_x=self.bbox[2]
            self.ul_y,self.ul_x=self.bbox[3]
            self.aspect_ratio = self.master.distances((self.lr_y,self.lr_x),(self.ur_y,self.ur_x)).get_dy() / self.master.distances((self.ll_y,self.ll_x),(self.lr_y,self.lr_x)).get_dx()

        def draw_trajectories(self,title=True,scale=8,dpi=100) -> None:
            """
            description
            -----------
            this function draws the trajectories of all vehicles in the data
            dictionary with a speed heatmap

            arguments
            ---------
            1) (optional) display the title of the plot or not
            2) (optional) the scale of the plot
            3) (optional) dpi (higher dpi -> higher quality)

            output
            ------
            plot of the information written in the description 
            """
            vmin = int(min(np.mean(set) for set in self.u))
            vmax = int(max(np.mean(set) for set in self.u))

            x_flat = [value for i,vec in enumerate(self.x) for value in vec if self.vehicle_type[i]!='Motorcycle']
            y_flat = [value for i,vec in enumerate(self.y) for value in vec if self.vehicle_type[i]!='Motorcycle']

            u = self.master.analysis(self.data, self.spatio_temporal_info).get_speed()
            u_flat = [value for i,vec in enumerate(u) for value in vec if self.vehicle_type[i]!='Motorcycle']

            xlim_left,xlim_right=min(x_flat),max(x_flat)
            ylim_bottom,ylim_top = min(y_flat),max(y_flat)

            _fig,ax = plt.subplots(figsize=(scale,self.aspect_ratio*scale),dpi=dpi)
            if title:
                ax.set_title('Trajectories')
            ax.set_facecolor('black')
            scatter = ax.scatter(x_flat,y_flat,c=u_flat,cmap='gist_rainbow_r',vmin=vmin,vmax=vmax,s=0.05)
            cbar = plt.colorbar(scatter,ax=ax)
            cbar.set_label('Speed (km/h)')
            plt.xlabel('latitude')
            plt.ylabel('logntitude')
            plt.xlim(xlim_left,xlim_right)
            plt.ylim(ylim_bottom,ylim_top)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            plt.tight_layout()
            plt.show()
            plt.close()

        def draw_trajectories_od(self,title=True,scale=6,dpi=100) -> None:
            """
            description
            -----------
            this function makes a visual separation of the components
            of the intersection according to the different entry (o) and exit (d) points.
            The point is for the user to see if the intersection is separated properly or if
            they need to adjust the bbox or/and the intersection center.
            
            arguments
            ---------
            1) (optional) display the title of the plot or not
            2) (optional) the scale of the plot
            3) (optional) dpi (higher dpi -> higher quality)

            output
            ------
            plot of the information written in the description 
            """
            alpha=(1/len(self.data.get('id')))**(1/3)
            horizontal = self.master.distances((self.ll_y,self.ll_x),(self.ur_y,self.ur_x)).get_dy()
            vertical = self.master.distances((self.ul_y,self.ul_x),(self.lr_y,self.lr_x)).get_dx()
            aspect_ratio = horizontal/vertical

            _fig,ax=plt.subplots(figsize=(scale,scale*aspect_ratio),dpi=dpi)
            if title:
                ax.set_title('Route map')

            for l,_ in enumerate(self.x):

                if self.vehicle_type[l]!='Motorcycle':

                    ax.plot(self.x[l],self.y[l],color='k',linewidth=0.5,alpha=alpha)

            ax.plot([element[-1] for element in self.bbox]+[self.bbox[0][-1]],[element[0] for element in self.bbox] + [self.bbox[0][0]],color='blue',linewidth=0.5)
            ax.plot([self.ul_x,self.x_center,self.lr_x],[self.ul_y,self.y_center,self.lr_y],color='blue',linewidth=0.5)
            ax.plot([self.ll_x,self.x_center,self.ur_x],[self.ll_y,self.y_center,self.ur_y],color='blue',linewidth=0.5)
            ax.text(0.5*(self.ll_x+self.lr_x),0.5*(self.ll_y+self.lr_y),'1',fontsize=14,fontweight='bold',color='red')
            ax.text(0.5*(self.ll_x+self.ul_x),0.5*(self.ll_y+self.ul_y),'2',fontsize=14,fontweight='bold',color='red')
            ax.text(0.5*(self.ul_x+self.ur_x),0.5*(self.ul_y+self.ur_y),'3',fontsize=14,fontweight='bold',color='red')
            ax.text(0.5*(self.ur_x+self.lr_x),0.5*(self.ur_y+self.lr_y),'4',fontsize=14,fontweight='bold',color='red')
            ax.set_xlim(min(self.ll_x,self.ul_x),max(self.ur_x,self.lr_x))
            ax.set_ylim(min(self.ll_y,self.lr_y),max(self.ul_y,self.ur_y))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('longitude')
            ax.set_ylabel('latitude')
            plt.tight_layout()
            plt.show()
            plt.close()

        def draw_trajectories_cc(self, valid_od_pairs:list, title=True,scale=6,dpi=100):
            """
            description
            -----------
            this function draws the trajectories of all vehicles in the data
            dictionary color-coded according to their entry (o) and
            exit (o) point in the intersection.

            arguments
            ---------
            1) list of tuples with the correct od pairs based on the intersection
            2) (optional) display the title of the plot or not
            3) (optional) the scale of the plot
            4) (optional) dpi (higher dpi -> higher quality)

            output
            ------
            plot of the information written in the description 
            """
            od_pairs = self.master.analysis(self.data, self.spatio_temporal_info).get_od_pairs()
            colors= ['blue','orange','red','green','cyan','violet','gray','brown']
            alphas = [(1 - (sum(1 for _ in od_pairs if _==pair))/(len(od_pairs))) for pair in valid_od_pairs]

            horizontal = self.master.distances((self.ll_y,self.ll_x),(self.ur_y,self.ur_x)).get_dy()
            vertical = self.master.distances((self.ul_y,self.ul_x),(self.lr_y,self.lr_x)).get_dx()
            aspect_ratio = horizontal/vertical

            _fig,ax = plt.subplots(figsize=(scale,scale*aspect_ratio),dpi=dpi)
            if title:
                ax.set_title('Color-Coded Trajectories')
            ax.set_facecolor('k')
            for l,_ in enumerate(self.x):

                if od_pairs[l] in valid_od_pairs and self.vehicle_type[l]!='Motorcycle':

                    ax.plot(self.x[l],self.y[l],color=colors[valid_od_pairs.index(od_pairs[l])],linewidth=1,alpha=alphas[valid_od_pairs.index(od_pairs[l])])
            ax.plot([element[-1] for element in self.bbox]+[self.bbox[0][-1]],[element[0] for element in self.bbox] + [self.bbox[0][0]],color='white')

            ax.set_xlim(min(self.ll_x,self.ul_x),max(self.ur_x,self.lr_x))
            ax.set_ylim(min(self.ll_y,self.lr_y),max(self.ul_y,self.ur_y))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('longitude')
            ax.set_ylabel('latitude')
            plt.tight_layout()
            plt.show()
            plt.close()

        def draw_spacetime_diagram(self) -> None:
            """
            description
            -----------
            this function draws a spacetime diagram of all the vehicles in the data dictionary

            output
            ------
            plot of the information written in the description 
            """
            vmin = int(min(np.mean(set) for set in self.u))
            vmax = int(max(np.mean(set) for set in self.u))

            x = [value for set in self.t for value in set]
            y = [value for set in self.master.analysis(self.data,self.spatio_temporal_info).get_distance_travelled() for value in set]
            z = [value for set in self.u for value in set]

            _fig,ax= plt.subplots(figsize=(18,5))

            scatter = ax.scatter(x,y,c=z,cmap='jet_r',vmin=vmin,vmax=vmax,s=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Speed (km/h)')

            plt.title('Spacetime Diagram')
            plt.xlabel('t (s)')
            plt.ylabel('Distance Travelled (m)')
            plt.tight_layout()
            plt.show()
            plt.close()

        def draw_distance_travelled(self, vehicle_id:int):
            """
            description
            -----------
            this function draws the cumulative distance travelled by a single
            vehicle as a function of time

            arguments
            ---------
            1) the id of the vehicle the user wants to visualize

            output
            ------
            plot of the information written in the description 
            """
            _fig,ax = plt.subplots(figsize=(6,3))

            ax.plot(self.t[self.vehicle_id.index(vehicle_id)],self.master.analysis(self.data, self.spatio_temporal_info).get_distance_travelled()[self.vehicle_id.index(vehicle_id)],color='k',label=f'Vehicle ID: {vehicle_id}')

            plt.title('Distance Travalled vs Time')
            plt.xlabel('t (s)')
            plt.ylabel('Distance Travelled (m)')
            plt.tight_layout()
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.close()

        def draw_speed(self, vehicle_id:int):
            """
            description
            -----------
            this function draws the speed of a single vehicle as a function of time

            arguments
            ---------
            1) the id of the vehicle the user wants to visualize

            output
            ------
            plot of the information written in the description 
            """
            _fig,ax = plt.subplots(figsize=(6,3))

            ax.plot(self.t[self.vehicle_id.index(vehicle_id)],self.u[self.vehicle_id.index(vehicle_id)],color='blue',label=f'Vehicle ID: {vehicle_id}')

            plt.title('Speed vs Time')
            plt.xlabel('t (s)')
            plt.ylabel('Speed (km/h)')
            plt.tight_layout()
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.close()

        def draw_acceleration(self, vehicle_id:int):
            """
            description
            -----------
            this function draws the acceleration of a single
            vehicle as a function of time

            arguments
            ---------
            1) the id of the vehicle the user wants to visualize

            output
            ------
            plot of the information written in the description 
            """
            a = self.master.analysis(self.data, self.spatio_temporal_info).get_acceleration()

            _fig,ax = plt.subplots(figsize=(6,3))

            ax.plot(self.t[self.vehicle_id.index(vehicle_id)],a[self.vehicle_id.index(vehicle_id)],color='red',label=f'Vehicle ID: {vehicle_id}')

            plt.title('Acceleration vs Time')
            plt.xlabel('t (s)')
            plt.ylabel(r'Acceleration $(m/s^2)$')
            plt.tight_layout()
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.close()
        
        def draw_traffic_light_phases(self, legend_1:str, legend_2:str, norm_flow_1:list, norm_flow_2:list, flow_1:list, flow_2:list, plt_norm_flow_1=True, plt_norm_flow_2=True, plt_flow_1=True, plt_flow_2=True, activate_zoom=False, low_lim=0,high_lim=1e2, title=True, dpi=100):
            """
            description
            -----------
            this function draws the normalized/unnormalized flows of the intersection to
            visualize the traffic light cycles

            arguments
            ---------
            1) name of origins-destinations for the first flow
            2) name of origins-destinations for the second flow
            3) the normalized first flow
            4) the normalized second flow
            5) the unnormalized first flow
            6) the unnormalized second flow
            7) (optional) bool; plot the normalized first flow
            8) (optional) bool; plot the normalized second flow
            9) (optional) bool; plot the unnormalized first flow
            10) (optional) bool; plot the unnormalized second flow
            11) (optional) bool; zoom in activate the get_cursed_id() function to spot vehicles that cause problematic counts 
                if above is True:
                    12) lower limit of zoom
                    13) higher limit of zoom
            14) (optional) dpi (higher dpi -> higher quality)
            15) (optional) display the title of the plot or not

            output
            ------
            plot of the information written in the description 
            """
            _fig,ax=plt.subplots(figsize=(18,5),dpi=dpi)
            if title:
                ax.set_title('Traffic Light Phases')

            ax.set_xlabel('Time (s)')
            ax.set_xticks(np.arange(0,self.time_axis[-1],25))

            if plt_norm_flow_1:
                ax.plot(self.time_axis,norm_flow_1,color='k',alpha=1, label=f'{legend_1}',linewidth=3)
            if plt_norm_flow_2:
                ax.plot(self.time_axis,norm_flow_2,color='r',alpha=1,label=f'{legend_2}',linewidth=3)
            if plt_flow_1:
                ax.plot(self.time_axis,flow_1,color='k',alpha=0.3)
            if plt_flow_2:
                ax.plot(self.time_axis,flow_2,color='r',alpha=0.3)

            if plt_flow_1 is False and plt_flow_2 is False:
                ax.set_ylim(-0.1,2)
                ax.set_yticks([0,1],['Detector OFF (0)','Detector ON (1)'])
            else:
                ax.set_ylabel('Counts')

            if activate_zoom:
                ax.set_xlim(low_lim,high_lim)
                ax.set_xticks(np.arange(low_lim,high_lim+1,1))

            ax.axhline(y=0,color='white',linewidth=3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()
