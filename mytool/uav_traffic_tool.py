"""docstring"""

from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from shapely import Point,Polygon
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

class Wiz:
    """docstring"""

    def __init__(self):
        """docstring"""

    def distances(self, initial_coordinates:tuple, final_coordinates:tuple, wgs:bool):
        """docstring"""
        return self._distances(self, initial_coordinates, final_coordinates, wgs)

    def dataloader(self, raw_data:dict, spatio_temporal_info:dict):
        """docstring"""
        return self._dataloader(self, raw_data, spatio_temporal_info)

    def analysis(self, data:dict, spatio_temporal_info:dict):
        """docstring"""
        return self._analysis(self, data, spatio_temporal_info)

    def visualization(self, data:dict, spatio_temporal_info:dict):
        """docstring"""
        return self._visualization(self, data, spatio_temporal_info)

    class _distances:
        """docstring"""

        def __init__(self, mother:'Wiz', initial_coordinates:tuple, final_coordinates:tuple, wgs:bool):
            """docstring"""

            self.mother = mother
            self.y_i,self.x_i = initial_coordinates
            self.y_f,self.x_f = final_coordinates
            self.wgs = wgs
            self.factor = 2*np.pi*6371000/360

        def get_dx(self) -> float:
            """docstring"""
            return (self.factor*float(np.cos(np.deg2rad(self.y_i)))*(self.x_f - self.x_i))*(self.wgs) + (self.x_f - self.x_i)*(not self.wgs)

        def get_dy(self) -> float:
            """docstring"""
            return (self.factor*(self.y_f - self.y_i))*(self.wgs) + (self.y_f - self.y_i)*(not self.wgs)

        def get_distance(self) -> float:
            """docstring"""
            dx = self.get_dx()
            dy = self.get_dy()
            return pow((pow(dx,2) + pow(dy,2)),0.5)

    class _dataloader:
        """docstring"""

        def __init__(self, mother:'Wiz', raw_data:dict, spatio_temporal_info:dict):
            """docstring"""
            self.mother = mother
            needed_keys_data = ['id','vtype','x','y','time','speed']
            needed_keys_info = ['wgs','bbox','intersection center','time axis']
            if any(key not in raw_data.keys() for key in needed_keys_data) or any(key not in spatio_temporal_info.keys() for key in needed_keys_info):
                raise KeyError(f'data directory needs keys {needed_keys_data} to work! \nspatiotemporal info directory needs {needed_keys_info} keys to work!')

            self.raw_data = raw_data
            self.vehicle_id,self.vehicle_type,self.x,self.y,self.t,self.u  = itemgetter('id','vtype','x','y','time','speed')(raw_data)
            self.wgs,self.bbox = itemgetter('wgs','bbox')(spatio_temporal_info)

        def get_intersection_data(self) -> dict:
            """docstring"""
            box = Polygon(self.bbox)
            x_,y_,t_,u_,id_,vtype_=[],[],[],[],[],[]
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
            intersection_data ={'id':id_,'vtype':vtype_,'x':x_,'y':y_,'time':t_,'speed':u_}
            return intersection_data

        def get_filtered_data(self, cursed_ids=None) -> dict:
            """docstring"""
            if cursed_ids is None:
                cursed_ids=[]

            intersection_data  = self.get_intersection_data()
            vehicle_id,vehicle_type,x,y,t,u = itemgetter('id','vtype','x','y','time','speed')(intersection_data)
            id_,vtype_,x_,y_,t_,u_=[],[],[],[],[],[]
            for i,vec in enumerate(x):
                immobility = sum(1 if self.mother.distances(initial_coordinates=(y[i][j-1],x[i][j-1]),final_coordinates=(y[i][j],x[i][j]),wgs=self.wgs).get_distance() <1e-4 else 0 for j,element in enumerate(vec[1:]))
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
        """docstring"""

        def __init__(self, mother:'Wiz', data:dict, spatio_temporal_info:dict):
            """docstring"""

            self.mother = mother
            needed_keys_data = ['id','vtype','x','y','time','speed']
            needed_keys_info = ['wgs','bbox','intersection center','time axis']
            if any(key not in data.keys() for key in needed_keys_data) or any(key not in spatio_temporal_info.keys() for key in needed_keys_info):
                raise KeyError(f'data dictionary needs keys {needed_keys_data} to work! \nspatiotemporal info directory needs {needed_keys_info} keys to work!')

            self.data = data
            self.spatio_temporal_info = spatio_temporal_info
            self.vehicle_id,self.vehicle_type,self.x, self.y, self.t, self.u = itemgetter('id','vtype','x','y','time','speed')(data)
            self.wgs, self.bbox,self.center,self.time_axis = itemgetter('wgs','bbox','intersection center','time axis')(spatio_temporal_info)
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
            """docstring"""

            distance_travelled=[]
            for i,vec in enumerate(self.x):
                temp_distance_travelled=[]
                temp_sum=0
                for j,_ in enumerate(vec):
                    if j==0:
                        temp_distance_travelled.append(0)
                    else:
                        temp_sum+= self.mother.distances(initial_coordinates=(self.y[i][j-1],self.x[i][j-1]),final_coordinates=(self.y[i][j],self.x[i][j]),wgs=self.wgs).get_distance()
                        temp_distance_travelled.append(temp_sum)
                distance_travelled.append(temp_distance_travelled)
            return distance_travelled

        def get_speed(self) -> list:
            """docstring"""

            distance_travelled = self.get_distance_travelled()
            km_per_h = True
            multiplier = 3.6*(km_per_h) + 1.0*(not km_per_h)
            u = [[float(value) for value in np.gradient(_,self.t[i])*multiplier] for i,_ in enumerate(distance_travelled)]
            smoothing_factor=2
            u_smooth = [gaussian_filter(vec,sigma=smoothing_factor).tolist() for vec in u]
            return u_smooth

        def get_acceleration(self) -> list:
            """docstring"""

            u = self.get_speed()
            multiplier  = 1000/3600
            acc = [[float(value)*multiplier for value in np.gradient(_,self.t[i])] for i,_ in enumerate(u)]
            smoothing_factor=2
            a_smooth = [gaussian_filter(vec,sigma=smoothing_factor).tolist() for vec in acc]
            return a_smooth

        def get_od_pairs(self) -> list:
            """docstring"""

            def is_inside_triangle(a,b,c,p):
                """docstring"""
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
            """docstring"""
            def is_inside_triangle(a,b,c,p):
                """docstring"""
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
            """docstring"""

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

        def get_d_from_bbox_edge(self, flow_direction:str) -> list:
            """docstring"""

            if flow_direction not in ['up','down','right','left']:
                raise ValueError(f'Invalid flow direction, must be one of {['up','down','left','right']}')
            y1,x1 = self.bbox[0]
            y2,x2 = self.bbox[1]*(flow_direction in ['right','left']) + self.bbox[3]*(flow_direction in ['up','down'])
            b_x = self.mother.distances(initial_coordinates=(y1,x1),final_coordinates=(y2,x2),wgs=self.wgs).get_dx()
            b_y = self.mother.distances(initial_coordinates=(y1,x1),final_coordinates=(y2,x2),wgs=self.wgs).get_dy()
            b = pow((pow(b_x,2)+pow(b_y,2)),0.5)
            d_from_edge=[]
            for i,vec in enumerate(self.y):
                temp_dfromedge=[]
                for j,_ in enumerate(vec):
                    a_x = self.mother.distances(initial_coordinates=(y1,x1),final_coordinates=(self.y[i][j],self.x[i][j]),wgs=self.wgs).get_dx()
                    a_y = self.mother.distances(initial_coordinates=(y1,x1),final_coordinates=(self.y[i][j],self.x[i][j]),wgs=self.wgs).get_dy()
                    temp_dfromedge.append(abs(a_x*b_y - a_y*b_x)/b)
                d_from_edge.append(temp_dfromedge)
            self.d_from_edge = d_from_edge
            self.flow_direction=flow_direction
            return d_from_edge

        def get_lane_info(self, flow_direction, nbins=200, valid_od_pairs=None, avg_d_from_bbox_edge=False, custom_boundaries=False) -> None:
            """docstring""" 

            d_from_edge = self.get_d_from_bbox_edge(flow_direction)
            if valid_od_pairs is None:
                flat_d_from_edge = [element for vec in d_from_edge for element in vec]*(not avg_d_from_bbox_edge) + [float(np.mean(vec)) for vec in d_from_edge]*avg_d_from_bbox_edge
            else:
                od_pairs=self.get_od_pairs()
                flat_d_from_edge = [element for i,vec in enumerate(d_from_edge) for element in vec if od_pairs[i] in valid_od_pairs]*(not avg_d_from_bbox_edge) + [float(np.mean(vec)) for i,vec in enumerate(d_from_edge) if od_pairs[i] in valid_od_pairs]*(avg_d_from_bbox_edge)
            #--------------------------------------------------------
            plt.figure(figsize=(10,2))
            plt.hist(flat_d_from_edge, color='blue', bins=nbins, density=True)
            plt.xlabel('Distance from bbox edge (m)')
            plt.ylabel('Normalized Occurences')
            plt.xticks(np.arange(np.trunc(np.mean(flat_d_from_edge)-20), np.trunc(np.mean(flat_d_from_edge)+20),2))
            plt.xlim(np.trunc(np.mean(flat_d_from_edge)-20), np.trunc(np.mean(flat_d_from_edge)+20))
            plt.title("Lane Distribution")
            plt.show(close=True)
            #--------------------------------------------------------
            n_clusters = int(input('How many lanes?'))
            low_lim = float(input('low limit?'))
            high_lim = float(input('high limit?'))
            bounded_d_from_edge = [element for element in flat_d_from_edge if low_lim<element<high_lim]
            #--------------------------------------------------------
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
            #--------------------------------------------------------
            plt.figure(figsize=(8.75,3))
            plt.hist(bounded_d_from_edge,bins=int(nbins/2),color='red',density=True)
            for boundary in lane_boundaries:
                plt.axvline(boundary, color='black', linestyle='--',linewidth=2)
            plt.title("Lane Clustering with Boundaries")
            plt.xlabel('Distance from bbox edge (m)')
            plt.ylabel('Probability Density')
            plt.tight_layout()
            plt.show(close=True)
            #--------------------------------------------------------
            lane_number = len(lane_boundaries)-1
            lane_distribution=[]

            for i,vec in enumerate(self.d_from_edge):
                temp_lanes=[]
                for j,element in enumerate(vec):
                    if element>max(lane_boundaries) or element<min(lane_boundaries):
                        temp_lanes.append(None)
                    else:
                        temp_lanes.append(max(b for b,boundary in enumerate(lane_boundaries) if element>boundary))
                lane_distribution.append(temp_lanes)
            lane_info = {
                'number':lane_number,
                'boundaries':lane_boundaries,
                'distribution':lane_distribution
                }
            self.lane_info = lane_info
            return lane_info

        def get_flow_info(self, detector_positions:list) -> list:
            """docstring"""
            
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

                        #qoi = quantity of interest
                        qoi_now = qoi[v][self.t[v].index(self.time_axis[m])]
                        qoi_before =  qoi[v][self.t[v].index(self.time_axis[m-1])]

                        if before_detector=='lower':
                            if qoi_before<= detector_positions[detector_index] <= qoi_now:
                                tempid.append(self.vehicle_id[v])
                                counted_ids.append(self.vehicle_id[v])
                        elif before_detector=='higher':
                            if qoi_now<= detector_positions[detector_index] <= qoi_before:
                                tempid.append(self.vehicle_id[v])
                                counted_ids.append(self.vehicle_id[v])

                tempdict['time stamp'] = moment
                tempdict['flow'] = len(tempid)
                tempdict['id'] = tempid
                flow_info.append(tempdict)

            self.flow_info = flow_info
            self.detector_positions = detector_positions
            return flow_info

        def get_normalized_flow(self, threshold:int) -> tuple[list,list]:
            """docstring"""

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
            """docstring"""

            sampling_period = round(self.time_axis[1]-self.time_axis[0])
            for _ in self.flow_info:
                if _.get('time stamp') in  np.arange(low_lim,high_lim+sampling_period,sampling_period):
                    if len(_.get('id'))==0:
                        continue
                    print(f"ids {_.get('id')} passed at time {_.get('time stamp')}")

        def get_traffic_light_phases(self):
            """docstring"""

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
            """docstring"""

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
            """docstring"""

            lane_distriubtion = self.lane_info.get('distribution')
            lane_number = self.lane_info.get('number')
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
            """docstring"""

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
                            intralane_gaps = [self.mother.distances(initial_coordinates=(refy[i],refx[i]),final_coordinates=(nxty[i],_),wgs=self.wgs).get_distance() - 0.5*(vlengths[i]+vlengths[i+1]) for i,_ in enumerate(nxtx)]
                            final_gaps = [round(value,ndigits=1) if value>0 else 1 for value in intralane_gaps]
                            if len(final_gaps)>0:
                                final_gaps.append(-1.0) #for the leader
                            temp_dict[f'lane {k-1}']=final_gaps
                gaps.append(temp_dict)
            self.gaps = gaps
            return gaps

        def get_queue_info(self, speed_threshold:float, gap_threshold:float) -> list:
            """docstring"""
            
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
                    for i,identity in enumerate(id_at_green):
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
        """docstring"""

        def __init__(self, mother:'Wiz', data:dict, spatio_temporal_info:dict):
            """docstring"""

            self.mother = mother
            needed_keys_data = ['id','vtype','x','y','time','speed']
            needed_keys_info = ['wgs','bbox','intersection center','time axis']
            if any(key not in data.keys() for key in needed_keys_data ) or any(key not in spatio_temporal_info.keys() for key in needed_keys_info ):
                raise KeyError(f'data dictionary needs keys {needed_keys_data} to work! \nspatiotemporal info directory needs {needed_keys_info} keys to work!')

            self.data = data
            self.spatio_temporal_info = spatio_temporal_info
            self.vehicle_id,self.vehicle_type,self.x, self.y, self.t, self.u = itemgetter('id','vtype','x','y','time','speed')(data)
            self.wgs, self.bbox,self.center,self.time_axis = itemgetter('wgs','bbox','intersection center','time axis')(spatio_temporal_info)
            self.y_center,self.x_center = self.center

        def draw_trajectories(self) -> None:
            """docstring"""

            vmin = int(min(np.mean(set) for set in self.u))
            vmax = int(max(np.mean(set) for set in self.u))

            x_flat = [value for i,vec in enumerate(self.x) for value in vec if self.vehicle_type[i]!='Motorcycle']
            y_flat = [value for i,vec in enumerate(self.y) for value in vec if self.vehicle_type[i]!='Motorcycle']

            u = self.mother.analysis(self.data, self.spatio_temporal_info).get_speed()
            u_flat = [value for i,vec in enumerate(u) for value in vec if self.vehicle_type[i]!='Motorcycle']

            xlim_left,xlim_right=min(x_flat),max(x_flat)
            ylim_bottom,ylim_top = min(y_flat),max(y_flat)

            _fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6),dpi=400)
            ax.set_facecolor('black')
            scatter = ax.scatter(x_flat,y_flat,c=u_flat,cmap='gist_rainbow_r',vmin=vmin,vmax=vmax,s=0.05)
            cbar = plt.colorbar(scatter,ax=ax)
            cbar.set_label('Speed (km/h)')
            plt.title('Trajectories')
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')
            plt.xlim(xlim_left,xlim_right)
            plt.ylim(ylim_bottom,ylim_top)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            plt.tight_layout()
            plt.show(close=True)

        def draw_trajectories_od(self, valid_od_pairs:list, only_colored_trajectories=False) -> None:
            """docstring"""

            od_pairs = self.mother.analysis(self.data, self.spatio_temporal_info).get_od_pairs()
            ll_y,ll_x=self.bbox[0]
            lr_y,lr_x=self.bbox[1]
            ur_y,ur_x=self.bbox[2]
            ul_y,ul_x=self.bbox[3]
            colors_and_alphas = [('blue',0.05),('orange',1),('red',1),('green',1),('violet',0.7),('cyan',1)]
            
            if only_colored_trajectories is False:
                _fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(12,5))
                for l,_ in enumerate(self.x):
                    if od_pairs[l] in valid_od_pairs and self.vehicle_type[l]!='Motorcycle':
                        ax[0].plot(self.x[l],self.y[l],color='k',linewidth=0.5,alpha=colors_and_alphas[valid_od_pairs.index(od_pairs[l])][-1])
                ax[0].plot([element[-1] for element in self.bbox]+[self.bbox[0][-1]],[element[0] for element in self.bbox] + [self.bbox[0][0]],color='blue',linewidth=0.5)
                ax[0].plot([ul_x,self.x_center,lr_x],[ul_y,self.y_center,lr_y],color='blue',linewidth=0.5)
                ax[0].plot([ll_x,self.x_center,ur_x],[ll_y,self.y_center,ur_y],color='blue',linewidth=0.5)
                ax[0].text(0.5*(ll_x+lr_x),0.5*(ll_y+lr_y),'1',fontsize=14,fontweight='bold',color='red')
                ax[0].text(0.5*(ll_x+ul_x),0.5*(ll_y+ul_y),'2',fontsize=14,fontweight='bold',color='red')
                ax[0].text(0.5*(ul_x+ur_x),0.5*(ul_y+ur_y),'3',fontsize=14,fontweight='bold',color='red')
                ax[0].text(0.5*(ur_x+lr_x),0.5*(ur_y+lr_y),'4',fontsize=14,fontweight='bold',color='red')
                ax[1].set_facecolor('black')
                for l,_ in enumerate(self.x):
                    if od_pairs[l] in valid_od_pairs:
                        ax[1].plot(self.x[l],self.y[l],color=colors_and_alphas[valid_od_pairs.index(od_pairs[l])][0],linewidth=1,alpha=colors_and_alphas[valid_od_pairs.index(od_pairs[l])][-1])
                ax[1].plot([element[-1] for element in self.bbox]+[self.bbox[0][-1]],[element[0] for element in self.bbox]+[self.bbox[0][0]],color='white')
                for axis in ax:
                    axis.set_xlim(min(ll_x,ul_x),max(ur_x,lr_x))
                    axis.set_ylim(min(ll_y,lr_y),max(ul_y,ur_y))
                    axis.set_xticks([])
                    axis.set_yticks([])
                    axis.set_xlabel('x coordinate')
                    axis.set_ylabel('y coordinate')
            else:
    
                _fig,ax = plt.subplots(figsize=(10,8))
                ax.set_title('Vehicle Trajectories')
                ax.set_facecolor('white')
                for l,_lista in enumerate(self.x):
                    if od_pairs[l] in valid_od_pairs and self.vehicle_type[l]!='Motorcycle':
                        ax.plot(self.x[l],self.y[l],color=colors_and_alphas[valid_od_pairs.index(od_pairs[l])][0],linewidth=1,alpha=colors_and_alphas[valid_od_pairs.index(od_pairs[l])][-1])
                ax.plot([element[-1] for element in self.bbox]+[self.bbox[0][-1]],[element[0] for element in self.bbox]+[self.bbox[0][0]],color='k')
                ax.set_xlim(min(ll_x,ul_x),max(ur_x,lr_x))
                ax.set_ylim(min(ll_y,lr_y),max(ul_y,ur_y))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('x coordinate')
                ax.set_ylabel('y coordinate')

            plt.show(close=True)

        def draw_spacetime_diagram(self) -> None:
            """docstring"""

            vmin = int(min(np.mean(set) for set in self.u))
            vmax = int(max(np.mean(set) for set in self.u))
            x = [value for set in self.t for value in set]
            y = [value for set in self.mother.analysis(self.data,self.spatio_temporal_info).get_distance_travelled() for value in set]
            z = [value for set in self.u for value in set]
            _fig,ax= plt.subplots(nrows=1,ncols=1,figsize=(18,5))
            scatter = ax.scatter(x,y,c=z,cmap='jet_r',vmin=vmin,vmax=vmax,s=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Speed (km/h)')
            plt.title('Spacetime Diagram')
            plt.xlabel('t (s)')
            plt.ylabel('Distance Travelled (m)')
            plt.tight_layout()
            plt.show(close=True)

        def draw_distance_travelled(self, vehicle_id:int):
            """docstring"""

            _fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4))
            ax.plot(self.t[self.vehicle_id.index(vehicle_id)],self.mother.analysis(self.data, self.spatio_temporal_info).get_distance_travelled()[self.vehicle_id.index(vehicle_id)],color='k',label=f'Vehicle ID: {vehicle_id}')
            plt.title('Distance Travalled vs Time')
            plt.xlabel('t (s)')
            plt.ylabel('Distance Travelled (m)')
            plt.tight_layout()
            plt.grid(True)
            plt.show(close=True)

        def draw_speed(self, vehicle_id:int):
            """docstring"""

            _fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4))
            ax.plot(self.t[self.vehicle_id.index(vehicle_id)],self.u[self.vehicle_id.index(vehicle_id)],color='blue',label=f'Vehicle ID: {vehicle_id}')
            plt.title('Speed vs Time')
            plt.xlabel('t (s)')
            plt.ylabel('Speed (km/h)')
            plt.tight_layout()
            plt.grid(True)
            plt.show(close=True)

        def draw_acceleration(self, vehicle_id:int):
            """docstring"""

            a = self.mother.analysis(self.data, self.spatio_temporal_info).get_acceleration()
            _fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4))
            ax.plot(self.t[self.vehicle_id.index(vehicle_id)],a[self.vehicle_id.index(vehicle_id)],color='red',label=f'Vehicle ID: {vehicle_id}')
            plt.title('Acceleration vs Time')
            plt.xlabel('t (s)')
            plt.ylabel(r'Acceleration $(m/s^2)$')
            plt.tight_layout()
            plt.grid(True)
            plt.show(close=True)
