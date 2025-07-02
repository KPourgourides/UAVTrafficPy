import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from shapely import Point,Polygon
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

class Wiz:
    
        def __init__(self):
            pass

        def Distances(self, initial_coordinates:tuple, final_coordinates:tuple, WGS:bool):
            return self._Distances(self, initial_coordinates, final_coordinates, WGS)

        def DataLoader(self, Raw_VD:dict, WGS:bool, bbox:list):
            return self._DataLoader(self, Raw_VD, WGS, bbox)

        def Analysis(self, VD:dict, SpatioTemporalInfo:dict):
            return self._Analysis(self, VD, SpatioTemporalInfo)

        def Visualization(self, VD:dict, SpatioTemporalInfo:dict):
            return self._Visualization(self, VD, SpatioTemporalInfo)
    
        class _Distances:

            def __init__(self, mother:'Wiz', initial_coordinates:tuple, final_coordinates:tuple, WGS:bool):

                self.mother = mother
                self.y_i,self.x_i = initial_coordinates
                self.y_f,self.x_f = final_coordinates
                self.WGS = WGS
                self.factor = 2*np.pi*6371000/360

            def get_Dx(self) -> float:
                return (self.factor*float(np.cos(np.deg2rad(self.y_i)))*(self.x_f - self.x_i))*(self.WGS) + (self.x_f - self.x_i)*(not self.WGS)
            
            def get_Dy(self) -> float:
                return (self.factor*(self.y_f - self.y_i))*(self.WGS) + (self.y_f - self.y_i)*(not self.WGS)
            
            def get_Distance(self) -> float:
                dx = self.get_Dx()
                dy = self.get_Dy()
                return pow((pow(dx,2) + pow(dy,2)),0.5) 

        class _DataLoader:

            def __init__(self, mother:'Wiz', Raw_VD:dict, WGS:bool, bbox:list):
                self.mother = mother
                needed_keys = ['id','vtype','x','y','time','speed']
                if any(key not in Raw_VD.keys() for key in needed_keys ):
                    print(f'Error: vehicle dictionary needs keys {needed_keys} to work!')
                    return()
                else:
                    self.Raw_VD = Raw_VD
                    self.id,self.vtype,self.x,self.y,self.t,self.u  = itemgetter('id','vtype','x','y','time','speed')(Raw_VD)
                    self.WGS = WGS
                    self.bbox = bbox

            def get_Intersection_VD(self) -> dict:
                box = Polygon([(lati, long) for lati, long in self.bbox])
                x_,y_,t_,u_,id_,vtype_=[],[],[],[],[],[]
                for i,vec in enumerate(self.x):
                    flag=False
                    for j,element in enumerate(vec):
                        if box.contains(Point(self.y[i][j], self.x[i][j])):
                            flag=True
                            index_start = j
                            break
                    if flag:
                        for k,element in enumerate(vec[index_start+1:]):
                            index_end=k+index_start+1
                            if box.contains(Point(self.y[i][index_end], self.x[i][index_end])):
                                continue
                            else:
                                break
                            
                        if len(self.t[i][index_start:index_end])>1:
                            id_.append(self.id[i])
                            vtype_.append(self.vtype[i])
                            x_.append(self.x[i][index_start:index_end])
                            y_.append(self.y[i][index_start:index_end])
                            t_.append(self.t[i][index_start:index_end])
                            u_.append(self.u[i][index_start:index_end])
                Intersection_VD ={'id':id_,'vtype':vtype_,'x':x_,'y':y_,'time':t_,'speed':u_}

                return Intersection_VD

            def get_Filtered_VD(self, cursed_ids:list) -> dict:
                Intersection_VD  = self.get_Intersection_VD()
                id,vtype,x,y,t,u = itemgetter('id','vtype','x','y','time','speed')(Intersection_VD)
                id_,vtype_,x_,y_,t_,u_=[],[],[],[],[],[]
                for i,vec in enumerate(x):

                    immobility = sum([1 if self.mother.Distances(initial_coordinates=(y[i][j-1],x[i][j-1]),final_coordinates=(y[i][j],x[i][j]),WGS=self.WGS).get_Distance() <1e-4 else 0 for j,element in enumerate(vec[1:])])
                    if  id[i] in cursed_ids or immobility>=0.95*len(vec):
                        continue
                    else:
                        id_.append(id[i])
                        vtype_.append(vtype[i])
                        x_.append(x[i])
                        y_.append(y[i])
                        t_.append(t[i])
                        u_.append(u[i])
                Filtered_VD = {'id':id_,'vtype':vtype_,'x':x_,'y':y_,'time':t_,'speed':u_}
                return Filtered_VD
            
        class _Analysis:

            def __init__(self, mother:'Wiz', VD:dict, SpatioTemporalInfo:dict):

                self.mother = mother

                needed_keys_VD = ['id','vtype','x','y','time','speed']
                needed_keys_STI = ['WGS','bbox','y center','x center','time axis']

                if any(key not in VD.keys() for key in needed_keys_VD ):
                    print(f'Error: vehicle dictionary needs keys {needed_keys_VD} to work!')
                    return()
                if any(key not in SpatioTemporalInfo.keys() for key in needed_keys_STI ):
                    print(f'Error: spatial info dictionary needs keys {needed_keys_STI} to work!')
                    return()

                else:
                    self.VD = VD
                    self.SpatioTemporalInfo = SpatioTemporalInfo
                    self.id,self.vtype,self.x, self.y, self.t, self.u = itemgetter('id','vtype','x','y','time','speed')(VD)
                    self.WGS, self.bbox, self.y_center,self.x_center, self.time_axis = itemgetter('WGS','bbox','y center','x center', 'time axis')(SpatioTemporalInfo)


            def get_DistanceTravelled(self) ->list:

                distance_travelled=[]
                for i,vec in enumerate(self.x):
                    temp_distance_travelled=[]
                    temp_sum=0
                    for j,element in enumerate(vec):
                        if j==0:
                            temp_distance_travelled.append(0)
                        else:
                            temp_sum+= self.mother.Distances(initial_coordinates=(self.y[i][j-1],self.x[i][j-1]),final_coordinates=(self.y[i][j],self.x[i][j]),WGS=self.WGS).get_Distance()
                            temp_distance_travelled.append(temp_sum)
                    distance_travelled.append(temp_distance_travelled)

                self.distance_travelled = distance_travelled
                return self.distance_travelled

            def get_Speed(self) -> list:

                distance_travelled = self.get_DistanceTravelled()

                km_per_h = True
                multiplier = 3.6*(km_per_h) + 1.0*(not km_per_h)

                u = [[float(value) for value in np.gradient(distance_travelled[i],self.t[i])*multiplier] for i in range(len(distance_travelled))]

                smoothing_factor=2
                u_smooth = [gaussian_filter(vec,sigma=smoothing_factor).tolist() for vec in u]

                self.u = u_smooth
                return self.u

            def get_Acceleration(self) -> list:

                u = self.get_Speed()
                multiplier  = 1000/3600

                a = [[float(value)*multiplier for value in np.gradient(u[i],self.t[i])] for i in range(len(u))]
                smoothing_factor=2
                a_smooth = [gaussian_filter(vec,sigma=smoothing_factor).tolist() for vec in a]

                self.a = a_smooth
                return self.a

            def get_ODPairs(self) -> list:

                def is_inside_triangle(A, B, C, P):
                    triangle = Polygon([A, B, C])
                    point = Point(P)
                    return triangle.contains(point) or triangle.touches(point)

                ll_y,ll_x=self.bbox[0]
                lr_y,lr_x=self.bbox[1]
                ur_y,ur_x=self.bbox[2]
                ul_y,ul_x=self.bbox[3]


                triangle_1 = [(ll_x,ll_y),(self.x_center,self.y_center),(lr_x,lr_y)]
                triangle_2 = [(ul_x,ul_y),(self.x_center,self.y_center),(ll_x,ll_y)]
                triangle_3 = [(ur_x,ur_y),(self.x_center,self.y_center),(ul_x,ul_y)]
                triangle_4 = [(lr_x,lr_y),(self.x_center,self.y_center),(ur_x,ur_y)]

                OD_pairs=[]
                for v,vec in enumerate(self.x):

                    origin=0
                    destination=0

                    o = (self.x[v][0],self.y[v][0])
                    d = (self.x[v][-1],self.y[v][-1])

                    for t,Triangle in enumerate([triangle_1,triangle_2,triangle_3,triangle_4]):

                        if is_inside_triangle(*Triangle,o)==True:
                            origin=t+1
                            break
                        
                    for t,Triangle in enumerate([triangle_1,triangle_2,triangle_3,triangle_4]):

                          if is_inside_triangle(*Triangle,d)==True:
                            destination=t+1
                            break
                        
                    OD_pairs.append((origin,destination))
                self.OD_pairs = OD_pairs

                return self.OD_pairs


            def get_OD_VD(self, desirable_pairs:list) -> list:
            
                id_OD,vtype_OD,x_OD,y_OD,t_OD,u_OD,=[],[],[],[],[],[]

                OD_pairs = self.OD_pairs

                for p,pair in enumerate(OD_pairs):
                    if pair in desirable_pairs:
                        id_OD.append(self.id[p])
                        vtype_OD.append(self.vtype[p])
                        x_OD.append(self.x[p])
                        y_OD.append(self.y[p])
                        t_OD.append(self.t[p])
                        u_OD.append(self.u[p])

                OD_VD = {'id':id_OD,'vtype':vtype_OD,'x':x_OD,'y':y_OD,'time':t_OD,'speed':u_OD,}

                return OD_VD


            def get_dFromBboxEdge(self, flow_direction:str) -> list:

                if flow_direction not in ['up','down','right','left']:
                    print('Invalid flow direction')
                    return()

                y1,x1 = self.bbox[0]
                y2,x2 = self.bbox[1]*(flow_direction in ['right','left']) + self.bbox[3]*(flow_direction in ['up','down'])

                B_x = self.mother.Distances(initial_coordinates=(y1,x1),final_coordinates=(y2,x2),WGS=self.WGS).get_Dx()
                B_y = self.mother.Distances(initial_coordinates=(y1,x1),final_coordinates=(y2,x2),WGS=self.WGS).get_Dy()
                B = pow((pow(B_x,2)+pow(B_y,2)),0.5)

                d_from_edge=[]

                for i,vec in enumerate(self.y):
                    temp_dfromedge=[]
                    for j,element in enumerate(vec):

                        if self.WGS:
                            A_x = self.mother.Distances(initial_coordinates=(y1,x1),final_coordinates=(self.y[i][j],self.x[i][j]),WGS=self.WGS).get_Dx()
                            A_y = self.mother.Distances(initial_coordinates=(y1,x1),final_coordinates=(self.y[i][j],self.x[i][j]),WGS=self.WGS).get_Dy()
                        else:
                            A_x = self.x[i][j]-x1
                            A_y = self.y[i][j]-y1

                        temp_dfromedge.append(abs(A_x*B_y - A_y*B_x)/B)

                    d_from_edge.append(temp_dfromedge)

                self.d_from_edge = d_from_edge
                self.flow_direction = flow_direction

                return self.d_from_edge


            def get_LaneInfo(self, flow_direction:str) -> None:


                d_from_edge=self.get_dFromBboxEdge(flow_direction)
                flat_d_from_edge = [element for i,vec in enumerate(d_from_edge) for j,element in enumerate(vec) if self.vtype[i]!='Motorcycle']

                #--------------------------------------------------------
                plt.figure(figsize=(10,2))
                nbins=200
                plt.hist(flat_d_from_edge, color='blue', bins=nbins, density=True)
                plt.xlabel('Distance from bbox edge (m)')
                plt.ylabel('Normalized Occurences')
                plt.xticks(np.arange(np.trunc(np.mean(flat_d_from_edge)-20), np.trunc(np.mean(flat_d_from_edge)+20),2))
                plt.xlim(np.trunc(np.mean(flat_d_from_edge)-20), np.trunc(np.mean(flat_d_from_edge)+20))
                plt.title("Lane Distribution")
                plt.show(close=True)
                #--------------------------------------------------------
                n_clusters = int(input('How many lanes?'))

                if n_clusters>1:
                    low_lim = int(input('low limit?'))
                    high_lim = int(input('high limit?'))
                    bounded_d_from_edge = [element for element in flat_d_from_edge if low_lim<element<high_lim]
                    #--------------------------------------------------------
                    data_for_clustering = np.sort(np.array(bounded_d_from_edge))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_for_clustering.reshape(-1, 1))
                    centers = np.sort(kmeans.cluster_centers_.flatten())
                    boundaries = (centers[:-1] + centers[1:]) / 2
                    #--------------------------------------------------------
                
                    plt.figure(figsize=(10,2))
                    plt.hist(bounded_d_from_edge,bins=nbins,color='red',density=True)
                    for boundary in boundaries:
                        plt.axvline(boundary, color='black', linestyle='--',linewidth=2)
                    plt.title("Lane Clustering with Boundaries")
                    plt.xlabel('Distance from bbox edge (m)')
                    plt.ylabel('Probability Density')
                    plt.tight_layout()
                    plt.show(close=True)
                #--------------------------------------------------------   
                
                try:
                    lane_boundaries = [0] + [float(value) for value in boundaries] + [1e3]
                except:
                    lane_boundaries = [0,1e3]

                lane_number = len(lane_boundaries)-1
                lane_distribution=[]

                for i,vec in enumerate(d_from_edge):
                    temp_lanes=[]
                    for j,element in enumerate(vec):
                        temp_lanes.append(max([b for b,boundary in enumerate(lane_boundaries) if element>boundary]))
                    lane_distribution.append(temp_lanes)


                lane_info = {
                    'number':lane_number,
                    'boundaries':lane_boundaries,
                    'distribution':lane_distribution
                    }

                self.lane_info = lane_info
                return lane_info


            def get_FlowInfo(self, detector_positions:list) -> list:

                x_detector_index = 1*self.WGS
                y_detector_index = 1 - x_detector_index
                self.detector_positions = detector_positions

                lane_distribution = self.lane_info.get('distribution')
                flow_info=[]

                for m,moment in enumerate(self.time_axis):

                    tempdict={}
                    tempid=[]

                    for v,vec in enumerate(self.x):

                        if self.time_axis[m] in self.t[v] and self.time_axis[m-1] in self.t[v]:

                            lane = lane_distribution[v][self.t[v].index(moment)]

                            # QoI = Quantity of Interest
                            QoI_now = (self.x[v][self.t[v].index(self.time_axis[m])])*(self.flow_direction in ['left','right']) + (self.y[v][self.t[v].index(self.time_axis[m])])*(self.flow_direction in ['up','down'])
                            QoI_before = (self.x[v][self.t[v].index(self.time_axis[m-1])])*(self.flow_direction in ['left','right']) + (self.y[v][self.t[v].index(self.time_axis[m-1])])*(self.flow_direction in ['up','down'])
                            QoI_detector_index = x_detector_index*(self.flow_direction in ['left','right']) + y_detector_index*(self.flow_direction in ['up','down'])
                            before_detector = 'lower'*(self.flow_direction in ['up','right']) + 'higher'*(self.flow_direction in ['left','down'])

                            if before_detector=='lower':
                                if QoI_now>=detector_positions[lane][QoI_detector_index] and QoI_before<detector_positions[lane][QoI_detector_index]:
                                    tempid.append(self.id[v])
                            elif before_detector=='higher':
                                if QoI_now<=detector_positions[lane][QoI_detector_index] and QoI_before>detector_positions[lane][QoI_detector_index]:
                                    tempid.append(self.id[v])

                    tempdict['time stamp'] = moment
                    tempdict['flow'] = len(tempid)
                    tempdict['id'] = tempid
                    flow_info.append(tempdict)

                self.flow_info = flow_info

                return self.flow_info

            def get_NormalizedFlow(self, threshold:int) -> list:

                flow_info = self.flow_info
                flow=[_.get('flow') for i,_ in enumerate(flow_info)]
                self.flow=flow

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

                return self.flow, self.normalized_flow

            def get_CursedId(self, lowlim:float, highlim:float) -> None:
            
                sampling_period = round(self.time_axis[1]-self.time_axis[0])
                for i,_ in enumerate(self.flow_info):
                    if _.get('time stamp') in  np.arange(lowlim,highlim+sampling_period,sampling_period):
                        print(f' ids {_.get('id')} passed at time {_.get('time stamp')}')
                return None


            def get_TrafficLightPhases(self):

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
                for d,dict in enumerate(trafficlights):
                
                    if d+1<len(trafficlights):
                        dict['Duration OFF'] = round(trafficlights[d+1].get('Green') - trafficlights[d].get('Red'),ndigits=1)
                        dict['Phase Duration'] = round(trafficlights[d].get('Duration ON') + trafficlights[d].get('Duration OFF'),ndigits=1)
                    else:
                        dict['Duration OFF'] = None
                        dict['Phase Duration'] = None

                    trafficlightphases.append(dict)

                self.trafficlightphases = trafficlightphases
                return self.trafficlightphases


            def get_TrafficLightCycles(self, *traffic_lights_phases:dict) -> dict:

                traffic_lights_one,traffic_lights_two=traffic_lights_phases
                cycles=[]
                for p,phase in enumerate(traffic_lights_one):

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
                return cycles

            def get_SortedId(self):
            
                flow_direction = self.flow_direction

                lane_distriubtion = self.lane_info.get('distribution')
                lane_number = self.lane_info.get('number')

                if flow_direction not in ['up','down','left','right']:
                    print('Wrong flow_direction or sort_by input')
                    return()

                QOI = self.y*(flow_direction in ['up','down']) + self.x*(flow_direction in ['left','right'])

                sorted_id=[]
                for m,moment in enumerate(self.time_axis):

                    temp_dict = {'time stamp':moment }

                    for l in range(lane_number):

                        temp_lane_ids=[]
                        temp_lane_QOI=[]

                        for v,vec in enumerate(QOI):

                            if moment in self.t[v] and lane_distriubtion[v][self.t[v].index(moment)]==l:

                                temp_lane_ids.append(self.id[v])
                                temp_lane_QOI.append(QOI[v][self.t[v].index(moment)])


                        if len(temp_lane_ids)==0:
                            temp_sorted_id=None
                        else:
                            temp_sorted_id = [temp_lane_ids for _, temp_lane_ids in sorted(zip(temp_lane_QOI, temp_lane_ids),reverse=(flow_direction in ['left','down']))]

                        temp_dict[f'lane {l}']= temp_sorted_id

                    sorted_id.append(temp_dict) 

                self.sorted_id = sorted_id
                return self.sorted_id

            def get_Gaps(self) -> list:

                sorted_id = self.sorted_id
                gaps=[]

                for d,dict in enumerate(sorted_id):

                    moment = dict.get('time stamp')
                    temp_dict = {'time stamp': moment}

                    for k,key in enumerate(dict): #each key for k>0 corresponds to the different lanes

                        if k>0: #skip the time stamp

                            sorted_id_in_this_lane = dict.get(key)

                            if sorted_id_in_this_lane==None:
                                temp_dict[f'lane {k-1}']=None

                            elif len(sorted_id_in_this_lane)==1:
                                temp_dict[f'lane {k-1}']=[-1.0]

                            else:
                                refx = [self.x[self.id.index(value)][self.t[self.id.index(value)].index(moment)] for v,value in enumerate(sorted_id_in_this_lane)]
                                refy = [self.y[self.id.index(value)][self.t[self.id.index(value)].index(moment)] for v,value in enumerate(sorted_id_in_this_lane)]

                                nxtx = [self.x[self.id.index(sorted_id_in_this_lane[v+1])][self.t[self.id.index(sorted_id_in_this_lane[v+1])].index(moment)] for v,value in enumerate(sorted_id_in_this_lane[:-1])]
                                nxty = [self.y[self.id.index(sorted_id_in_this_lane[v+1])][self.t[self.id.index(sorted_id_in_this_lane[v+1])].index(moment)] for v,value in enumerate(sorted_id_in_this_lane[:-1])]

                                vtypes = [self.vtype[self.id.index(value)] for v,value in enumerate(sorted_id_in_this_lane)]
                                vlengths = [5*(vt=='Car' or vt=='Taxi')+5.83*(vt=='Medium')+12.5*(vt=='Heavy' or vt=='Bus')+2.5*(vt=='Motorcycle') for vt in vtypes]

                                intralane_gaps = [self.mother.Distances(initial_coordinates=(refy[i],refx[i]),final_coordinates=(nxty[i],nxtx[i]),WGS=self.WGS).get_Distance() - 0.5*(vlengths[i]+vlengths[i+1]) for i,_ in enumerate(nxtx)]
                                final_gaps = [round(value,ndigits=1) if value>0 else 1 for value in intralane_gaps]

                                if len(final_gaps)>0:
                                    final_gaps.append(-1.0) #for the leader

                                temp_dict[f'lane {k-1}']=final_gaps

                    gaps.append(temp_dict)
                return gaps

            def get_QueueInfo(self, speed_threshold:float, gap_threshold:float) -> list:
            
                traffic_light_phases = self.trafficlightphases
                before_detector = 'lower'*(self.flow_direction in ['up','right']) + 'higher'*(self.flow_direction in ['down','left'])
                QOI = self.x*(self.flow_direction in ['left','right']) + self.y*(self.flow_direction in ['up','down'])
                detector_index = 0*(QOI==self.y) + 1*(QOI==self.x)

                queue_info=[]

                for p,phase in enumerate(traffic_light_phases):

                    phase_info=[]

                    green = phase.get('Green')
                    red = phase.get('Red')


                    for l in range(self.lane_info.get('number')):

                        temp_info={'Lane':l}
                        id_at_green = self.sorted_id[self.time_axis.index(green)].get(f'lane {l}')
                        if id_at_green==None:

                            temp_info['Queued Vehicles']=None
                            temp_info['Queue Length']=None
                            temp_info['Dissipation Duration']=None

                        else:
                            id_before_trafficlight=[]

                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            for i,identity in enumerate(id_at_green):

                                vehicle_index = self.id.index(identity)
                                green_index  = self.t[self.id.index(identity)].index(green)

                                if self.u[vehicle_index][green_index]<=speed_threshold:
                                    #---------------------------
                                    if before_detector=='lower':
                                        #-----------------------------------------------------------------------------
                                        if QOI[vehicle_index][green_index]<=self.detector_positions[l][detector_index]:

                                            if red not in self.t[vehicle_index]:
                                                id_before_trafficlight.append(identity)
                                            else:
                                                red_index  = self.t[vehicle_index].index(red)
                                                if QOI[vehicle_index][red_index]>=self.detector_positions[l][detector_index]:
                                                    id_before_trafficlight.append(identity)

                                    #---------------------------
                                    elif before_detector=='higher':
                                        #-----------------------------------------------------------------------------
                                        if QOI[vehicle_index][green_index]>=self.detector_positions[l][detector_index]:

                                            if red not in self.t[vehicle_index]:
                                                id_before_trafficlight.append(identity)
                                            else:
                                                red_index  = self.t[vehicle_index].index(red)
                                                if QOI[vehicle_index][red_index]<=self.detector_positions[l][detector_index]:
                                                    id_before_trafficlight.append(identity)
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                            if len(id_before_trafficlight)==0:
                                temp_info['Queued Vehicles']=None
                                temp_info['Queue Length']=None
                                temp_info['Dissipation Duration']=None

                            else:

                                gaps_before_trafficlight = [self.get_Gaps()[self.time_axis.index(green)].get(f'lane {l}')[i] for i,identity in enumerate(id_at_green) if identity in id_before_trafficlight[:-1]]
                                revised_id_before_trafficlight = [identity for i,identity in enumerate(id_before_trafficlight[:-1]) if gaps_before_trafficlight[i]<=gap_threshold]+[id_before_trafficlight[-1]]
                                types_before_trafficlight = [self.vtype[self.id.index(identity)] for i,identity in enumerate(revised_id_before_trafficlight)]
                                lengths_before_trafficlight = [5*(vt=='Car' or vt=='Taxi')+ 5.83*(vt=='Medium') + 12.5*(vt=='Heavy' or vt=='Bus') + 2.5*(vt=='Motorcycle') for vt in types_before_trafficlight]

                                critical_condition_1 = len(revised_id_before_trafficlight)>0
                                critical_condition_2 = len(revised_id_before_trafficlight)==0

                                if critical_condition_1:

                                    temp_info['Queued Vehicles'] = len(revised_id_before_trafficlight)
                                    temp_info['Queue Length'] = round(sum(gaps_before_trafficlight)+sum(lengths_before_trafficlight),ndigits=0)

                                    flag=False
                                    for m,moment in enumerate(self.time_axis[self.time_axis.index(green):]):
                                        last_vehicle = id_before_trafficlight[0] #leading vehicle is last in list, last vehicle always first in list 
                                        if moment in self.t[self.id.index(last_vehicle)]:
                                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                            if before_detector=='lower':
                                                if QOI[self.id.index(last_vehicle)][self.t[self.id.index(last_vehicle)].index(moment)]>=self.detector_positions[l][detector_index]:
                                                    temp_info['Dissipation Duration'] = round(moment-green,ndigits=1)
                                                    flag=True
                                                    break
                                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                            elif before_detector=='higher':
                                                if QOI[self.id.index(last_vehicle)][self.t[self.id.index(last_vehicle)].index(moment)]<=self.detector_positions[l][detector_index]:
                                                    temp_info['Dissipation Duration'] = round(moment-green,ndigits=1)
                                                    flag=True
                                                    break
                                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        else:
                                            continue
                                    if flag==False:
                                        temp_info['Dissipation Time'] = round(red-green,ndigits=1)

                                elif critical_condition_2:
                                    temp_info['Queued Vehicles'] = None
                                    temp_info['Queue Length'] = None
                                    temp_info['Dissipation Duration'] = None    

                        phase_info.append(temp_info)
                    queue_info.append(phase_info)
                return queue_info
            
        
        class _Visualization:
 
            def __init__(self, mother:'Wiz', VD:dict, SpatioTemporalInfo:dict):
            
                self.mother = mother
                needed_keys_VD = ['id','vtype','x','y','time','speed']
                needed_keys_STI = ['WGS','bbox','y center','x center','time axis']
                if any(key not in VD.keys() for key in needed_keys_VD ):
                    print(f'Error: vehicle dictionary needs keys {needed_keys_VD} to work!')
                    return()
                if any(key not in SpatioTemporalInfo.keys() for key in needed_keys_STI ):
                    print(f'Error: spatial info dictionary needs keys {needed_keys_STI} to work!')
                    return()
                else:
                    self.VD = VD
                    self.SpatioTemporalInfo = SpatioTemporalInfo
                    self.id,self.vtype,self.x, self.y, self.t, self.u = itemgetter('id','vtype','x','y','time','speed')(VD)
                    self.WGS, self.bbox, self.y_center,self.x_center, self.time_axis = itemgetter('WGS','bbox','y center','x center', 'time axis')(SpatioTemporalInfo)

            def Trajectories(self) -> None:

                vmin = int(min([np.mean(set) for set in self.u]))
                vmax = int(max([np.mean(set) for set in self.u]))

                x_flat = [value for vec in self.x for value in vec]
                y_flat = [value for vec in self.y for value in vec]
                u_flat = [value for vec in self.u for value in vec]

                xlim_left,xlim_right=min(x_flat),max(x_flat)
                ylim_bottom,ylim_top = min(y_flat),max(y_flat)

                fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6))
                ax.set_facecolor('black') 
                scatter = ax.scatter(x_flat,y_flat,c=u_flat,cmap='jet',vmin=vmin,vmax=vmax,s=0.1)

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

            def Trajectories_OD(self, valid_OD_pairs:list) -> None:

                OD_pairs = self.mother.Analysis(self.VD, self.SpatioTemporalInfo).get_ODPairs()

                ll_y,ll_x=self.bbox[0]
                lr_y,lr_x=self.bbox[1]
                ur_y,ur_x=self.bbox[2]
                ul_y,ul_x=self.bbox[3]


                colors_and_alphas = [('blue',0.05),('orange',0.5),('red',1),('green',0.5),('white',1),('magenta',1)]

                only_colored_trajectories=False

                if only_colored_trajectories is False:

                    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(12,5))
                    for l,lista in enumerate(self.x):
                        if OD_pairs[l] in valid_OD_pairs and self.vtype[l]!='Motorcycle':
                            ax[0].plot(self.x[l],self.y[l],color='k',linewidth=0.5,alpha=colors_and_alphas[valid_OD_pairs.index(OD_pairs[l])][-1])

                    ax[0].plot([element[-1] for element in self.bbox]+[self.bbox[0][-1]],[element[0] for element in self.bbox] + [self.bbox[0][0]],color='blue',linewidth=0.5)
                    ax[0].plot([ul_x,self.x_center,lr_x],[ul_y,self.y_center,lr_y],color='blue',linewidth=0.5)
                    ax[0].plot([ll_x,self.x_center,ur_x],[ll_y,self.y_center,ur_y],color='blue',linewidth=0.5)

                    ax[0].text(0.5*(ll_x+lr_x),0.5*(ll_y+lr_y),'1',fontsize=14,fontweight='bold',color='red')
                    ax[0].text(0.5*(ll_x+ul_x),0.5*(ll_y+ul_y),'2',fontsize=14,fontweight='bold',color='red')
                    ax[0].text(0.5*(ul_x+ur_x),0.5*(ul_y+ur_y),'3',fontsize=14,fontweight='bold',color='red')
                    ax[0].text(0.5*(ur_x+lr_x),0.5*(ur_y+lr_y),'4',fontsize=14,fontweight='bold',color='red')


                    ax[1].set_facecolor('black') 
                    for l,lista in enumerate(self.x):
                        if(OD_pairs[l] in valid_OD_pairs):
                            ax[1].plot(self.x[l],self.y[l],color=colors_and_alphas[valid_OD_pairs.index(OD_pairs[l])][0],linewidth=1,alpha=colors_and_alphas[valid_OD_pairs.index(OD_pairs[l])][-1])

                    ax[1].plot([element[-1] for element in self.bbox]+[self.bbox[0][-1]],[element[0] for element in self.bbox]+[self.bbox[0][0]],color='white')

                    for axis in ax:
                        axis.set_xlim(min(ll_x,ul_x),max(ur_x,lr_x))
                        axis.set_ylim(min(ll_y,lr_y),max(ul_y,ur_y))
                        axis.set_xticks([])
                        axis.set_yticks([])
                        axis.set_xlabel('x coordinate')
                        axis.set_ylabel('y coordinate')

                else:

                    fig,ax = plt.subplots(figsize=(10,10),dpi=100)
                    ax.set_title('Trajectory Clusters')
                    ax.set_facecolor('black') 
                    for l,lista in enumerate(self.x):
                        if OD_pairs[l] in valid_OD_pairs and self.vtype[l]!='Motorcycle':
                            ax.plot(self.x[l],self.y[l],color=colors_and_alphas[valid_OD_pairs.index(OD_pairs[l])][0],linewidth=1,alpha=colors_and_alphas[valid_OD_pairs.index(OD_pairs[l])][-1])

                    ax.plot([element[-1] for element in self.bbox]+[self.bbox[0][-1]],[element[0] for element in self.bbox]+[self.bbox[0][0]],color='white')

                    ax.set_xlim(ul_x,lr_x)
                    ax.set_ylim(ll_y,ur_y)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel('x coordinate')
                    ax.set_ylabel('y coordinate')


                plt.show(close=True)

            def SpacetimeDiagram(self) -> None:

                vmin = int(min([np.mean(set) for set in self.u]))
                vmax = int(max([np.mean(set) for set in self.u]))

                x = [value for set in self.t for value in set]
                y = [value for set in self.mother.Analysis(self.VD,self.SpatioTemporalInfo).get_DistanceTravelled() for value in set]
                z = [value for set in self.u for value in set]

                fig,ax= plt.subplots(nrows=1,ncols=1,figsize=(15,4))
                scatter = ax.scatter(x,y,c=z,cmap='jet_r',vmin=vmin,vmax=vmax,s=0.5)

                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Speed (km/h)')

                plt.title('Spacetime Diagram')
                plt.xlabel('t (s)')
                plt.ylabel('Distance Travelled (m)')
                plt.tight_layout()
                plt.show(close=True)

            def DistanceTravelled(self, id:int):

                fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(15,4))
                ax.plot(self.t[self.id.index(id)],self.mother.Analysis(self.VD, self.SpatioTemporalInfo).get_DistanceTravelled()[self.id.index(id)],color='k',label=f'Vehicle ID: {id}')

        
                plt.title('Distance Travalled vs Time')
                plt.xlabel('t (s)')
                plt.ylabel('Distance Travelled vs Time')
                plt.tight_layout()
                plt.show(close=True)

            def Speed(self, id:int):

                fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(15,4))
                ax.plot(self.t[self.id.index(id)],self.u[self.id.index(id)],color='blue',label=f'Vehicle ID: {id}')

                plt.title('Speed vs Time')
                plt.xlabel('t (s)')
                plt.ylabel('Speed (km/h)')
                plt.tight_layout()
                plt.grid(True)
                plt.show(close=True)


            def Acceleration(self, id:int):

                a = self.mother.Analysis(self.VD, self.SpatioTemporalInfo).get_Acceleration()

                fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(15,4))
                ax.plot(self.t[self.id.index(id)],a[self.id.index(id)],color='red',label=f'Vehicle ID: {id}')

                plt.title('Acceleration vs Time')
                plt.xlabel('t (s)')
                plt.ylabel(r'Acceleration $(m/s^2)$')
                plt.tight_layout()
                plt.grid(True)
                plt.show(close=True)
