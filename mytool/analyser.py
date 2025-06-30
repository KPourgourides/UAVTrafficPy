from .distances import _Distances as Distances
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from shapely import Point,Polygon
from sklearn.cluster import KMeans

from operator import itemgetter

class _Analysis:

        def __init__(self, mother, VD:dict, SpatioTemporalInfo:dict):
            
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
                        temp_sum+= Distances(self.mother,initial_coordinates=(self.y[i][j-1],self.x[i][j-1]),final_coordinates=(self.y[i][j],self.x[i][j]),WGS=self.WGS).get_Distance()
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

            B_x = Distances(self.mother,initial_coordinates=(y1,x1),final_coordinates=(y2,x2),WGS=self.WGS).get_Dx()
            B_y = Distances(self.mother,initial_coordinates=(y1,x1),final_coordinates=(y2,x2),WGS=self.WGS).get_Dy()
            B = pow((pow(B_x,2)+pow(B_y,2)),0.5)

            d_from_edge=[]

            for i,vec in enumerate(self.y):
                temp_dfromedge=[]
                for j,element in enumerate(vec):

                    if self.WGS:
                        A_x = Distances(self.mother,initial_coordinates=(y1,x1),final_coordinates=(self.y[i][j],self.x[i][j]),WGS=self.WGS).get_Dx()
                        A_y = Distances(self.mother,initial_coordinates=(y1,x1),final_coordinates=(self.y[i][j],self.x[i][j]),WGS=self.WGS).get_Dy()
                    else:
                        A_x = self.x[i][j]-x1
                        A_y = self.y[i][j]-y1

                    temp_dfromedge.append(abs(A_x*B_y - A_y*B_x)/B)

                d_from_edge.append(temp_dfromedge)

            self.d_from_edge = d_from_edge
            self.flow_direction = flow_direction

            return self.d_from_edge
        

        def get_LaneInfo(self, flow_direction:str, color:str) -> None:


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
            low_lim = int(input('low limit?'))
            high_lim = int(input('high limit?'))
            bounded_d_from_edge = [element for element in flat_d_from_edge if low_lim<element<high_lim]
            #--------------------------------------------------------
            data_for_clustering = np.sort(np.array(bounded_d_from_edge))
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_for_clustering.reshape(-1, 1))
            centers = np.sort(kmeans.cluster_centers_.flatten())
            boundaries = (centers[:-1] + centers[1:]) / 2
            #--------------------------------------------------------
            if n_clusters>1:
                plt.figure(figsize=(10,2))
                plt.hist(bounded_d_from_edge,bins=nbins,color=color,density=True)
                for boundary in boundaries:
                    plt.axvline(boundary, color='black', linestyle='--',linewidth=2)
                plt.title("Lane Clustering with Boundaries")
                plt.xlabel('Distance from bbox edge (m)')
                plt.ylabel('Probability Density')
                plt.tight_layout()
                plt.show(close=True)
             #--------------------------------------------------------   
            lane_boundaries = [0] + [float(value) for value in boundaries] + [1e3]
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

                            intralane_gaps = [Distances(self.mother,initial_coordinates=(refy[i],refx[i]),final_coordinates=(nxty[i],nxtx[i]),WGS=self.WGS).get_Distance() - 0.5*(vlengths[i]+vlengths[i+1]) for i,_ in enumerate(nxtx)]
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