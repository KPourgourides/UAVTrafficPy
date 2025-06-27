from .distances import _Distances as Distances
from operator import itemgetter
from shapely import Point,Polygon


class _DataLoader:
    def __init__(self, mother, Raw_VD:dict, WGS:bool, bbox:list):
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
            
            immobility = sum([1 if Distances(self.mother,initial_coordinates=(y[i][j-1],x[i][j-1]),final_coordinates=(y[i][j],x[i][j]),WGS=self.WGS).get_Distance() <1e-4 else 0 for j,element in enumerate(vec[1:])])
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