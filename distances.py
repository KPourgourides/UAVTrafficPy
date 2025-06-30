import numpy as np

class _Distances:

    def __init__(self, mother, initial_coordinates:tuple, final_coordinates:tuple, WGS:bool):
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