# core.py
from .distances import _Distances
from .dataloader import _DataLoader
from .analyser import _Analysis
from .visualizer import _Visualization

class Wiz:
    
    def __init__(self):
        pass

    def Distances(self, initial_coordinates:tuple, final_coordinates:tuple, WGS:bool):
        return _Distances(self, initial_coordinates, final_coordinates, WGS)

    def DataLoader(self, Raw_VD:dict, WGS:bool, bbox:list):
        return _DataLoader(self, Raw_VD, WGS, bbox)

    def Analysis(self, VD:dict, SpatioTemporalInfo:dict):
        return _Analysis(self, VD, SpatioTemporalInfo)

    def Visualization(self, VD:dict, SpatioTemporalInfo:dict):
        return _Visualization(self, VD, SpatioTemporalInfo)
