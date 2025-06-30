from .analyser import _Analysis as Analysis
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt


class _Visualization:
 
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

            OD_pairs = Analysis(self.mother, self.VD, self.SpatioTemporalInfo).get_ODPairs()

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

                ax[0].plot([element[-1] for element in self.bbox]+[self.bbox[0][-1]],[element[0] for element in self.bbox] + [self.bbox[0][0]],color='red')
                ax[0].plot([ul_x,self.x_center,lr_x],[ul_y,self.y_center,lr_y],color='red')
                ax[0].plot([ll_x,self.x_center,ur_x],[ll_y,self.y_center,ur_y],color='red')


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
            y = [value for set in Analysis(self.mother,self.VD,self.SpatioTemporalInfo).get_DistanceTravelled() for value in set]
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
            ax.plot(self.t[self.id.index(id)],Analysis().get_DistanceTravelled()[self.id.index(id)],color='k',label=f'Vehicle ID: {id}')

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
            
            a = Analysis(self.mother, self.VD, self.SpatioTemporalInfo).get_Acceleration()

            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(15,4))
            ax.plot(self.t[self.id.index(id)],a[self.id.index(id)],color='red',label=f'Vehicle ID: {id}')

            plt.title('Acceleration vs Time')
            plt.xlabel('t (s)')
            plt.ylabel(r'Acceleration $(m/s^2)$')
            plt.tight_layout()
            plt.grid(True)
            plt.show(close=True)