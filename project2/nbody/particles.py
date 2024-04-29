import numpy as np
import matplotlib.pyplot as plt
import os

class Particles:
    """
    Particle class to store particle properties
    """
    
    def __init__(self,N:int=100):
        self.nparticles=N
        self._masses=np.ones((N,1))
        self._positions=np.zeros((N,3))
        self._velocities=np.zeros((N,3))
        self._accelerations=np.zeros((N,3))
        self._tags=np.arange(N)
        self.time=0
        return
    
    @property
    def masses(self):
        return self._masses
    
    @masses.setter
    def masses(self,masses):
        if len(masses)!=self.nparticles:
            print("Number of particles does not match!")
            raise ValueError
        self._masses=masses
        return
     
    @property
    def positions(self):
        return self._positions
    
    @positions.setter
    def positions(self,positions):
        if len(positions)!=self.nparticles:
            print("Number of particles does not match!")
            raise ValueError
        self._positions=positions
        return
    
    @property
    def velocities(self):
        return self._velocities
    
    @velocities.setter
    def velocities(self,velocities):
        if len(velocities)!=self.nparticles:
            print("Number of particles does not match!")
            raise ValueError
        self._velocities=velocities
        return
     
    @property
    def accelerations(self):
        return self._accelerations
    
    @accelerations.setter
    def accelerations(self,accelerations):
        if len(accelerations)!=self.nparticles:
            print("Number of particles does not match!")
            raise ValueError
        self._accelerations=accelerations
        return
    
    @property
    def tags(self):
        return self._tags
    
    @tags.setter
    def tags(self,tags):
        if len(tags)!=self.nparticles:
            print("Number of particles does not match!")
            raise ValueError
        self._tags=tags
        return
    

    def set_particles(self,pos,vel,acc):
        self.positions=pos
        self.velocities=vel
        self.accelerations=acc
        return
    
    def add_particles(self,add_particles,mass,pos,vel,acc):
        self.nparticles=self.nparticles+add_particles
        self.masses=np.vstack((self.masses,mass))
        self.positions=np.vstack((self.positions,pos))
        self.velocities=np.vstack((self.velocities,vel))
        self.accelerations=np.vstack((self.accelerations,acc))
        self.tags=np.arange(self.nparticles)
        return
        
    
    def output(self,filename):
        masses=self.masses
        pos=self.positions
        vel=self.velocities
        acc=self.accelerations
        tags=self.tags
        
        header = 'tag           mass      x position   y position   z position   x velocity   y velocity   z velocity x acceleration y acceleration z acceleration'

        fmt = '%-12s %-12.8f %-12.8f %-12.8f %-12.8f %-12.8f %-12.8f %-12.8f %-12.8f %-12.8f %-12.8f'

        np.savetxt(filename,np.c_[tags[:],masses[:,0]
                                  ,pos[:,0],pos[:,1],pos[:,2]
                                  ,vel[:,0],vel[:,1],vel[:,2]
                                  ,acc[:,0],acc[:,1],acc[:,2]],fmt=fmt,header=header)
        return
    
    def draw(self,dim=2):
        if dim==2:
            plt.scatter(self.positions[:,0],self.positions[:,1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Random Position')
            plt.grid(True)
            plt.show()
            
        elif dim==3:
            fig = plt.figure()
            plt.scatter(self.positions[:,0],self.positions[:,1],self.positions[:,2])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Random Position')
            plt.grid(True)
            plt.show()  
        else:
            print('Cannot draw.')  
        return
    
    