import numpy as np
import matplotlib.pyplot as plt


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
    
    def output(self,filename):
        """
        Out[ut particle properties to a text file.
        """
        np.savetxt(filename)
        
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
            ax = fig.add_subplot(111, projection='3d')
            plt.scatter(self.positions[:,0],self.positions[:,1],self.positions[:,2])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.title('Random Position')
            plt.grid(True)
            plt.show()  
        else:
            print('Cannot draw.')  
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
    
    
