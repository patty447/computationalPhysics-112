import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from .particles import Particles
from numba import jit, njit, prange, set_num_threads

"""
The N-Body Simulator class is responsible for simulating the motion of N bodies



"""

class NBodySimulator:

    def __init__(self, particles: Particles):
        
        self.particles=particles
        self.time=particles.time
        self.setup()
        return

    def setup(self, G=1,
                    rsoft=0.01,
                    method="Euler",
                    io_freq=10,
                    io_header="nbody",
                    io_screen=True,
                    visualization=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_header: the output header
        :param io_screen: print message on screen or not.
        :param visualization: on the fly visualization or not. 
        """
        self.G=G
        self.rsoft=rsoft
        self.method=method
        self.io_freq=io_freq
        self.io_header=io_header
        self.io_screen=io_screen
        self.visualization=visualization
        return

    def evolve(self, dt:float, tmax:float):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """

        self.dt=dt
        self.tmax=tmax
        
        # setup which method
        method=self.method
        if method=="Euler":
            _advance_particles=self._advance_particles_Euler
        elif method=="RK2":
            _advance_particles=self._advance_particles_RK2
        elif method=="RK4":
            _advance_particles=self._advance_particles_RK4
        else:
            print("Error: mysolve doesn't supput the method",method)
            quit()

        time=self.time
        particles=self.particles
        header=self.io_header
        # setup the steps of the particles
        nsteps=int(np.ceil(tmax/dt))
        
        # create a folder for output data
        # folder='data_'+str(self.io_header)
        # folder_path='./'+str(folder)
        # os.makedirs(folder_path,exist_ok=True)
        
        io_folder = "data_"+self.io_header
        Path(io_folder).mkdir(parents=True, exist_ok=True)

        
        
        for n in range(nsteps):
            if time+dt>tmax:
                dt=tmax-time

            particles=_advance_particles(dt,particles)
            
            if n%self.io_freq==0:
                print('step=',n,'time=',time)
            # filename=header+'_'+str(n).zfill(6)
            # file_path = os.path.join(folder_path, filename)
                fn = self.io_header+"_"+str(n).zfill(6)+".dat"
                fn = io_folder+"/"+fn
                particles.output(fn)  
                                  
            time=time+dt
            
        print("Simulation is done!")
        return

    def _calculate_acceleration(self, nparticles, masses, positions):
        """
        Calculate the acceleration of the particles
        """
        accelerations = np.zeros_like(positions)
        
        G=self.G
        rsoft=self.rsoft
        accelerations=_calculate_acceleration_kernel(nparticles,masses,positions,accelerations,G,rsoft)

        return accelerations
        
    def _advance_particles_Euler(self, dt, particles):
        # setup the parameters
        nparticles=particles.nparticles
        mass=particles.masses
        pos=particles.positions
        vel=particles.velocities
        acc=self._calculate_acceleration(nparticles,mass,pos)
        
        # Euler method calculation
        pos=pos+dt*vel
        vel=vel+dt*acc
        
        # update particles
        particles.set_particles(pos,vel,acc)   
        return particles

    def _advance_particles_RK2(self, dt, particles):

        nparticles=particles.nparticles
        mass=particles.masses
        pos=particles.positions
        vel=particles.velocities
        acc=self._calculate_acceleration(nparticles,mass,pos)
        
        # RK2 method calculation
        pos1=pos+dt *vel
        vel1=vel+dt *pos
        acc1=self._calculate_acceleration(nparticles,mass,pos1)
        
        pos2=pos1+dt *vel1
        vel2=vel1+dt *acc1
        acc2=self._calculate_acceleration(nparticles,mass,pos2)
        
        pos=pos+dt/2 *(vel1+vel2)
        vel=vel+dt/2 *(acc1+acc2)
        acc=self._calculate_acceleration(nparticles,mass,pos)

        particles.set_particles(pos,vel,acc) 
        return particles

    def _advance_particles_RK4(self, dt, particles):
        
        nparticles=particles.nparticles
        mass=particles.masses
        pos=particles.positions
        vel=particles.velocities
        acc=self._calculate_acceleration(nparticles,mass,pos)
        
        # RK4 method calculation
        pos1=pos+dt *vel
        vel1=vel+dt *pos
        acc1=self._calculate_acceleration(nparticles,mass,pos1)
        
        pos2=pos1+dt/2 *vel1
        vel2=vel1+dt/2 *acc1
        acc2=self._calculate_acceleration(nparticles,mass,pos2)
        
        pos3=pos2+dt/2 *vel2
        vel3=vel2+dt/2 *acc2
        acc3=self._calculate_acceleration(nparticles,mass,pos3)
        
        pos4=pos3+dt *vel3
        vel4=vel3+dt *acc3
        acc4=self._calculate_acceleration(nparticles,mass,pos4)
        
        pos=pos+dt/6 *(vel1+2*vel2+2*vel3+vel4)
        vel=vel+dt/6 *(acc1+2*acc2+2*acc3+acc4)
        acc=self._calculate_acceleration(nparticles,mass,pos)

        particles.set_particles(pos,vel,acc) 
        return particles

@njit(parallel=True)
def _calculate_acceleration_kernel(nparticles,masses,positions,accelerations,G,rsoft):
    for i in prange(nparticles):
        for j in prange(nparticles):
            if (j>i): 
                rij = positions[i,:] - positions[j,:]
                r = np.sqrt(np.sum(rij**2) + rsoft**2)
                force = - G * masses[i,0] * masses[j,0] * rij / r**3
                accelerations[i,:] += force[:] / masses[i,0]
                accelerations[j,:] -= force[:] / masses[j,0]

    
    
    
    return accelerations



if __name__ == "__main__":
    
    pass