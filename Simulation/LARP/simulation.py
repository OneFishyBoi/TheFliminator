# import math
import numpy as np

from setup import *
#----------------------
density = 1e3
viscosity = 1e-6
Lx = Ly = 1
Nx = Ny = 3
dt = 69
total_time = 1

'''NOTE: Below is temporary, in future we need an initial velocity/pressure/vf function'''
init_velocity = np.zeros(2) + 0.1
init_pressure = 0
init_volume_fraction = 1
#----------------------
simulation = Grid(density=density, viscosity=viscosity,
                  x_length=Lx, y_length=Ly,
                  x_partitions=Nx, y_partitions=Ny,
                  dt=dt, total_time=total_time,
                  initial_velocity=init_velocity,
                  initial_pressure=init_pressure, 
                  initial_volume_fraction=init_volume_fraction)

dx = Lx/Nx
dy = Ly/Ny

velocities = simulation.get_velocities()
pressures = simulation.get_pressures()
volume_fractions = simulation.get_volume_fractions()

simulation.dt = suggest_timestep(velocity=velocities,dx=dx,dy=dy,viscosity=viscosity)

print(f"\nSuggested dt = {simulation.dt}")
print(f"Velocity: {velocities[1,2]}")
print(f"Pressure: {pressures[0,1]}")
print(f"Volume Fraction: {volume_fractions[0,2]}")
print(f"dt: {simulation.dt}")

