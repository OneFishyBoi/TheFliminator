# import math
import numpy as np

class Grid:
    def __init__(self,
                 density: float,
                 viscosity: float,
                 x_length: float,
                 y_length: float,
                 x_partitions: int,
                 y_partitions: int,
                 dt: float,
                 total_time: float,
                 initial_velocity: np.ndarray,
                 initial_pressure: float,
                 initial_volume_fraction: float
                ) -> None:
        
        self.density = density
        self.viscosity = viscosity
        self.x_length = x_length
        self.y_length = y_length
        self.x_partitions = x_partitions
        self.y_partitions = y_partitions
        self.dt = dt
        self.total_time = total_time
        self.initial_velocity = initial_velocity
        self.initial_pressure = initial_pressure
        self.initial_volume_fraction = initial_volume_fraction

        self.dx = x_length/x_partitions
        self.dy = y_length/y_partitions

        cells = [ 
            [FluidCell(velocity=initial_velocity.copy(),pressure=initial_pressure,volume_fraction=initial_volume_fraction)
             for _ in range(self.y_partitions)]
                for _ in range(self.x_partitions)]
        self.cells = np.array(cells)
        pass

    def get_velocities(self) -> np.ndarray:
        return np.array([[cell.velocity for cell in row] for row in self.cells])

    def get_pressures(self) -> np.ndarray:
        return np.array([[cell.pressure for cell in row] for row in self.cells])

    def get_volume_fractions(self) -> np.ndarray:
        return np.array([[cell.volume_fraction for cell in row] for row in self.cells])
    
class FluidCell:
    def __init__(self,
                 velocity: np.ndarray,
                 pressure: float,
                 volume_fraction: float) -> None:
        
        self.velocity = velocity
        self.pressure = pressure
        self.volume_fraction = volume_fraction
        pass

# class BoundaryCell:
#     def __init__(self,
#                  velocity: np.ndarray) -> None:

#         self.velocity = velocity
#         pass

def suggest_timestep(velocity: np.ndarray,
                    dx: float,
                    dy: float,
                    viscosity: float,
                    CFL: float = 0.5) -> float:

    max_vel = np.abs(velocity).max(axis=(0, 1))

    u_max = max_vel[0]
    v_max = max_vel[1]

    diffusion = 2 * viscosity * (1 / dx**2 + 1 / dy**2)
    advection = u_max / dx + v_max / dy

    if (u_max == 0) and (v_max == 0):
        if diffusion == 0:
            recommendation = float("inf")
        else:
            recommendation = CFL / diffusion
    else:
        recommendation = CFL / (advection + diffusion)

    return recommendation

def central_derivative(field,dg,axis,i,j) -> float:

    if axis == 0:   
        dfdg = (field[i+1,j] - field[i-1,j])/(2*dg)
    if axis == 1:
        dfdg = (field[i,j+1] - field[i,j-1])/(2*dg)
    return dfdg

def forward_derivative(field,dg,axis,i,j) -> float:
    
    if axis == 0:   
        dfdg = (field[i,j] - field[i-1,j])/(2*dg)
    if axis == 1:
        dfdg = (field[i,j] - field[i,j-1])/(2*dg)
    return dfdg

def second_central_derivative(field,dg,axis,i,j) -> float:

    if axis == 0:   
        dfdg = (field[i+1,j] - field[i,j] + field[i-1,j])/(dg**2)
    if axis == 1:
        dfdg = (field[i,j+1] - field[i,j] + field[i,j-1])/(dg**2)
    return dfdg