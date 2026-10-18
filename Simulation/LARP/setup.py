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
                 initial_volume_fraction: float,
                 body_force: np.ndarray = np.zeros((3,3,2)) # TEMPORARY
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
        self.body_force = body_force

        self.dx = x_length/x_partitions
        self.dy = y_length/y_partitions

        cells = [ 
            [FluidCell(velocity=initial_velocity.copy(),pressure=initial_pressure,volume_fraction=initial_volume_fraction)
             for _ in range(self.y_partitions)]
                for _ in range(self.x_partitions)]
        self.cells = np.array(cells)
        pass

    def get_velocities(self, 
                       axis: int = -1) -> np.ndarray:
        if axis == -1:
            return np.array([[cell.velocity for cell in row] for row in self.cells])
        else:
            return np.array([[cell.velocity[axis] for cell in row] for row in self.cells]) # i think this is correct

    def get_pressures(self) -> np.ndarray:
        return np.array([[cell.pressure for cell in row] for row in self.cells])

    def get_volume_fractions(self) -> np.ndarray:
        return np.array([[cell.volume_fraction for cell in row] for row in self.cells])

    
    def solve_poisson(
        self,
        u_star: np.ndarray,
        v_star: np.ndarray,
        pressure_error_tolerance: float = 1e-2
    ) -> None:

        pass

    def simulate(self,
                 pressure_error_tolerance: float) -> None:
        
        t=0
        u = self.get_velocities(axis=0) # pretty sure this completely defeats the point of having a class, but whatever 
        v = self.get_velocities(axis=1) # UPDATE: I tried without ^^^ and it sucks
        p = self.get_pressures()
        body_force_x = self.body_force[1:-1, 1:-1, 0]
        body_force_y = self.body_force[1:-1, 1:-1, 1]
        # c = self.get_volume_fractions()  

        def boundary():
            # Cheap way of avoiding the setup, in the future, set the boundary conditions on self.cells before simulating (using different cell types)
            u[:,0] = 0.0 # Bottom wall
            u[:,-1] = +1 # Top wall
            u[0,:] = 0.0 # Left wall
            u[-1,:] = 0.0 # Right wall

            v[:,0] = 0.0 # Bottom wall
            v[:,-1] = 0.0 # Top wall
            v[0,:] = 0.0 # Left wall
            v[-1,:] = 0.0 # Right wall

        boundary()

        while t <= self.total_time:

            # 1. Calculate advection terms
            u_dudx = u[1:-1,1:-1] * ( u[2:,1:-1] - u[:-2,1:-1] ) / (2*self.dx)
            v_dudy = v[1:-1,1:-1] * ( u[1:-1,2:] - u[1:-1,:-2] ) / (2*self.dy)
            u_dvdx = u[1:-1,1:-1] * ( v[2:,1:-1] - v[:-2,1:-1] ) / (2*self.dx)
            v_dvdy = v[1:-1,1:-1] * ( v[1:-1,2:] - v[1:-1,:-2] ) / (2*self.dy)

            # 2. Calculate diffusion terms
            d2u_dx2 = ( u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1] ) / (self.dx**2)
            d2u_dy2 = ( u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2] ) / (self.dy**2)
            d2v_dx2 = ( v[2:,1:-1] - 2*v[1:-1,1:-1] + v[:-2,1:-1] ) / (self.dx**2)
            d2v_dy2 = ( v[1:-1,2:] - 2*v[1:-1,1:-1] + v[1:-1,:-2] ) / (self.dy**2)

            nu_lap_u = self.viscosity * ( d2u_dx2 + d2u_dy2 )
            nu_lap_v = self.viscosity * ( d2v_dx2 + d2v_dy2 )

            # 3. Calculate intermediate velocities
            u_star = u[1:-1,1:-1] - self.dt * ( u_dudx + v_dudy - nu_lap_u - body_force_x )
            v_star = v[1:-1,1:-1] - self.dt * ( u_dvdx + v_dvdy - nu_lap_v - body_force_y )

            # 4. Solve Poisson pressure equation
            self.solve_poisson(u_star=u_star, v_star=v_star, pressure_error_tolerance=pressure_error_tolerance)

            # 5. Calculate pressure gradients
            dp_dx = ( p[2:,1:-1] - p[:-2,1:-1]) / (2*self.dx)
            dp_dy = ( p[1:-1,2:] - p[1:-1,:-2]) / (2*self.dy)

            # 6. Calculate new velocities
            u_new = u_star - (self.dt / self.density) * dp_dx
            v_new = v_star - (self.dt / self.density) * dp_dy

            # 7. CFL check
            force_time = suggest_timestep(velocity=np.ndarray([u,v]),dx=self.dx,dy=self.dy,viscosity=self.viscosity)
            if self.dt > force_time:
                self.dt = force_time
                continue

            # 8. Save u,v,p,c NOTE: p already saved in 4.
            u[1:-1, 1:-1] = u_new
            v[1:-1, 1:-1] = v_new

            ''' Uncomment below section if cells object is needed for frontend'''
            # for i in range(self.x_partitions): 
            #     for j in range(self.y_partitions):
            #         self.cells[i,j].velocity[0] = u_new[i-1,j-1]
            #         self.cells[i,j].velocity[1] = v_new[i-1,j-1]
            #         self.cells[i,j].pressure = p[i,j]

            t += self.dt
        print("done")
        pass
    
class FluidCell:
    def __init__(self,
                 velocity: np.ndarray,
                 pressure: float,
                 volume_fraction: float) -> None:
        
        self.velocity = velocity
        self.pressure = pressure
        self.volume_fraction = volume_fraction
        pass

class WallCell:
    def __init__(self,
                 velocity: np.ndarray) -> None:

        self.velocity = velocity
        pass

class FreeCell:
    def __init__(self,
                 pressure: float,
                 velocity: np.ndarray) -> None:

        self.velocity = velocity
        self.pressure = pressure
        pass

class AirCell: # Currently no pressure or velocity, but may be added in future
    def __init__(self,
                 volume_fraction: float) -> None:
        
        self.volume_fraction = volume_fraction
        pass

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
