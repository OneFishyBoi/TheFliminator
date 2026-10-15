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

        dx = self.dx
        dy = self.dy
        dt = self.dt
        rho = self.density

        # Right-hand side: rho/dt * divergence(u_star)
        b = np.zeros_like(u_star)

        b[1:-1, 1:-1] = rho / dt * (
            (u_star[2:, 1:-1] - u_star[:-2, 1:-1]) / (2 * dx)
            + (v_star[1:-1, 2:] - v_star[1:-1, :-2]) / (2 * dy)
        )

        # Initial pressure estimate
        p = np.zeros_like(u_star)

        # Jacobi iteration
        while True:
            p_new = p.copy()

            p_new[1:-1, 1:-1] = (
                (p[2:, 1:-1] + p[:-2, 1:-1]) * dy**2
                + (p[1:-1, 2:] + p[1:-1, :-2]) * dx**2
                - b[1:-1, 1:-1] * dx**2 * dy**2
            ) / (2 * (dx**2 + dy**2))

            # Zero-gradient (Neumann) pressure boundary conditions
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]

            # Fix the arbitrary pressure reference
            p_new -= p_new[0, 0]

            # Check convergence
            error = np.max(np.abs(p_new - p))
            p = p_new

            if error < pressure_error_tolerance:
                break

        # Store the solution in your FluidCell objects
        for i in range(self.x_partitions):
            for j in range(self.y_partitions):
                self.cells[i, j].pressure = p[i, j]


    def simulate(self,
                 pressure_error_tolerance: float) -> None:
        
        t=0
        u = self.get_velocities(axis=0) # pretty sure this completely defeats the point of having a class, but whatever
        v = self.get_velocities(axis=1)
        p = self.get_pressures()
        u_star = np.zeros((self.x_partitions, self.y_partitions))
        v_star = np.zeros((self.x_partitions, self.y_partitions))
        u_new = np.zeros((self.x_partitions, self.y_partitions))
        v_new = np.zeros((self.x_partitions, self.y_partitions))

        while t <= self.total_time:
            for i in range(self.x_partitions):
                for j in range(self.y_partitions):
                
                    # 1. Calculate advection terms
                    u_dudx = u[i,j] * central_derivative(field=u, dg=self.dx, axis=0, i=i, j=j)
                    v_dudy = v[i,j] * central_derivative(field=u, dg=self.dy, axis=1, i=i, j=j)
                    u_dvdx = u[i,j] * central_derivative(field=v, dg=self.dx, axis=0, i=i, j=j)
                    v_dvdy = v[i,j] * central_derivative(field=v, dg=self.dy, axis=1, i=i, j=j)

                    # 2. Calculate diffusion terms
                    d2u_dx2 = second_central_derivative(field=u, dg=self.dx, axis=0, i=i, j=j)
                    d2u_dy2 = second_central_derivative(field=u, dg=self.dy, axis=1, i=i, j=j)
                    d2v_dx2 = second_central_derivative(field=v, dg=self.dx, axis=0, i=i, j=j)
                    d2v_dy2 = second_central_derivative(field=v, dg=self.dy, axis=1, i=i, j=j)
                    nu_lap_u = self.viscosity * ( d2u_dx2 + d2u_dy2 )
                    nu_lap_v = self.viscosity * ( d2v_dx2 + d2v_dy2 )

                    # 3. Calculate intermediate velocities
                    u_star[i,j] = u[i,j] - self.dt*( u_dudx + v_dudy - nu_lap_u - self.body_force[i,j] )
                    v_star[i,j] = v[i,j] - self.dt*( u_dvdx + v_dvdy - nu_lap_v - self.body_force[i,j] )

            # 4. Solve Poisson pressure equation
            self.solve_poisson(u_star=u_star, v_star=v_star, pressure_error_tolerance=pressure_error_tolerance)

            for i in range(self.x_partitions):
                for j in range(self.y_partitions):
                    # 5. Calculate pressure gradients
                    dp_dx = central_derivative(field=p, dg=self.dx, axis=0, i=i, j=j)
                    dp_dy = central_derivative(field=p, dg=self.dy, axis=1, i=i, j=j)

                    # 6. Calculate new velocities
                    u_new[i,j] = u_star[i,j] - (self.dt / self.density) * dp_dx
                    v_new[i,j] = v_star[i,j] - (self.dt / self.density) * dp_dy

            # 7. Apply boundary conditions

            # 8. CFL check

                    # 9. Save u,v,p,c
                    self.cells[i, j] = FluidCell(velocity=np.array([u_new[i, j], v_new[i, j]]), pressure=p[i, j], volume_fraction=1.0)


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


def central_derivative(field: np.ndarray,
                       dg: float,
                       axis: int,
                       i: int,
                       j: int) -> float:

    dfdg = 0.0
    if axis == 0:
        dfdg = (field[i + 1, j] - field[i - 1, j]) / (2 * dg)
    elif axis == 1:
        dfdg = (field[i, j + 1] - field[i, j - 1]) / (2 * dg)
    return dfdg


def forward_derivative(field: np.ndarray,
                      dg: float,
                      axis: int,
                      i: int,
                      j: int) -> float:

    dfdg = 0.0
    if axis == 0:
        dfdg = (field[i, j] - field[i - 1, j]) / dg
    elif axis == 1:
        dfdg = (field[i, j] - field[i, j - 1]) / dg
    return dfdg


def second_central_derivative(field: np.ndarray,
                             dg: float,
                             axis: int,
                             i: int,
                             j: int) -> float:

    dfdg = 0.0
    if axis == 0:
        dfdg = (field[i + 1, j] - 2 * field[i, j] + field[i - 1, j]) / (dg ** 2)
    elif axis == 1:
        dfdg = (field[i, j + 1] - 2 * field[i, j] + field[i, j - 1]) / (dg ** 2)
    return dfdg