import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML



def animate(boid_positions_per_time_step, boid_velocities_per_time_step, space_length, num_time_steps, save = False):
        """ Animates the simulation creating a video/GIF

        Parameters:
        -----------
            boid_positions_per_time_step : np.ndarray
                Birds positions per each time step

            boid_velocities_per_time_step : np.ndarray
                Birds positions per each time step

            space_length : float
                Length of the side of the square containing the birds

            num_time_steps : int
                Total number of time steps

            save : bool, optional
                Bool variable to save or not the gif produce, default is False

        Returns:
        -----------
            None
        """

        fig, ax = plt.subplots(figsize=(7,7))

        velocities_magnitudes = np.linalg.norm(boid_velocities_per_time_step[0], axis=1)
        velocities_normalized = boid_velocities_per_time_step[0] / np.reshape(velocities_magnitudes, (-1,1))
        
        scat = ax.quiver(boid_positions_per_time_step[0][:,0], 
                        boid_positions_per_time_step[0][:,1],
                        velocities_normalized[:,0],
                        velocities_normalized[:,1], scale=14, scale_units='inches')
        
        ax.set_xlim([0,space_length])
        ax.set_ylim([0,space_length])

        def update(frame):
            scat.set_offsets(boid_positions_per_time_step[frame])

            velocities_magnitudes = np.linalg.norm(boid_velocities_per_time_step[frame], axis=1)
            velocities_normalized = boid_velocities_per_time_step[frame]/ np.reshape(velocities_magnitudes, (-1,1))
            scat.set_UVC(velocities_normalized[:,0], 
                        velocities_normalized[:,1])
        

            return scat,

        ani = FuncAnimation(fig, update, frames=num_time_steps, blit=True)
        ax.axis('off')
        print("Animation finished. Video processing . . .")
        display(HTML(ani.to_jshtml()))

        if save:
              ani.save('flock_simulation.gif', writer="pillow", fps=60)