import numpy as np


class Flock:

    def __init__(self, N_birds = 200, space_length = 100, seed = 1999):
        ''' Class constructor.

        This function inizializes an object of Flock class, with the following attributes: number of birds, space length, 
        random seed, max_speed, positions, velocities and last forces array.

        Parameters:
        -----------
        N_birds : int, optional
            Number of birds of the simulation
        
        space_length : float, optional
            Length of the side of the square containing the birds

        seed : int, optional
            Integer initializing the random seed for reproducibility

        
        Attributes of a Flock object:
        -----------
        N_birds : int
            Number of birds of the simulation
        
        space_length : float
            Length of the side of the square containing the birds

        seed : int
            Integer initializing the random seed for reproducibility
        
        max_speed : float
            maximum speed allowed in the simulation

        positions : numpy.ndarray
            Array of birds positions with shape (N_birds,2)

        velocities : numpy.ndarray
            Array of birds velocities with shape (N_birds,2)

        last_forces : numpy.ndarray
            Array of the last forces computed to update the simulation

        Raises:
        -----------
        TypeError
            If "N_birds" or "seed" are not integers
            If "space_length" is not an integer or float

        ValueError
            If "N_birds" or "space_length" are <= 0
            If "seed" is not in the allowed range [0, 2**32-1]
        '''

        if not isinstance(N_birds, (int, np.integer)) or isinstance(N_birds, bool):
            raise TypeError('Number of birds must be an integer number')
        
        if not isinstance(space_length, (int, float, np.integer, np.floating)) or isinstance(space_length, bool):
            raise TypeError('Space length must be a floating number')
        
        
        if N_birds <= 0:
            raise ValueError('Number of birds must be > 0')
        
        if space_length <= 0:
            raise ValueError('Space length must be > 0')
        
        

        self.N_birds = N_birds
        self.space_length = space_length

        np.random.seed(seed)

        self.max_speed = space_length/9

        self.positions = np.zeros((N_birds, 2))
        self.velocities = np.zeros((N_birds, 2)) 
        self.last_forces = np.zeros_like(self.velocities)