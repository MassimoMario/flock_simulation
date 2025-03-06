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

        self.positions = np.random.uniform(0, self.space_length, (self.N_birds, 2))
        self.velocities = np.random.randn(self.N_birds, 2) * np.sqrt(self.max_speed/2) 
        self.last_forces = np.empty((N_birds, 2))


    
    def init_given_positions(self, array):
        ''' Initialize birds positions with a given 2D array.

        This function initializes the positions attribute with a given array with shape (N_birds, 2).

        Parameters:
        -----------
        array : np.ndarray
            2D numpy array with coordinates of every bird initial positions

        Returns:
        -----------
        None

        Raises:
        -----------
        TypeError:
            If the input array is not a np.ndarray

        ValueError:
            If the input array shape is not correct
            If array values are outside the space_length range
        '''

        if not isinstance(array, (np.ndarray)):
            raise TypeError('The input array must be a np.ndarray')
        
        if np.shape(array) != (self.N_birds, 2):
            raise ValueError(f'The input array must have shape ({self.N_birds},2)')
        
        if not np.all((array >= 0) & (array <= self.space_length)):
            raise ValueError(f'Every value of the array must be >= 0 and <= {self.space_length}')
        
        self.positions = array



    def init_given_velocities(self, array):
        ''' Initialize birds velocities with a given 2D array.

        This function initializes the velocities attribute with a given array with shape (N_birds, 2).

        Parameters:
        -----------
        array : np.ndarray
            2D numpy array with velocity components of every bird initial condition

        Returns:
        -----------
        None

        Raises:
        -----------
        TypeError:
            If the input array is not a np.ndarray

        ValueError:
            If the input array shape is not correct
            If the array values exceed the maximum speed
        '''

        if not isinstance(array, (np.ndarray)):
            raise TypeError('The input array must be a np.ndarray')
        
        if np.shape(array) != (self.N_birds, 2):
            raise ValueError(f'The input array must have shape ({self.N_birds},2)')
        
        if not np.all((array < self.max_speed) & (array > - self.max_speed)):
            raise ValueError(f'Every value of the array must be < {self.max_speed} and > -{self.max_speed} (a maximum speed is needed for a good simulation behaviour)')
        
        self.velocities = array


    
    def _directions_between_birds(self):
        ''' Compute directions between any couple of birds.

        This function computes the array direction between any couple of birds as the difference between their positions.

        Parameters:
        -----------

        Returns:
        -----------
        directions : np.ndarray
            Array of directions between any couple of birds, has shape (N_birds, N_birds, 2)

        Raises:
        -----------
        '''
        directions = self.positions - self.positions[:, None]

        return directions
    


    def _distances_between_birds(self):
        ''' Compute distances between any couple of birds.

        This function computes the scalar distance between any couple of birds as the norm of the direction between them.

        Parameters:
        -----------

        Returns:
        -----------
        distances : np.ndarray
            Array of distances between any couple of birds, has shape (N_birds, N_birds)

        Raises:
        -----------
        '''

        directions = self._directions_between_birds()

        distances = np.linalg.norm(directions, axis=2)
        distances[distances == 0] = np.inf

        return distances
    


    def _directions_unitary_vectors(self):
        ''' Compute unitary direction arrays between any couple of birds.

        This function computes the unitary direction arrays between any couple of birds as the ratio between directions and distances between them.

        Parameters:
        -----------

        Returns:
        -----------
        unit_directions : np.ndarray
            Array of unitary direction between any couple of birds, has shape (N_birds, N_birds, 2)

        Raises:
        -----------
        '''

        directions = self._directions_between_birds()
        distances = self._distances_between_birds()

        unit_directions = directions / distances[:,:,None]

        return unit_directions
    


    def _visual_range_mask(self, visual_range):
        ''' Compute mask of boolean values if another bird is near considering a visual range.

        This function computes the boolean mask of having another bird near within a visual range.

        Parameters:
        -----------
        visual_range : float
            Radius of a circle with which a bird can see other birds
        
        Returns:
        -----------
        mask : np.ndarray
            Boolean np.ndarray that is True when another bird is within the visual range, has shape (N_birds, N_birds)

        Raises:
        -----------
        TypeError:
            If visual_range is not an integer or float

        ValueError:
            If visual_range is lower than 0 or higher than self.space_length
        '''

        if not isinstance(visual_range, (int, np.integer, float, np.floating)) or isinstance(visual_range, bool):
            raise TypeError('Visual range must be a floating number')
        
        if visual_range < 0 or visual_range > self.space_length:
            raise ValueError(f'Visual range must be in range [{0}, {self.space_length}]')
        

        distances = self._distances_between_birds()

        mask = distances < visual_range

        return mask
    


    def _closest_index(self):
        ''' Compute the indices of closest bird for each bird of the flock.

        This function computes the array containing indices of the closest bird for each bird of the flock.

        Parameters:
        -----------
        
        Returns:
        -----------
        closest_index : np.ndarray
            np.ndarray which value are the index of the closest bird, has shape (N_birds)

        Raises:
        -----------
        '''

        distances = self._distances_between_birds()

        closest_index = np.argmin(distances, axis=1)

        return closest_index