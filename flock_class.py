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
        
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError('The input array must contain only numeric values')
        
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
        
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError('The input array must contain only numeric values')
        
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
    


    def _num_close_non_zero(self, visual_range):
        ''' Compute the number of birds within the visual range for every bird.

        This function computes two arrays, one contains the number of birds within the visual range for every bird and one is the copy of the latter but with no zeros.

        Parameters:
        -----------
        visual_range : float
            Radius of a circle with which a bird can see other birds

        Returns:
        -----------
        num_close : np.ndarray
            Array of number of birds within the visual range for every bird, shape (N_birds)

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
        
        mask = self._visual_range_mask(visual_range)

        num_close = np.count_nonzero(mask, axis=1)
        num_close[num_close == 0] = 1

        return num_close
    


    def _alignment_vector(self, visual_range):
        ''' Compute the direction of the alignment force.

        Parameters:
        -----------
        visual_range : float
            Radius of a circle with which a bird can see other birds

        Returns:
        -----------
        alignment_vector : np.ndarray
            Array of alignment force direction, shape (N_birds, 2)

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
        
        mask = self._visual_range_mask(visual_range)
        num_close_non_zero = self._num_close_non_zero(visual_range)

        aligment_vector = (mask[:, :, None] * self.velocities).sum(axis=1) / num_close_non_zero[:, None]

        return aligment_vector
    



    def _coherence_vector(self, visual_range):
        ''' Compute the direction of the coherence force.

        Parameters:
        -----------
        visual_range : float
            Radius of a circle with which a bird can see other birds

        Returns:
        -----------
        coherence_vector : np.ndarray
            Array of alignment force direction, shape (N_birds, 2)

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
        
        mask = self._visual_range_mask(visual_range)
        num_close_non_zero = self._num_close_non_zero(visual_range)

        coherence_vector = (mask[:, :, None] * self.positions).sum(axis=1) / num_close_non_zero[:, None] - self.positions

        return coherence_vector
    


    def _edge_mask(self, avoid_range):
        ''' Compute the boolean array describing if the edge of the simulation space is within a given range.

        Parameters:
        -----------
        avoid_range : float
            Radius of a circle with which a bird can see simulation barriers

        Returns:
        -----------
        edge_mask : np.ndarray
            Boolean array, shape (N_birds)

        Raises:
        -----------
        TypeError:
            If avoid_range is not an integer or float

        ValueError:
            If avoid_range is lower than 0 or higher than self.space_length
        '''

        if not isinstance(avoid_range, (int, np.integer, float, np.floating)) or isinstance(avoid_range, bool):
            raise TypeError('Avoid range must be a floating number')
        
        if avoid_range < 0 or avoid_range > self.space_length:
            raise ValueError(f'Avoid range must be in range [{0}, {self.space_length}]')
        
        edge_mask = np.any(np.abs(self.positions - self.space_length/2.0) >= (self.space_length/2.0 - avoid_range), axis=1)

        return edge_mask
    


    def _center_direction(self):
        ''' Compute the center direction array.

        Parameters:
        -----------

        Returns:
        -----------
        center_direction : np.ndarray
            Unitary array of the center direction, shape (N_birds, 2)

        Raises:
        -----------
        '''
        
        center = np.array([self.space_length/2, self.space_length/2])
        center_directions = center - self.positions

        center_directions[np.all(center_directions==[0,0], axis=1)] = [1e-5, 1e-5]
        center_directions /= np.linalg.norm(center_directions, axis=1)[:,None]

        return center_directions
    


    def _separation_force(self, separation, visual_range):
        ''' Computes the separation force for every bird.

        Parameters:
        -----------
        separation : float
            Value of the separation parameter

        visual_range : float
            Radius of a circle with which a bird can see other birds

        Returns:
        -----------
        force_separation : np.ndarray
            Array of separation force direction, shape (N_birds, 2)

        Raises:
        -----------
        TypeError:
            If separation is not an integer or float
            If visual_range is not an integer or float

        ValueError:
            If separation is negative
            If visual_range is lower than 0 or higher than self.space_length
        '''

        if not isinstance(separation, (int, np.integer, float, np.floating)) or isinstance(separation, bool):
            raise TypeError('Separation parameter must be a floating number')
        
        if not isinstance(visual_range, (int, np.integer, float, np.floating)) or isinstance(visual_range, bool):
            raise TypeError('Visual range must be a floating number')
        
        if separation < 0:
            raise ValueError('Separation parameter must be >= 0')
        
        if visual_range < 0 or visual_range > self.space_length:
            raise ValueError(f'Visual range must be in range [{0}, {self.space_length}]')
        
        unit_directions = self._directions_unitary_vectors()

        mask = self._visual_range_mask(visual_range)
        
        closest_index = self._closest_index()

        force_separation = - separation * unit_directions[np.arange(unit_directions.shape[0]),closest_index] * mask[np.arange(mask.shape[0]),closest_index][:,None]

        return force_separation
    


    def _alignment_force(self, alignment, visual_range):
        ''' Computes the alignment force for every bird.

        Parameters:
        -----------
        alignment : float
            Value of the alignment parameter

        visual_range : float
            Radius of a circle with which a bird can see other birds

        Returns:
        -----------
        force_alignment : np.ndarray
            Array of alignment force direction, shape (N_birds, 2)

        Raises:
        -----------
        TypeError:
            If alignment is not an integer or float
            If visual_range is not an integer or float

        ValueError:
            If alignment is negative
            If visual_range is lower than 0 or higher than self.space_length
        '''

        if not isinstance(alignment, (int, np.integer, float, np.floating)) or isinstance(alignment, bool):
            raise TypeError('Alignment parameter must be a floating number')
        
        if not isinstance(visual_range, (int, np.integer, float, np.floating)) or isinstance(visual_range, bool):
            raise TypeError('Visual range must be a floating number')
        
        if alignment < 0:
            raise ValueError('Alignment parameter must be >= 0')
        
        if visual_range < 0 or visual_range > self.space_length:
            raise ValueError(f'Visual range must be in range [{0}, {self.space_length}]')
        

        aligment_vector = self._alignment_vector(visual_range)

        aligment_lengths = np.linalg.norm(aligment_vector, axis=1)
        aligment_lengths[aligment_lengths == 0] = 1
        
        force_alignment = alignment * aligment_vector / aligment_lengths[:,None]
    
        return force_alignment
    


    def _coherence_force(self, coherence, visual_range):
        ''' Computes the coherence force for every bird.

        Parameters:
        -----------
        coherence : float
            Value of the coherence parameter

        visual_range : float
            Radius of a circle with which a bird can see other birds

        Returns:
        -----------
        force_coherence : np.ndarray
            Array of coherence force direction, shape (N_birds, 2)

        Raises:
        -----------
        TypeError:
            If coherence is not an integer or float
            If visual_range is not an integer or float

        ValueError:
            If coherence is negative
            If visual_range is lower than 0 or higher than self.space_length
        '''

        if not isinstance(coherence, (int, np.integer, float, np.floating)) or isinstance(coherence, bool):
            raise TypeError('Coherence parameter must be a floating number')
        
        if not isinstance(visual_range, (int, np.integer, float, np.floating)) or isinstance(visual_range, bool):
            raise TypeError('Visual range must be a floating number')
        
        if coherence < 0:
            raise ValueError('Coherence parameter must be >= 0')
        
        if visual_range < 0 or visual_range > self.space_length:
            raise ValueError(f'Visual range must be in range [{0}, {self.space_length}]')
        
        coherence_vector = self._coherence_vector(visual_range)
        
        force_coherence = coherence * coherence_vector / np.linalg.norm(coherence_vector, axis=1)[:,None]

        return force_coherence
    


    def _avoidance_force(self, avoidance, avoid_range):
        ''' Computes the avoidance force for every bird.

        Parameters:
        -----------
        avoidance : float
            Value of the avoidance parameter

        avoid_range : float
            Radius of a circle with which a bird sees the simulation edges

        Returns:
        -----------
        force_avoidance : np.ndarray
            Array of avoidance force direction, shape (N_birds, 2)

        Raises:
        -----------
        TypeError:
            If avoidance is not an integer or float
            If avoid_range is not an integer or float

        ValueError:
            If avoidance is negative
            If avoid_range is lower than 0 or higher than self.space_length
        '''

        if not isinstance(avoidance, (int, np.integer, float, np.floating)) or isinstance(avoidance, bool):
            raise TypeError('Avoidance parameter must be a floating number')
        
        if not isinstance(avoid_range, (int, np.integer, float, np.floating)) or isinstance(avoid_range, bool):
            raise TypeError('Avoid range must be a floating number')
        
        if avoidance < 0:
            raise ValueError('Avoidance parameter must be >= 0')
        
        if avoid_range < 0 or avoid_range > self.space_length:
            raise ValueError(f'Avoid range must be in range [{0}, {self.space_length}]')
        
        edge_mask = self._edge_mask(avoid_range)
        center_directions = self._center_direction()

        force_avoidance = avoidance * center_directions * edge_mask[:,None]

        return force_avoidance
    



    def _compute_forces(self, separation, alignment, coherence, avoidance, visual_range, avoid_range):
        ''' Computes the total 2D force acting on a single bird, for every bird.

        Parameters:
        -----------
        separation : float
            Value of the separation parameter

        alignment : float
            Value of the alignment parameter

        coherence : float
            Value of the coherence parameter

        avoidance : float
            Value of the avoidance parameter

        visual_range : float
            Radius of a circle with which a bird can see other birds

        avoid_range : float
            Radius of a circle with which a bird sees the simulation edges

        Returns:
        -----------
        None

        Raises:
        -----------
        '''
        
        force_separation = self._separation_force(separation, visual_range)
        force_alignment = self._alignment_force(alignment, visual_range)
        force_coherence = self._coherence_force(coherence, visual_range)
        
        force_avoidance = self._avoidance_force(avoidance, avoid_range)
                    
        self.last_forces =  force_coherence + force_avoidance + force_alignment + force_separation