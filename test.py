# -*- coding: utf-8 -*-

import pytest
import numpy as np
from flock_class import Flock

random_seed = 1999
np.random.seed(random_seed)

def test_invalid_type_N_birds_in_constructor():
    """Test that the Flock class constructor raises an error when an invalid type in N_birds argument is provided.

    GIVEN: An invalid type for N_birds in the class constructor

    WHEN: The constructor is called to create a Flock object

    THEN: A TypeError is raised 
    """

    with pytest.raises(TypeError,
                       match = 'Number of birds must be an integer number',
                ): 
                    flock = Flock(N_birds = 'mille', space_length = 100, seed = random_seed)



def test_invalid_value_N_birds_in_constructor():
    """Test that the Flock class constructor raises an error when a negative number of birds is provided.

    GIVEN: A negative number for N_birds in the class constructor

    WHEN: The constructor is called to create a Flock object

    THEN: A ValueError is raised 
    """

    with pytest.raises(ValueError,
                       match = 'Number of birds must be > 0',
                ): 
                    flock = Flock(N_birds = -1, space_length = 100, seed = random_seed)



def test_invalid_type_space_length_in_constructor():
    """Test that the Flock class constructor raises an error when an invalid type in space_length argument is provided.

    GIVEN: An invalid type for space_length in the class constructor

    WHEN: The constructor is called to create a Flock object

    THEN: A TypeError is raised 
    """

    with pytest.raises(TypeError,
                       match = 'Space length must be a floating number',
                ): 
                    flock = Flock(N_birds = 200, space_length = 'mille', seed = random_seed)



def test_invalid_value_space_length_in_constructor():
    """Test that the Flock class constructor raises an error when a negative number of space length is provided.

    GIVEN: A negative number for space_length in the class constructor

    WHEN: The constructor is called to create a Flock object

    THEN: A ValueError is raised 
    """

    with pytest.raises(ValueError,
                       match = 'Space length must be > 0',
                ): 
                    flock = Flock(N_birds = 200, space_length = -1, seed = random_seed)



def test_N_birds_initialized_correctly():
    """Test that the Flock class constructor correctly inizialize N_birds attribute inside the object.

    GIVEN: An acceptable N_birds value in the class constructor

    WHEN: The constructor is called to create a Flock object

    THEN: object.N_birds is equal to the given value
    """

    flock = Flock(N_birds = 234, space_length = 100, seed = random_seed)

    assert np.isclose(flock.N_birds, 234)



def test_space_length_initialized_correctly():
    """Test that the Flock class constructor correctly inizialize space_length attribute inside the object.

    GIVEN: An acceptable space_length value in the class constructor

    WHEN: The constructor is called to create a Flock object

    THEN: object.space_length is equal to the given value
    """

    flock = Flock(N_birds = 200, space_length = 111.1, seed = random_seed)

    assert np.isclose(flock.space_length, 111.1)



def test_init_positions_shape():
    """Test that the initialized positions array of a Flock object has the right shape.

    GIVEN: A Flock object

    WHEN: The init_positions method is called

    THEN: object.positions has the right shape
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    flock.init_positions()

    assert np.shape(flock.positions) == (200,2)


