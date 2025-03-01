# -*- coding: utf-8 -*-

import pytest
import numpy as np
from flock_class import Flock

random_seed = 1999


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



def test_positions_shape_initialized_correctly():
    """Test that the positions attribute of the object has the correct shape after an object is created.

    GIVEN: A Flock object

    WHEN: I access to his attribute positions

    THEN: object.positions has the right shape
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    assert np.shape(flock.positions) == (200,2)



def test_positions_values_initialized_correctly():
    """Test that the positions attribute of the object has every value in the right range.

    GIVEN: A Flock object

    WHEN: I access to his attribute positions

    THEN: Every entry of object.positions array is within the right range
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    assert np.all((flock.positions >= 0) & (flock.positions <= 100))



def test_velocities_shape_initialized_correctly():
    """Test that the velocities attribute of the object has the correct shape after an object is created.

    GIVEN: A Flock object

    WHEN: I access to his attribute velocities

    THEN: object.velocities has the right shape
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    assert np.shape(flock.velocities) == (200,2)



def test_init_given_positions_type_error():
    """Test that the init_given_positions method raises a TypeError when a list is given as input.

    GIVEN: An invalid input type for input_given_positions method

    WHEN: I call input_given_positions method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    wrong_list = [[0,0]]*200

    with pytest.raises(TypeError,
                       match = 'The input array must be a np.ndarray',
                ): 
                    flock.init_given_positions(wrong_list)



def test_init_given_positions_value_error():
    """Test that the init_given_positions method raises a ValueError when an array with invalid shape is given as input.

    GIVEN: An array with invalid shape for input_given_positions method

    WHEN: I call input_given_positions method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    wrong_array = np.zeros((199, 2))

    with pytest.raises(ValueError): 
                    flock.init_given_positions(wrong_array)



def test_init_given_positions_value_error_when_not_in_range():
    """Test that the init_given_positions method raises a ValueError when the input array has values out of the right range.

    GIVEN: An array with invalid values for input_given_positions method

    WHEN: I call input_given_positions method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 3, space_length = 100, seed = random_seed)
    wrong_array = np.array([[1,2], [3,4], [100,-1]])

    with pytest.raises(ValueError): 
                    flock.init_given_positions(wrong_array)



def test_init_given_positions_typical_usage():
    """Test that the init_given_positions input array is equal to the object.positions attribute after calling the method

    GIVEN: A valid array for init_given_positions method

    WHEN: I check object.positions attribute

    THEN: The two arrays are equal
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    input_array = np.zeros((200,2))
    flock.init_given_positions(input_array)

    assert np.allclose(input_array, flock.positions)



def test_init_given_velocities_type_error():
    """Test that the init_given_velocities method raises a TypeError when a list is given as input.

    GIVEN: An invalid input type for input_given_velocities method

    WHEN: I call input_given_velocities method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    wrong_list = [[0,0]]*200

    with pytest.raises(TypeError,
                       match = 'The input array must be a np.ndarray',
                ): 
                    flock.init_given_velocities(wrong_list)



def test_init_given_velocities_value_error():
    """Test that the init_given_velocities method raises a ValueError when an array with invalid shape is given as input.

    GIVEN: An array with invalid shape for input_given_velocities method

    WHEN: I call input_given_velocities method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    wrong_array = np.zeros((199, 2))

    with pytest.raises(ValueError): 
                    flock.init_given_velocities(wrong_array)



def test_init_given_velocities_value_error_when_not_in_range():
    """Test that the init_given_velocities method raises a ValueError when the input array has values out of the right range.

    GIVEN: An array with invalid values for input_given_velocities method

    WHEN: I call input_given_velocities method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 3, space_length = 100, seed = random_seed)
    wrong_array = np.array([[0,0], [0,0], [flock.max_speed+1,0]])

    with pytest.raises(ValueError): 
                    flock.init_given_velocities(wrong_array)



def test_init_given_velocities_typical_usage():
    """Test that the init_given_velocities input array is equal to the object.velocities attribute after calling the method

    GIVEN: A valid array for init_given_velocities method

    WHEN: I check object.velocities attribute

    THEN: The two arrays are equal
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    input_array = np.ones((200,2))
    flock.init_given_velocities(input_array)

    assert np.allclose(input_array, flock.velocities)