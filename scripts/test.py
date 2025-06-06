# -*- coding: utf-8 -*-
'''
Author: Mario Massimo
Date: March 2025
'''

import pytest
import numpy as np
import os
from flock_class import Flock

SEED = int(os.getenv("SEED", 1999))

random_seed = SEED

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



def test_flock_object_attributes_initialized_correctly():
    """Test that the Flock class constructor correctly inizialize object attributes.

    GIVEN: Acceptable N_birds and space_length values in the class constructor

    WHEN: The constructor is called to create a Flock object

    THEN: object.N_birds, object.space_length are equal to the given values
    """

    flock = Flock(N_birds = 234, space_length = 111.1, seed = random_seed)

    assert flock.N_birds == 234

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


def test_init_given_positions_type_error_bool():
    """Test that the init_given_positions method raises a TypeError when a np.array with boolean values is given as input.

    GIVEN: An invalid input type for input_given_positions method

    WHEN: I call input_given_positions method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    wrong_array =  np.array([True, False, False])

    with pytest.raises(TypeError,
                       match = 'The input array must contain only numeric values',
                ): 
                    flock.init_given_positions(wrong_array)


def test_init_given_positions_type_error_list():
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



def test_init_given_positions_type_error_string():
    """Test that the init_given_positions method raises a TypeError when a np.array full of strings is given as input.

    GIVEN: An invalid input type for input_given_positions method

    WHEN: I call input_given_positions method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    wrong_array =  np.array(['se telefonando', 'io', 'potessi dirti addio'])

    with pytest.raises(TypeError,
                       match = 'The input array must contain only numeric values',
                ): 
                    flock.init_given_velocities(wrong_array)



def test_init_given_velocities_type_error_list():
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

    with pytest.warns(UserWarning):
                    flock.init_given_velocities(wrong_array)



def test_init_given_velocities_typical_usage():
    """Test that the init_given_velocities input array is equal to the object.velocities attribute after calling the method.

    GIVEN: A valid array for init_given_velocities method

    WHEN: I check object.velocities attribute

    THEN: The two arrays are equal
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    input_array = np.ones((200,2))
    flock.init_given_velocities(input_array)

    assert np.allclose(input_array, flock.velocities)



def test_directions_between_birds_right_shape():
    """Test that the _directions_between_birds method returns an array with the correct shape.

    GIVEN: A Flock object

    WHEN: I call _directions_between_birds method

    THEN: The resulting array has shape (N_birds, N_birds, 2)
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    directions = flock._directions_between_birds()

    assert np.shape(directions) == (200,200,2)



def test_directions_between_birds_single_bird():
    """Test that the _directions_between_birds method computed with only one bird returns an array of zeros.

    GIVEN: A Flock object with a single bird

    WHEN: I call _directions_between_birds method

    THEN: The resulting array is an array of zeros
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)
    directions = flock._directions_between_birds()
    zero_array = np.zeros((1,1,2))

    assert np.allclose(directions, zero_array)



def test_directions_between_birds_collapsed_positions():
    """Test that the _directions_between_birds method computed when every bird is in the same position returns an array of zeros.

    GIVEN: A Flock object with every bird having the same position

    WHEN: I call _directions_between_birds method

    THEN: The resulting array is an array of zeros
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    zero_positions = np.ones((200,2))
    flock.init_given_positions(zero_positions)

    directions = flock._directions_between_birds()
    zero_array = np.zeros((200,200,2))

    assert np.allclose(directions, zero_array)



def test_directions_between_birds_typical_usage():
    """Test that the _directions_between_birds returns the correct array when called on two birds.

    GIVEN: A Flock object with two birds with known positions

    WHEN: I call _directions_between_birds method

    THEN: The resulting array is computed correctly
    """
    
    flock = Flock(N_birds = 2, space_length = 100, seed = random_seed)
    initial_positions = np.array([[1,2],[3,4]])
    flock.init_given_positions(initial_positions)

    directions = flock._directions_between_birds()

    right_directions = np.array([[[ 0,  0],
                                [ 2,  2]],
                                [[-2, -2],
                                [ 0,  0]]])

    assert np.allclose(directions, right_directions)



def test_distances_between_birds_right_shape():
    """Test that the _distances_between_birds method returns an array with the correct shape.

    GIVEN: A Flock object

    WHEN: I call _distances_between_birds method

    THEN: The resulting array has shape (N_birds, N_birds)
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    distances = flock._distances_between_birds()

    assert np.shape(distances) == (200,200)



def test_distances_between_birds_single_bird():
    """Test that the _distances_between_birds method computed with only one bird returns an array of np.inf.

    GIVEN: A Flock object with a single bird

    WHEN: I call _distances_between_birds method

    THEN: The resulting array is an array of np.inf
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)
    distances = flock._distances_between_birds()
    inf_array = np.ones((1,1))*np.inf

    assert np.allclose(distances, inf_array)



def test_distances_between_birds_collapsed_positions():
    """Test that the _distances_between_birds method computed when every bird is in the same position returns an array of np.inf.

    GIVEN: A Flock object with every bird having the same position

    WHEN: I call _distances_between_birds method

    THEN: The resulting array is an array of np.inf
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    zero_positions = np.ones((200,2))
    flock.init_given_positions(zero_positions)

    distances = flock._distances_between_birds()
    inf_array = np.ones((200,200))*np.inf

    assert np.allclose(distances, inf_array)



def test_distances_between_birds_typical_usage():
    """Test that the _distances_between_birds returns the correct array when called on two birds.

    GIVEN: A Flock object with two birds with known positions

    WHEN: I call _distances_between_birds method

    THEN: The resulting array is computed correctly
    """
    
    flock = Flock(N_birds = 2, space_length = 100, seed = random_seed)
    initial_positions = np.array([[1,2],[3,4]])
    flock.init_given_positions(initial_positions)

    distances = flock._distances_between_birds()

    right_distances = np.array([[np.inf, 2*np.sqrt(2)],
                                [2*np.sqrt(2), np.inf]])

    assert np.allclose(distances, right_distances)



def test_directions_unitary_vectors_correct_shape():
    """Test that the _directions_unitary_vectors method returns an array with the correct shape.

    GIVEN: A Flock object

    WHEN: I call _directions_unitary_vectors method

    THEN: The resulting array has shape (N_birds, N_birds, 2)
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    unit_distances = flock._directions_unitary_vectors()

    assert np.shape(unit_distances) == (200,200,2)




def test_directions_unitary_vectors_single_bird():
    """Test that the _directions_unitary_vectors method computed with only one bird returns an array of zeros.

    GIVEN: A Flock object with a single bird

    WHEN: I call _directions_unitary_vectors method

    THEN: The resulting array is an array of zeros
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)
    unit_distances = flock._directions_unitary_vectors()
    zero_array = np.zeros((1,1,2))

    assert np.allclose(unit_distances, zero_array)



def test_directions_unitary_vectors_collapsed_positions():
    """Test that the _directions_unitary_vectors method computed when every bird is in the same position returns an array of zeros.

    GIVEN: A Flock object with every bird having the same position

    WHEN: I call _directions_unitary_vectors method

    THEN: The resulting array is an array of zeros
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    zero_positions = np.ones((200,2))
    flock.init_given_positions(zero_positions)

    unit_directions = flock._directions_unitary_vectors()
    zero_array = np.zeros((200,200,2))

    assert np.allclose(unit_directions, zero_array)



def test_directions_unitary_vectors_typical_usage_off_diagonal():
    """Test that the _directions_unitary_vectors returns an array which rows are normalized to one.

    GIVEN: A Flock object 

    WHEN: I call _directions_unitary_vectors method

    THEN: The resulting array rows are normalized to one
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    unit_distances = flock._directions_unitary_vectors()

    diagonal_mask = np.eye(200, dtype=bool)
    normalized_rows = np.linalg.norm(unit_distances[~diagonal_mask], axis=1)

    correct_normalization = np.ones(200*200-200)
    

    assert np.allclose(normalized_rows, correct_normalization)



def test_directions_unitary_vectors_typical_usage_on_diagonal():
    """Test that the _directions_unitary_vectors returns a matrix which has 0 on the diagonal.

    GIVEN: A Flock object 

    WHEN: I call _directions_unitary_vectors method

    THEN: The resulting matrix has zeros on the diagonal
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    unit_distances = flock._directions_unitary_vectors()

    diagonal_mask = np.eye(200, dtype=bool)
    normalized_rows = np.linalg.norm(unit_distances[diagonal_mask], axis=1)

    correct_normalization = np.zeros(200)
    

    assert np.allclose(normalized_rows, correct_normalization, atol = 1e-3)



def test_directions_unitary_vectors_typical_usage():
    """Test that the _directions_unitary_vectors returns the correct array when called on two birds.

    GIVEN: A Flock object with two birds with known positions

    WHEN: I call _directions_unitary_vectors method

    THEN: The resulting array is computed correctly
    """
    
    flock = Flock(N_birds = 2, space_length = 100, seed = random_seed)
    initial_positions = np.array([[1,2],[3,4]])
    flock.init_given_positions(initial_positions)

    unit_directions = flock._directions_unitary_vectors()

    right_unit_directions = np.array([[[ 0.,  0.],
                                        [ 1/np.sqrt(2), 1/np.sqrt(2)]],
                                        [[-1/np.sqrt(2), -1/np.sqrt(2)],
                                        [ 0., 0.]]])

    assert np.allclose(unit_directions, right_unit_directions)




def test_speed_limit_factors_correct_shape():
    """Test that the _speed_limit_factors method returns an array with the correct shape.

    GIVEN: A Flock object

    WHEN: I call _speed_limit_factors method

    THEN: The resulting array has shape (N_birds)
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    speed_limit_factors = flock._speed_limit_factors()

    assert np.shape(speed_limit_factors) == (200,)




def test_speed_limit_factors_zero_velocities():
    """Test that the _speed_limit_factors method returns an array full one ones if the object.velocities are zeros.

    GIVEN: A Flock object with object.velocities being full of zeros

    WHEN: I call _speed_limit_factors method

    THEN: The resulting array is full of ones
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    zero_array = np.zeros((200,2))
    flock.init_given_velocities(zero_array)

    speed_limit_factors = flock._speed_limit_factors()
    correct_speed_limits = np.ones((200))

    assert np.allclose(speed_limit_factors, correct_speed_limits)




def test_speed_limit_factors_typical_usage():
    """Test that the _speed_limit_factors returns the correct array when called on two birds.

    GIVEN: A Flock object

    WHEN: I call _speed_limit_factors method

    THEN: The resulting array is computed correctly
    """
    
    flock = Flock(N_birds = 2, space_length = 100, seed = random_seed)
    
    speed_limit_factors = flock._speed_limit_factors()

    correct_speed_limit_factors = np.linalg.norm(flock.velocities, axis=1) / flock.max_speed
    correct_speed_limit_factors[correct_speed_limit_factors < 1] = 1

    assert np.allclose(correct_speed_limit_factors, speed_limit_factors)


       
def test_visual_range_mask_typeerror():
    """Test that the _visual_range_mask method raises a TypeError when a string is given as input.

    GIVEN: An invalid input type for _visual_range_mask method

    WHEN: I call _visual_range_mask method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Visual range must be a floating number',
                ): 
                    flock._visual_range_mask(visual_range = 'ventimilioni')



def test_visual_range_mask_valueerror():
    """Test that the _visual_range_mask method raises a ValueError when a negative value is given as input.

    GIVEN: An invalid input value for _visual_range_mask method

    WHEN: I call _visual_range_mask method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError):
                    flock._visual_range_mask(visual_range = -0.4)




def test_visual_range_mask_correct_shape():
    """Test that the _visual_range_mask method returns a np.ndarray with the correct shape.

    GIVEN: A Flock object

    WHEN: I call _visual_range_mask method

    THEN: The resulting array has shape (N_birds, N_birds)
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    mask = flock._visual_range_mask(visual_range = 20)

    assert np.shape(mask) == (200,200)




def test_visual_range_mask_zero_visual_range():
    """Test that the _visual_range_mask method returns a mask full of False when visual_range is 0.

    GIVEN: A Flock object

    WHEN: I call _visual_range_mask method with visual_range = 0

    THEN: The resulting mask is full of False
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    mask = flock._visual_range_mask(visual_range = 0)
    zero_mask = np.zeros((200,200), dtype = bool)

    assert np.allclose(mask, zero_mask)



def test_visual_range_mask_zero_tyipical_usage():
    """Test that the _visual_range_mask method returns a mask with True off the diagonal when birds are near each other.

    GIVEN: A Flock object

    WHEN: I call _visual_range_mask method having two birds near each other

    THEN: The resulting mask has True off the diagonal
    """
    
    flock = Flock(N_birds = 2, space_length = 100, seed = random_seed)
    initial_positions = np.array([[1,2], [3,4]])
    flock.init_given_positions(initial_positions)

    diagonal_mask = np.eye(2, dtype=bool)
    mask = flock._visual_range_mask(visual_range = 50)

    true_array = np.array([True, True])
    

    assert np.allclose(mask[~diagonal_mask], true_array)



def test_closest_index_correct_shape():
    """Test that the _closest_index method returns a np.ndarray with the correct shape.

    GIVEN: A Flock object

    WHEN: I call _closest_index method

    THEN: The resulting array has shape (N_birds)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    closest_index = flock._closest_index()

    assert np.shape(closest_index) == (200,)



def test_closest_index_only_one_bird():
    """Test that the _closest_index method returns a [0] np.ndarray if there is only one bird.

    GIVEN: A Flock object with one bird

    WHEN: I call _closest_index method

    THEN: The resulting array is equal to [0]
    """

    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)
    closest_index = flock._closest_index()
    one_closest = np.array([0])

    assert np.allclose(closest_index, one_closest)



def test_closest_index_typical_usage():
    """Test that the _closest_index method returns the expected np.ndarray given three bird with known positions.

    GIVEN: A Flock object with three birds with known position

    WHEN: I call _closest_index method

    THEN: The resulting array has the correct values
    """

    flock = Flock(N_birds = 3, space_length = 100, seed = random_seed)
    initial_positions = np.array([[0,1], [0,3], [0,10]])
    flock.init_given_positions(initial_positions)

    closest_index = flock._closest_index()

    correct_closest = np.array([1, 0, 1])

    assert np.allclose(closest_index, correct_closest)




def test_num_close_non_zero_correct_shape():
    """Test that the array returned from _num_close_non_zero has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _num_close_non_zero method

    THEN: The returned array has shape (N_birds)
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    num_close_non_zero = flock._num_close_non_zero(visual_range = 20)

    assert np.shape(num_close_non_zero) == (200,)




def test_num_close_non_zero_only_one_bird():
    """Test that the returned array from _num_close_non_zero is [1] if only one bird is present.

    GIVEN: A Flock object with only one bird

    WHEN: I call _num_close_non_zero method

    THEN: The returned array is equal to [1]
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)
    num_close_non_zero = flock._num_close_non_zero(visual_range = 20)
    one_array = np.array([1])

    assert np.allclose(num_close_non_zero, one_array)



def test_num_close_non_zero_zero_visual_range():
    """Test that the _num_close_non_zero method returns an array full of ones when visual_range is 0.

    GIVEN: A Flock object

    WHEN: I call _num_close_non_zero method with visual_range = 0

    THEN: The resulting array is full of ones
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    num_close_non_zero = flock._num_close_non_zero(visual_range = 0)
    one_array = np.ones((200,))

    assert np.allclose(num_close_non_zero, one_array)



def test_num_close_non_zero_typical_usage():
    """Test that the _num_close_non_zero method returns an array as expected given three birds with known positions.

    GIVEN: A Flock object with three birds with known positions

    WHEN: I call _num_close_non_zero method

    THEN: The returned array is equal to the expected one
    """
    
    flock = Flock(N_birds = 3, space_length = 100, seed = random_seed)
    initial_positions = np.array([[1,1],[1,2],[2,2]])
    flock.init_given_positions(initial_positions)

    num_close_non_zero = flock._num_close_non_zero(visual_range = 20)

    expected_array = np.array([2, 2, 2])

    assert np.allclose(num_close_non_zero, expected_array)





def test_alignment_vector_correct_shape():
    """Test that the array returned from _alignment_vector has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _alignment_vector method

    THEN: The returned array has shape (N_birds, 2)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    alignment_vector = flock._alignment_vector(visual_range = 20)

    assert np.shape(alignment_vector) == (200,2)



def test_alignment_vector_only_one_bird():
    """Test that the returned array from _alignment_vector is [[0],[0]] if only one bird is present.

    GIVEN: A Flock object with only one bird

    WHEN: I call _alignment_vector method

    THEN: The returned array is equal to [[0],[0]]
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)
    alignment_vector = flock._alignment_vector(visual_range = 20)
    one_array = np.array([[0],[0]])

    assert np.allclose(alignment_vector, one_array)




def test_alignment_vector_zero_visual_range():
    """Test that the _alignment_vector method returns an array full of ones when visual_range is 0.

    GIVEN: A Flock object

    WHEN: I call _alignment_vector method with visual_range = 0

    THEN: The resulting array is full of zeros
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    alignment_vector = flock._alignment_vector(visual_range = 0)
    zero_array = np.zeros((200,2))

    assert np.allclose(alignment_vector, zero_array)



def test_alignment_vector_typical_usage():
    """Test that the _alignment_vector method returns an array as expected given two birds with known positions and velocities.

    GIVEN: A Flock object with two birds with known positions and velocities

    WHEN: I call _alignment_vector method

    THEN: The returned array is equal to the expected one
    """
    
    flock = Flock(N_birds = 2, space_length = 100, seed = random_seed)
    initial_positions = np.array([[1,1],[1,2]])
    initial_velocities = np.array([[1,1],[1,2]])
    flock.init_given_velocities(initial_velocities)
    flock.init_given_positions(initial_positions)

    alignment_vector = flock._alignment_vector(visual_range = 20)

    expected_array = np.array([[1., 2.],
                            [1., 1.]])

    assert np.allclose(alignment_vector, expected_array)


    

def test_coherence_vector_correct_shape():
    """Test that the array returned from _coherence_vector has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _coherence_vector method

    THEN: The returned array has shape (N_birds, 2)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    coherence_vector = flock._coherence_vector(visual_range = 20)

    assert np.shape(coherence_vector) == (200,2)




def test_coherence_vector_only_one_bird():
    """Test that the returned array from _coherence_vector with only one bird is equal to the opposite of his position.

    GIVEN: A Flock object with only one bird

    WHEN: I call _coherence_vector method

    THEN: The returned array is equal to the opposite of his position
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)
    initial_positions = np.array([[1,1]])
    flock.init_given_positions(initial_positions)

    coherence_vector = flock._coherence_vector(visual_range = 20)
    expected_array = np.array([[-1],[-1]])

    assert np.allclose(coherence_vector, expected_array)




def test_coherence_vector_zero_visual_range():
    """Test that the _coherence_vector method returns the opposite of object.positions when visual_range == 0.

    GIVEN: A Flock object

    WHEN: I call _coherence_vector method with visual_range = 0

    THEN: The resulting array is equal to  - object.positions
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    coherence_vector = flock._coherence_vector(visual_range = 0)
    expected_array = - flock.positions

    assert np.allclose(coherence_vector, expected_array)



def test_coherence_vector_typical_usage():
    """Test that the _coherence_vector method returns an array as expected given two birds with known positions.

    GIVEN: A Flock object with two birds with known positions

    WHEN: I call _coherence_vector method

    THEN: The returned array is equal to the expected one
    """
    
    flock = Flock(N_birds = 2, space_length = 100, seed = random_seed)
    initial_positions = np.array([[1,1],[1,2]])
    flock.init_given_positions(initial_positions)

    coherence_vector = flock._coherence_vector(visual_range = 20)

    expected_array = np.array([[ 0.,  1.],
                            [ 0., -1.]])

    assert np.allclose(coherence_vector, expected_array)



def test_edge_mask_typeerror():
    """Test that the _edge_mask method raises a TypeError when a string is given as input.

    GIVEN: An invalid input type for _edge_mask method

    WHEN: I call _edge_mask method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Avoid range must be a floating number',
                ): 
                    flock._edge_mask(avoid_range = 'ventinove e qualcosina')



def test_edge_mask_valueerror():
    """Test that the _edge_mask method raises a ValueError when a negative value is given as input.

    GIVEN: An invalid input value for _edge_mask method

    WHEN: I call _edge_mask method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError):
                    flock._edge_mask(avoid_range = -1.2)




def test_edge_mask_correct_shape():
    """Test that the array returned from _edge_mask has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _edge_mask method

    THEN: The returned array has shape (N_birds)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    edge_mask = flock._edge_mask(avoid_range = 20)

    assert np.shape(edge_mask) == (200,)




def test_edge_mask_zero_visual_range():
    """Test that the _edge_mask method returns a mask full of False when avoid_range is 0.

    GIVEN: A Flock object

    WHEN: I call _edge_mask method with avoid_range = 0

    THEN: The resulting mask is full of False
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    mask = flock._edge_mask(avoid_range = 0)
    zero_mask = np.zeros((200,), dtype = bool)

    assert np.allclose(mask, zero_mask)



def test_edge_mask_typical_usage():
    """Test that the _edge_mask method returns an array as expected given two birds with known positions.

    GIVEN: A Flock object with two birds with known positions

    WHEN: I call _edge_mask method

    THEN: The returned array is equal to the expected one
    """
    
    flock = Flock(N_birds = 2, space_length = 100, seed = random_seed)
    initial_positions = np.array([[1,1],[50,50]])
    flock.init_given_positions(initial_positions)

    edge_mask = flock._edge_mask(avoid_range = 20)

    expected_array = np.array([ True, False])

    assert np.allclose(edge_mask, expected_array)



def test_center_direction_correct_shape():
    """Test that the array returned from _center_direction has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _center_direction method

    THEN: The returned array has shape (N_birds)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    edge_mask = flock._center_direction()

    assert np.shape(edge_mask) == (200, 2)




def test_center_direction_normalization():
    """Test that the _center_direction returns an array which second dimension is normalized to 1.

    GIVEN: A Flock object 

    WHEN: I call _center_direction method

    THEN: The resulting array second dimension is normalized to one
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    unit_center_distances = flock._center_direction()

    normalized_vector = np.linalg.norm(unit_center_distances, axis=1)
    correct_normalization = np.ones(200)
    

    assert np.allclose(normalized_vector, correct_normalization)




def test_center_direction_typical_usage():
    """Test that the _center_direction returns the expected array when 4 birds with known positions are given.

    GIVEN: A Flock object with 4 birds with known positions

    WHEN: I call _center_direction method

    THEN: The resulting array is equal to the expected one
    """
    
    flock = Flock(N_birds = 4, space_length = 100, seed = random_seed)
    initial_positions = np.array([[0,50],
                                  [50,0],
                                  [100,50],
                                  [10,10]])
    flock.init_given_positions(initial_positions)

    unit_center_distances = flock._center_direction()

    expected_center_distances = np.array([[ 1.,  0.],
                                        [ 0.,  1.],
                                        [-1.,  0.],
                                        [ 1/np.sqrt(2),  1/np.sqrt(2)]])
    
    assert np.allclose(unit_center_distances, expected_center_distances)




def test_separation_force_type_error_separation():
    """Test that the _separation_force method raises an error when a string is given as input for separation argument.

    GIVEN: An invalid input type for separation in _separation_force method

    WHEN: I call _separation_force method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Separation parameter must be a floating number',
                ): 
                    flock._separation_force(separation = 'uno', visual_range = 20)




def test_separation_force_valueerror_separation():
    """Test that the _separation_force method raises a ValueError when a negative value for separation is given as input.

    GIVEN: An invalid input value for separation argument in _separation_force method

    WHEN: I call _separation_force method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError,
                       match = 'Separation parameter must be >= 0'
                       ):
                        flock._separation_force(separation = -1, visual_range = 20)




def test_separation_force_correct_shape():
    """Test that the array returned from _separation_force has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _separation_force method

    THEN: The returned array has shape (N_birds, 2)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    separation_force = flock._separation_force(separation = 1, visual_range = 20)

    assert np.shape(separation_force) == (200, 2)




def test_separation_force_only_one_bird():
    """Test that the returned array from _separation_force method with only one bird is equal to [0].

    GIVEN: A Flock object with only one bird

    WHEN: I call _separation_force method

    THEN: The returned array is equal to [0]
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)

    separation_force = flock._separation_force(separation = 2, visual_range = 20)
    expected_array = np.array([0])

    assert np.allclose(separation_force, expected_array)




def test_separation_force_zero_separation():
    """Test that the returned array from _separation_force method with separation = 0 is full of zeros.

    GIVEN: A Flock object 

    WHEN: I call _separation_force method with separation = 0

    THEN: The returned array is full of zeros
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    separation_force = flock._separation_force(separation = 0, visual_range = 20)
    expected_array = np.zeros((200,2))

    assert np.allclose(separation_force, expected_array)



def test_separation_force_zero_visual_range():
    """Test that the returned array from _separation_force method with visual_range = 0 is full of zeros.

    GIVEN: A Flock object 

    WHEN: I call _separation_force method with visual_range = 0

    THEN: The returned array is full of zeros
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    separation_force = flock._separation_force(separation = 2, visual_range = 0)
    expected_array = np.zeros((200,2))

    assert np.allclose(separation_force, expected_array)



def test_separation_force_typical_usage():
    """Test that the _separation_force returns the expected array.

    GIVEN: A Flock object 

    WHEN: I call _separation_force method

    THEN: The resulting array is equal to the expected one
    """
    
    flock = Flock(N_birds = 4, space_length = 100, seed = random_seed)
    initial_positions = np.array([[0,0],[1,10],[5,5],[10,10]])
    flock.init_given_positions(initial_positions)

    separation_force = flock._separation_force(separation = 2, visual_range = 20)

    expected_separation_force = np.array([[-1.41421356, -1.41421356],
                                        [-1.2493901 ,  1.56173762],
                                        [ 1.2493901 , -1.56173762],
                                        [ 1.41421356,  1.41421356]])
    
    assert np.allclose(separation_force, expected_separation_force)




def test_alignment_force_type_error_separation():
    """Test that the _alignment_force method raises an error when a string is given as input for alignment argument.

    GIVEN: An invalid input type for alignment in _alignment_force method

    WHEN: I call _alignment_force method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Alignment parameter must be a floating number',
                ): 
                    flock._alignment_force(alignment = 'uno', visual_range = 20)





def test_alignment_force_valueerror_separation():
    """Test that the _alignment_force method raises a ValueError when a negative value for alignment is given as input.

    GIVEN: An invalid input value for alignment argument in _alignment_force method

    WHEN: I call _alignment_force method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError,
                       match = 'Alignment parameter must be >= 0'
                       ):
                        flock._alignment_force(alignment = -1, visual_range = 20)




def test_alignment_force_correct_shape():
    """Test that the array returned from _alignment_force has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _alignment_force method

    THEN: The returned array has shape (N_birds, 2)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    alignment_force = flock._alignment_force(alignment = 1, visual_range = 20)

    assert np.shape(alignment_force) == (200, 2)



def test_alignment_force_only_one_bird():
    """Test that the returned array from _alignment_force method with only one bird is equal to [0].

    GIVEN: A Flock object with only one bird

    WHEN: I call _alignment_force method

    THEN: The returned array is equal to [0]
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)

    alignment_force = flock._alignment_force(alignment = 2, visual_range = 20)
    expected_array = np.array([0])

    assert np.allclose(alignment_force, expected_array)




def test_alignment_force_zero_alignment():
    """Test that the returned array from _alignment_force method with alignment = 0 is full of zeros.

    GIVEN: A Flock object 

    WHEN: I call _alignment_force method with alignment = 0

    THEN: The returned array is full of zeros
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    alignment_force = flock._alignment_force(alignment = 0, visual_range = 20)
    expected_array = np.zeros((200,2))

    assert np.allclose(alignment_force, expected_array)




def test_alignment_force_zero_visual_range():
    """Test that the returned array from _alignment_force method with visual_range = 0 is full of zeros.

    GIVEN: A Flock object 

    WHEN: I call _alignment_force method with visual_range = 0

    THEN: The returned array is full of zeros
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    alignment_force = flock._alignment_force(alignment = 2, visual_range = 0)
    expected_array = np.zeros((200,2))

    assert np.allclose(alignment_force, expected_array)



def test_alignment_force_typical_usage():
    """Test that the _alignment_force returns the expected array.

    GIVEN: A Flock object 

    WHEN: I call _alignment_force method

    THEN: The resulting array is equal to the expected one
    """
    
    flock = Flock(N_birds = 4, space_length = 100, seed = random_seed)
    initial_positions = np.array([[0,0],[1,10],[5,5],[10,10]])
    flock.init_given_positions(initial_positions)

    alignment_force = flock._alignment_force(alignment = 2, visual_range = 20)

    expected_force_alignment = np.array([[ 0.23519756,  1.98612238],
                                        [-0.51434264,  1.93273165],
                                        [ 1.34026622,  1.48448188],
                                        [ 0.68160638,  1.88026933]])
    
    assert np.allclose(alignment_force, expected_force_alignment)



def test_coherence_force_type_error_coherence():
    """Test that the _coherence_force method raises an error when a string is given as input for coherence argument.

    GIVEN: An invalid input type for coherence in _coherence_force method

    WHEN: I call _coherence_force method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Coherence parameter must be a floating number',
                ): 
                    flock._coherence_force(coherence = 'ho sceso dandoti il braccio almeno un milione di scale', visual_range = 20)




def test_coherence_force_valueerror_coherence():
    """Test that the _coherence_force method raises a ValueError when a negative value for coherence is given as input.

    GIVEN: An invalid input value for coherence argument in _coherence_force method

    WHEN: I call _coherence_force method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError,
                       match = 'Coherence parameter must be >= 0'
                       ):
                        flock._coherence_force(coherence = -1, visual_range = 20)



def test_coherence_force_correct_shape():
    """Test that the array returned from _coherence_force has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _coherence_force method

    THEN: The returned array has shape (N_birds, 2)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    coherence_force = flock._coherence_force(coherence = 1, visual_range = 20)

    assert np.shape(coherence_force) == (200, 2)




def test_coherence_force_only_one_bird():
    """Test that the returned array from _coherence_force method with only one bird is equal to the expected one.

    GIVEN: A Flock object with only one bird

    WHEN: I call _coherence_force method

    THEN: The returned array is equal to the expected one
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)

    coherence_force = flock._coherence_force(coherence = 2, visual_range = 20)
    expected_array = -2*flock.positions/np.linalg.norm(flock.positions)

    assert np.allclose(coherence_force, expected_array)





def test_coherence_force_zero_separation():
    """Test that the returned array from _coherence_force method with coherence = 0 is full of zeros.

    GIVEN: A Flock object 

    WHEN: I call _coherence_force method with coherence = 0

    THEN: The returned array is full of zeros
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    coherence_force = flock._coherence_force(coherence = 0, visual_range = 20)
    expected_array = np.zeros((200,2))

    assert np.allclose(coherence_force, expected_array)



def test_coherence_force_zero_visual_range():
    """Test that the returned array from _coherence_force method with visual_range = 0 is equal to the expected one.

    GIVEN: A Flock object 

    WHEN: I call _coherence_force method with visual_range = 0

    THEN: The returned array is equal to the expected one
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    coherence_force = flock._coherence_force(coherence = 2, visual_range = 0)
    expected_array = -2*flock.positions / np.linalg.norm(flock.positions, axis=1)[:,None]

    assert np.allclose(coherence_force, expected_array)



def test_coherence_force_typical_usage():
    """Test that the _coherence_force returns the expected array.

    GIVEN: A Flock object 

    WHEN: I call _coherence_force method

    THEN: The resulting array is equal to the expected one
    """
    
    flock = Flock(N_birds = 4, space_length = 100, seed = random_seed)
    initial_positions = np.array([[0,0],[1,10],[5,5],[10,10]])
    flock.init_given_positions(initial_positions)

    coherence_force = flock._coherence_force(coherence = 2, visual_range = 20)

    expected_force_coherence = np.array([[ 1.07810739,  1.6845428 ],
                                        [ 1.2493901 , -1.56173762],
                                        [-1.2493901 ,  1.56173762],
                                        [-1.69599661, -1.05999788]])
    
    assert np.allclose(coherence_force, expected_force_coherence)




def test_avoidance_force_type_error_avoidance():
    """Test that the _avoidance_force method raises an error when a string is given as input for avoidance argument.

    GIVEN: An invalid input type for avoidance in _avoidance_force method

    WHEN: I call _avoidance_force method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Avoidance parameter must be a floating number',
                ): 
                    flock._avoidance_force(avoidance = 'e ora che non ci sei è il vuoto ad ogni gradino', avoid_range = 20)




def test_avoidance_force_valueerror_avoidance():
    """Test that the _avoidance_force method raises a ValueError when a negative value for avoidance is given as input.

    GIVEN: An invalid input value for avoidance argument in _avoidance_force method

    WHEN: I call _avoidance_force method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError,
                       match = 'Avoidance parameter must be >= 0'
                       ):
                        flock._avoidance_force(avoidance = -1, avoid_range = 20)




def test_avoidance_force_correct_shape():
    """Test that the array returned from _avoidance_force has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _avoidance_force method

    THEN: The returned array has shape (N_birds, 2)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    avoidance_force = flock._avoidance_force(avoidance = 1, avoid_range = 20)

    assert np.shape(avoidance_force) == (200, 2)





def test_avoidance_force_zero_separation():
    """Test that the returned array from _avoidance_force method with avoidance = 0 is full of zeros.

    GIVEN: A Flock object 

    WHEN: I call _avoidance_force method with avoidance = 0

    THEN: The returned array is full of zeros
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    avoidance_force = flock._avoidance_force(avoidance = 0, avoid_range = 20)
    expected_array = np.zeros((200,2))

    assert np.allclose(avoidance_force, expected_array)



def test_avoidance_force_zero_avoid_range():
    """Test that the returned array from _avoidance_force method with avoid_range = 0 is equal to the expected one.

    GIVEN: A Flock object 

    WHEN: I call _avoidance_force method with avoid_range = 0

    THEN: The returned array is equal to the expected one
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    avoidance_force = flock._avoidance_force(avoidance = 2, avoid_range = 0)
    expected_array = np.zeros((200,2))

    assert np.allclose(avoidance_force, expected_array)



def test_avoidance_force_typical_usage():
    """Test that the _avoidance_force returns the expected array.

    GIVEN: A Flock object 

    WHEN: I call _avoidance_force method

    THEN: The resulting array is equal to the expected one
    """
    
    flock = Flock(N_birds = 4, space_length = 100, seed = random_seed)
    initial_positions = np.array([[0,0],[1,10],[5,5],[10,10]])
    flock.init_given_positions(initial_positions)

    avoidance_force = flock._avoidance_force(avoidance = 2, avoid_range = 20)

    expected_avoidance_force = np.array([[1.41421356, 1.41421356],
                                        [1.5493224 , 1.26475298],
                                        [1.41421356, 1.41421356],
                                        [1.41421356, 1.41421356]])
    
    assert np.allclose(avoidance_force, expected_avoidance_force)



def test_compute_forces_changes_attribute_last_forces():
    """Test that the _compute_forces method overwrite the last_forces object attribute.

    GIVEN: A Flock object 

    WHEN: I call _compute_forces method

    THEN: The last_forces object attribute is overwritten
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    initial_forces = flock.last_forces
    flock._compute_forces(separation = 10, 
                          alignment = 2.2, 
                          coherence = 2.2, 
                          avoidance = 10, 
                          visual_range = 20, 
                          avoid_range = 20)
    
    final_forces = flock.last_forces

    assert not np.allclose(initial_forces, final_forces)



def test_compute_forces_zero_force_parameters():
    """Test that the last_forces object attribute is full of zeros after calling _compute_forces when all the force parameters are 0.

    GIVEN: A Flock object

    WHEN: I call _compute_forces method with all the force parameters equal to 0

    THEN: The object last_forces attribute is full of zeros
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    flock._compute_forces(separation = 0, 
                          alignment = 0, 
                          coherence = 0, 
                          avoidance = 0, 
                          visual_range = 20, 
                          avoid_range = 20)
    
    zero_array = np.zeros((200,2))

    assert np.allclose(flock.last_forces, zero_array)



def test_compute_forces_zero_range_parameters():
    """Test that the last_forces object attribute is equal to the coherence force after calling _compute_forces when all the range parameters are 0.

    GIVEN: A Flock object

    WHEN: I call _compute_forces method with all the range parameters equal to 0

    THEN: The object last_forces attribute is equal to the coherence force
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    flock._compute_forces(separation = 10, 
                          alignment = 2.2, 
                          coherence = 2.2, 
                          avoidance = 10, 
                          visual_range = 0, 
                          avoid_range = 0)
    
    coherence_force = flock._coherence_force(coherence = 2.2, visual_range = 0)

    assert np.allclose(flock.last_forces, coherence_force)



def test_compute_forces_typical_usage():
    """Test that the _compute_forces returns the expected array.

    GIVEN: A Flock object 

    WHEN: I call _compute_forces method

    THEN: The resulting array is equal to the expected one
    """
    
    flock = Flock(N_birds = 4, space_length = 100, seed = random_seed)
    initial_positions = np.array([[0,0],[1,10],[5,5],[10,10]])
    flock.init_given_positions(initial_positions)

    flock._compute_forces(separation = 10, 
                          alignment = 2.2, 
                          coherence = 2., 
                          avoidance = 10, 
                          visual_range = 30, 
                          avoid_range = 20)


    expected_last_forces = np.array([[ 1.33682471,  3.86927742],
                                    [ 2.18327471, 14.69672019],
                                    [13.54292104,  2.45704741],
                                    [13.19590604, 15.150434  ]])
    
    assert np.allclose(flock.last_forces, expected_last_forces)



def test_update_state_type_error_dt():
    """Test that the _update_state method raises an error when a string is given as input for dt argument.

    GIVEN: An invalid input type for dt in _update_state method

    WHEN: I call _update_state method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Time step dt must be a floating number',
                ): 
                    flock._update_state(dt = '0.5')



def test_update_state_value_error_dt():
    """Test that the _update_state method raises an error when a negative value is given as input for dt argument.

    GIVEN: An invalid value for dt in _update_state method

    WHEN: I call _update_state method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError,
                       match = 'Time step dt must be > 0',
                ): 
                    flock._update_state(dt = -0.5)



def test_update_state_changes_positions():
    """Test that the _update_state method changed the internal attribute positions.

    GIVEN: A Flock object

    WHEN: I call _update_state method

    THEN: The object.positions array is changed
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    initial_positions = np.copy(flock.positions)

    flock._update_state(dt = 0.5)
    final_positions = np.copy(flock.positions)

    assert not np.allclose(initial_positions, final_positions)



def test_update_state_changes_velocities():
    """Test that the _update_state method changed the internal attribute velocities.

    GIVEN: A Flock object

    WHEN: I call _update_state method

    THEN: The object.velocities array is changed
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    initial_velocities = np.copy(flock.velocities)

    flock._update_state(dt = 0.5)
    final_velocities = np.copy(flock.velocities)

    assert not np.allclose(initial_velocities, final_velocities)



def test_update_state_positions_under_zero_force_parameters():
    """Test that the positions change as expected if all the force parameters are set to 0, i.e. the total force is 0, in the _update_state method.

    GIVEN: A Flock object

    WHEN: I call _update_state method with all the force parameters set to 0

    THEN: The object.positions array changes as expected
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    initial_velocities = np.copy(flock.velocities)
    correct_final_positions = np.copy(flock.positions)

    flock._update_state(dt = 0.5,
                        separation = 0, 
                        alignment = 0, 
                        coherence = 0, 
                        avoidance = 0, 
                        visual_range = 20, 
                        avoid_range = 20)
    
    final_positions = np.copy(flock.positions)

    correct_final_positions += initial_velocities*0.5

    assert np.allclose(final_positions, correct_final_positions)



def test_update_state_velocities_under_zero_force_parameters():
    """Test that the velocities change as expected if all the force parameters are set to 0, i.e. the total force is 0, in the _update_state method.

    GIVEN: A Flock object

    WHEN: I call _update_state method with all the force parameters set to 0

    THEN: The object.velocities array changes as expected
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    initial_velocities = np.copy(flock.velocities)

    flock._update_state(dt = 0.5,
                        separation = 0, 
                        alignment = 0, 
                        coherence = 0, 
                        avoidance = 0, 
                        visual_range = 20, 
                        avoid_range = 20)
    
    final_velocities = np.copy(flock.velocities)

    speed_limit_factors = flock._speed_limit_factors()
    initial_velocities = initial_velocities/ speed_limit_factors[:, None]

    correct_final_velocities = np.clip(initial_velocities, -flock.max_speed, flock.max_speed)

    assert np.allclose(final_velocities, correct_final_velocities)




def test_update_state_typical_usage_positions():
    """Test that positions attribute of the Flock object is correctly updated by _update_state method given typical arguments values.

    GIVEN: A Flock object 

    WHEN: I call _update_state method given typical arguments values

    THEN: The positions attribute of the Flock object is correctly updated
    """

    flock = Flock(N_birds = 4, space_length = 100, seed = random_seed)
    initial_positions = np.array([[0,0],[1,10],[5,5],[10,10]], dtype=float)
    flock.init_given_positions(initial_positions)

    flock._update_state(dt = 0.1,
                        separation = 10, 
                        alignment = 2.2, 
                        coherence = 2.2,
                        avoidance = 10,
                        visual_range = 30, 
                        avoid_range = 40)
    
    final_positions = np.copy(flock.positions)

    
    correct_final_positions = np.array([[ 0.05714559, -0.07679264],
                                        [ 1.23439129, 10.28635114],
                                        [ 4.93953773,  5.29700202],
                                        [10.04854382, 10.24230606]])

    assert np.allclose(final_positions, correct_final_positions)




def test_update_state_typical_usage_velocities():
    """Test that velocities attribute of the Flock object is correctly updated by _update_state method given typical arguments values.

    GIVEN: A Flock object 

    WHEN: I call _update_state method given typical arguments values

    THEN: The velocities attribute of the Flock object is correctly updated
    """

    flock = Flock(N_birds = 4, space_length = 100, seed = random_seed)
    initial_positions = np.array([[0,0],[1,10],[5,5],[10,10]], dtype=float)
    flock.init_given_positions(initial_positions)

    flock._update_state(dt = 0.1,
                        separation = 10, 
                        alignment = 2.2, 
                        coherence = 2.2,
                        avoidance = 10,
                        visual_range = 30, 
                        avoid_range = 40)
    
    final_velocities = np.copy(flock.velocities)

    correct_final_velocities = np.array([[ 0.64368771, -0.56603984],
                                        [ 2.45932357,  3.59053869],
                                        [ 0.06627643,  3.10068128],
                                        [ 1.13675347,  3.17528233]])

    assert np.allclose(final_velocities, correct_final_velocities)




def test_simulate_type_error_num_time_steps():
    """Test that the simulate method raises an error when a float is given as input for num_time_steps argument.

    GIVEN: An invalid input type for num_time_steps in simulate method

    WHEN: I call simulate method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Number of time steps must be an integer',
                ): 
                    flock.simulate(num_time_steps = 100.5)



def test_simulate_value_error_num_time_steps():
    """Test that the simulate method raises an error when a negative value is given as input for num_time_steps argument.

    GIVEN: A negative value for num_time_steps in simulate method

    WHEN: I call simulate method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError,
                       match = 'Number of time steps must be > 0',
                ): 
                    flock.simulate(num_time_steps = -100)



def test_simulate_correct_shape_first_array():
    """Test that the first array returned from simulate has the correct shape.

    GIVEN: A Flock object

    WHEN: I call simulate method

    THEN: The first returned array has shape (num_time_steps,N_birds, 2)
    """

    flock = Flock(N_birds = 50, space_length = 100, seed = random_seed)
    birds_positions_per_time_steps, _ = flock.simulate(num_time_steps = 10)

    assert np.shape(birds_positions_per_time_steps) == (10, 50, 2)



def test_simulate_correct_shape_second_array():
    """Test that the second array returned from simulate has the correct shape.

    GIVEN: A Flock object

    WHEN: I call simulate method

    THEN: The second returned array has shape (num_time_steps,N_birds, 2)
    """

    flock = Flock(N_birds = 50, space_length = 100, seed = random_seed)
    _, birds_velocities_per_time_steps = flock.simulate(num_time_steps = 10)

    assert np.shape(birds_velocities_per_time_steps) == (10, 50, 2)




def test_simulate_zero_velocity_and_forces():
    """Test that given initial static birds without forces, the final positions are equal to the initial positions.

    GIVEN: A Flock object with static birds

    WHEN: I call simulate method with force parameters set to 0, i.e. total force equal 0

    THEN: The initial and final positions are equal
    """

    flock = Flock(N_birds = 50, space_length = 100, seed = random_seed)
    zero_velocities = np.zeros((50,2))
    flock.init_given_velocities(zero_velocities)

    initial_positions = np.copy(flock.positions)

    _,_ = flock.simulate(separation = 0, 
                        alignment = 0, 
                        coherence = 0,
                        avoidance = 0,
                        num_time_steps = 50)

    final_positions = np.copy(flock.positions)

    assert np.allclose(initial_positions, final_positions)



def test_simulate_typical_usage_first_array():
    """Test that the first array returned from simulate is equal to the expected one given typical values of parameters.

    GIVEN: A Flock object 

    WHEN: I call simulate method 

    THEN: The first array is equal to the expected one
    """

    flock = Flock(N_birds = 50, space_length = 100, seed = random_seed)


    birds_positions_per_time_step,_ = flock.simulate(separation = 10, 
                                                    alignment = 2.2, 
                                                    coherence = 2.2,
                                                    avoidance = 10,
                                                    dt = 0.1,
                                                    num_time_steps = 40,
                                                    visual_range = 30,
                                                    avoid_range = 40)


    flock = Flock(N_birds = 50, space_length = 100, seed = random_seed)
    correct_positions_per_time_step = np.zeros((40, 50, 2))
    

    for i in (range(40)):
        correct_positions_per_time_step[i] = flock.positions

        flock._update_state(dt = 0.1,
                            separation = 10, 
                            alignment = 2.2, 
                            coherence = 2.2,
                            avoidance = 10,
                            visual_range = 30, 
                            avoid_range = 40)

    assert np.allclose(birds_positions_per_time_step, correct_positions_per_time_step)




def test_simulate_typical_usage_second_array():
    """Test that the second array returned from simulate is equal to the expected one given typical values of parameters.

    GIVEN: A Flock object 

    WHEN: I call simulate method 

    THEN: The second array is equal to the expected one
    """

    flock = Flock(N_birds = 50, space_length = 100, seed = random_seed)


    _, birds_velocities_per_time_step = flock.simulate(separation = 10, 
                                                    alignment = 2.2, 
                                                    coherence = 2.2,
                                                    avoidance = 10,
                                                    dt = 0.1,
                                                    num_time_steps = 40,
                                                    visual_range = 30,
                                                    avoid_range = 40)


    flock = Flock(N_birds = 50, space_length = 100, seed = random_seed)
    correct_velocities_per_time_step = np.zeros((40, 50, 2))
    

    for i in (range(40)):
        correct_velocities_per_time_step[i] = flock.velocities

        flock._update_state(dt = 0.1,
                            separation = 10, 
                            alignment = 2.2, 
                            coherence = 2.2,
                            avoidance = 10,
                            visual_range = 30, 
                            avoid_range = 40)

    assert np.allclose(birds_velocities_per_time_step, correct_velocities_per_time_step)