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

    with pytest.raises(ValueError): 
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

    right_directions = flock.positions - flock.positions[:, None]

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

    directions = flock._directions_between_birds()

    right_distances = np.linalg.norm(directions, axis=2)
    right_distances[right_distances == 0] = np.inf

    assert np.allclose(distances, right_distances)



def test_directions_unitary_vectors_right_shape():
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

    directions = flock._directions_between_birds()
    distances = flock._distances_between_birds()

    right_unit_directions = directions / distances[:,:,None]

    assert np.allclose(unit_directions, right_unit_directions)



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

    distances = flock._distances_between_birds()
    correct_closest = np.argmin(distances, axis=1)

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

    mask = flock._visual_range_mask(visual_range = 20)

    expected_array = np.count_nonzero(mask, axis=1)
    expected_array[expected_array == 0] = 1

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

    mask = flock._visual_range_mask(visual_range = 20)
    num_close_non_zero = flock._num_close_non_zero(visual_range = 20)

    expected_array = (mask[:, :, None] * flock.velocities).sum(axis=1) / num_close_non_zero[:, None]

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

    mask = flock._visual_range_mask(visual_range = 20)
    num_close_non_zero = flock._num_close_non_zero(visual_range = 20)

    expected_array = (mask[:, :, None] * flock.positions).sum(axis=1) / num_close_non_zero[:, None] - flock.positions

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

    expected_array = np.any(np.abs(flock.positions - flock.space_length/2.0) >= (flock.space_length/2.0 - 20), axis=1)

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

    center = np.array([flock.space_length/2, flock.space_length/2])
    expected_center_distances = center - flock.positions

    expected_center_distances[np.all(expected_center_distances==[0,0], axis=1)] = [1e-5, 1e-5]
    expected_center_distances /= np.linalg.norm(expected_center_distances, axis=1)[:,None]
    
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
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    separation_force = flock._separation_force(separation = 2, visual_range = 20)

    unit_directions = flock._directions_unitary_vectors()
    mask = flock._visual_range_mask(visual_range = 20)
    closest_index = flock._closest_index()
    expected_separation_force = - 2 * unit_directions[np.arange(unit_directions.shape[0]),closest_index] * mask[np.arange(mask.shape[0]),closest_index][:,None]
    
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




def test_alignment_force_zero_separation():
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
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    alignment_force = flock._alignment_force(alignment = 2, visual_range = 20)

    aligment_vector = flock._alignment_vector(visual_range = 20)
    aligment_lengths = np.linalg.norm(aligment_vector, axis=1)
    aligment_lengths[aligment_lengths == 0] = 1
    expected_force_alignment = 2 * aligment_vector / aligment_lengths[:,None]
    
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
                    flock._coherence_force(coherence = 'ho sceso dandoti il braccio', visual_range = 20)



def test_coherence_force_type_error_visual_range():
    """Test that the _coherence_force method raises an error when a string is given as input for visual_range argument.

    GIVEN: An invalid input type for visual_range in _coherence_force method

    WHEN: I call _coherence_force method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Visual range must be a floating number',
                ): 
                    flock._coherence_force(coherence = 1, visual_range = 'almeno un milione di scale')



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



def test_coherence_force_valueerror_visual_range():
    """Test that the _coherence_force method raises a ValueError when a negative value for visual_range is given as input.

    GIVEN: An invalid input value for visual_range argument in _coherence_force method

    WHEN: I call _coherence_force method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError):
                        flock._coherence_force(coherence = 1, visual_range = -20)



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
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    coherence_force = flock._coherence_force(coherence = 2, visual_range = 20)

    coherence_vector = flock._coherence_vector(visual_range = 20)
    expected_force_coherence = 2 * coherence_vector / np.linalg.norm(coherence_vector, axis=1)[:,None]
    
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
                    flock._avoidance_force(avoidance = 'e ora che non ci sei', avoid_range = 20)



def test_avoidance_force_type_error_avoid_range():
    """Test that the _avoidance_force method raises an error when a string is given as input for avoid_range argument.

    GIVEN: An invalid input type for avoid_range in _avoidance_force method

    WHEN: I call _avoidance_force method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Avoid range must be a floating number',
                ): 
                    flock._avoidance_force(avoidance = 1, avoid_range = 'è il vuoto ad ogni gradino')



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



def test_avoidance_force_valueerror_avoid_range():
    """Test that the _avoidance_force method raises a ValueError when a negative value for avoid_range is given as input.

    GIVEN: An invalid input value for avoid_range argument in _avoidance_force method

    WHEN: I call _avoidance_force method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError):
                        flock._avoidance_force(avoidance = 1, avoid_range = -20)



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
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    avoidance_force = flock._avoidance_force(avoidance = 2, avoid_range = 20)

    edge_mask = flock._edge_mask(avoid_range = 20)
    center_directions = flock._center_direction()
    expected_avoidance_force = 2 * center_directions * edge_mask[:,None]
    
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
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)

    flock._compute_forces(separation = 10, 
                          alignment = 2.2, 
                          coherence = 2., 
                          avoidance = 10, 
                          visual_range = 30, 
                          avoid_range = 20)

    force_separation = flock._separation_force(separation = 10, visual_range = 30)
    force_alignment = flock._alignment_force(alignment = 2.2, visual_range = 30)
    force_coherence = flock._coherence_force(coherence = 2., visual_range = 30)
    force_avoidance = flock._avoidance_force(avoidance = 10, avoid_range = 20)

    expected_last_forces = force_coherence + force_avoidance + force_alignment + force_separation
    
    assert np.allclose(flock.last_forces, expected_last_forces)