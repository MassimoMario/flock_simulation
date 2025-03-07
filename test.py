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
    right_directions = np.array([[[ 0,  0], [2, 2]],
                                [[ -2,  -2], [ 0,  0]]])

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
    right_distances = np.array([[np.inf, np.sqrt(8)],
                                [np.sqrt(8), np.inf]])

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

    unit_distances = flock._directions_unitary_vectors()
    right_unit_distances = np.array([[[ 0.        ,  0.        ],
                                    [ 2/np.sqrt(8),  2/np.sqrt(8)]],
                                    [[-2/np.sqrt(8), -2/np.sqrt(8)],
                                    [ 0.        ,  0.        ]]])

    assert np.allclose(unit_distances, right_unit_distances)



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
                    flock._visual_range_mask('ventimilioni')



def test_visual_range_mask_valueerror():
    """Test that the _visual_range_mask method raises a ValueError when a negative value is given as input.

    GIVEN: An invalid input type for _visual_range_mask method

    WHEN: I call _visual_range_mask method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError):
                    flock._visual_range_mask(-0.4)




def test_visual_range_mask_correct_shape():
    """Test that the _visual_range_mask method returns a np.ndarray with the correct shape.

    GIVEN: A Flock object

    WHEN: I call _visual_range_mask method

    THEN: The resulting array has shape (N_birds, N_birds)
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    mask = flock._visual_range_mask(20)

    assert np.shape(mask) == (200,200)




def test_visual_range_mask_zero_visual_range():
    """Test that the _visual_range_mask method returns a mask full of False when visual_range is 0.

    GIVEN: A Flock object

    WHEN: I call _visual_range_mask method with visual_range = 0

    THEN: The resulting mask is full of False
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    mask = flock._visual_range_mask(0)
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
    mask = flock._visual_range_mask(50)
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
    correct_closest = np.array([1,0,1])

    assert np.allclose(closest_index, correct_closest)



def test_num_close_non_zero_typeerror():
    """Test that the _num_close_non_zero method raises a TypeError when a string is given as input.

    GIVEN: An invalid input type for _num_close_non_zero method

    WHEN: I call _num_close_non_zero method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Visual range must be a floating number',
                ): 
                    flock._num_close_non_zero('quarantaquattro')



def test_num_close_non_zero_valueerror():
    """Test that the _num_close_non_zero method raises a ValueError when a negative value is given as input.

    GIVEN: An invalid input type for _num_close_non_zero method

    WHEN: I call _num_close_non_zero method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError):
                    flock._num_close_non_zero(-0.4)



def test_num_close_non_zero_correct_shape():
    """Test that the array returned from _num_close_non_zero has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _num_close_non_zero method

    THEN: The returned array has shape (N_birds)
    """
    
    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    num_close_non_zero = flock._num_close_non_zero(20)

    assert np.shape(num_close_non_zero) == (200,)




def test_num_close_non_zero_only_one_bird():
    """Test that the returned array from _num_close_non_zero is [1] if only one bird is present.

    GIVEN: A Flock object with only one bird

    WHEN: I call _num_close_non_zero method

    THEN: The returned array is equal to [1]
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)
    num_close_non_zero = flock._num_close_non_zero(20)
    one_array = np.array([1])

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

    num_close_non_zero = flock._num_close_non_zero(20)
    expected_array = np.array([2,2,2])

    assert np.allclose(num_close_non_zero, expected_array)




def test_alignment_vector_typeerror():
    """Test that the _alignment_vector method raises a TypeError when a string is given as input.

    GIVEN: An invalid input type for _alignment_vector method

    WHEN: I call _alignment_vector method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Visual range must be a floating number',
                ): 
                    flock._alignment_vector('se telefonando')




def test_alignment_vector_valueerror():
    """Test that the _alignment_vector method raises a ValueError when a negative value is given as input.

    GIVEN: An invalid input type for _alignment_vector method

    WHEN: I call _alignment_vector method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError):
                    flock._alignment_vector(-8.2)




def test_alignment_vector_correct_shape():
    """Test that the array returned from _alignment_vector has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _alignment_vector method

    THEN: The returned array has shape (N_birds, 2)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    alignment_vector = flock._alignment_vector(20)

    assert np.shape(alignment_vector) == (200,2)



def test_alignment_vector_only_one_bird():
    """Test that the returned array from _alignment_vector is [[0],[0]] if only one bird is present.

    GIVEN: A Flock object with only one bird

    WHEN: I call _alignment_vector method

    THEN: The returned array is equal to [[0],[0]]
    """
    
    flock = Flock(N_birds = 1, space_length = 100, seed = random_seed)
    alignment_vector = flock._alignment_vector(20)
    one_array = np.array([[0],[0]])

    assert np.allclose(alignment_vector, one_array)



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

    alignment_vector = flock._alignment_vector(20)
    expected_array = np.array([[1., 2.],
                                [1., 1.]])

    assert np.allclose(alignment_vector, expected_array)



def test_coherence_vector_typeerror():
    """Test that the _coherence_vector method raises a TypeError when a string is given as input.

    GIVEN: An invalid input type for _coherence_vector method

    WHEN: I call _coherence_vector method

    THEN: A TypeError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(TypeError,
                       match = 'Visual range must be a floating number',
                ): 
                    flock._coherence_vector('volevo essere un duro')



def test_coherence_vector_valueerror():
    """Test that the _coherence_vector method raises a ValueError when a negative value is given as input.

    GIVEN: An invalid input type for _coherence_vector method

    WHEN: I call _coherence_vector method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError):
                    flock._coherence_vector(-1.2)


    

def test_coherence_vector_correct_shape():
    """Test that the array returned from _coherence_vector has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _coherence_vector method

    THEN: The returned array has shape (N_birds, 2)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    coherence_vector = flock._coherence_vector(20)

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

    coherence_vector = flock._coherence_vector(20)
    expected_array = np.array([[-1],[-1]])

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

    coherence_vector = flock._coherence_vector(20)
    expected_array = np.array([[0., 1.],
                                [0., -1.]])

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
                    flock._edge_mask('ventinove e qualcosina')



def test_edge_mask_valueerror():
    """Test that the _edge_mask method raises a ValueError when a negative value is given as input.

    GIVEN: An invalid input type for _edge_mask method

    WHEN: I call _edge_mask method

    THEN: A ValueError is raised
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    

    with pytest.raises(ValueError):
                    flock._edge_mask(-1.2)




def test_edge_mask_correct_shape():
    """Test that the array returned from _edge_mask has the correct shape.

    GIVEN: A Flock object

    WHEN: I call _edge_mask method

    THEN: The returned array has shape (N_birds)
    """

    flock = Flock(N_birds = 200, space_length = 100, seed = random_seed)
    edge_mask = flock._edge_mask(20)

    assert np.shape(edge_mask) == (200,)



def test_edge_mask_typical_usage():
    """Test that the _edge_mask method returns an array as expected given two birds with known positions.

    GIVEN: A Flock object with two birds with known positions

    WHEN: I call _edge_mask method

    THEN: The returned array is equal to the expected one
    """
    
    flock = Flock(N_birds = 2, space_length = 100, seed = random_seed)
    initial_positions = np.array([[1,1],[50,50]])
    flock.init_given_positions(initial_positions)

    edge_mask = flock._edge_mask(20)
    expected_array = np.array([True, False])

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




def test_center_direction_vectors_normalization():
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




def test_center_direction_vectors_typical_usage():
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

    expected_center_distances = np.array([[1,0],
                                          [0,1],
                                          [-1,0],
                                          [np.sqrt(2)/2,np.sqrt(2)/2]])
    
    assert np.allclose(unit_center_distances, expected_center_distances)