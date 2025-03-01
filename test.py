# -*- coding: utf-8 -*-

import pytest
import numpy as np
from flock_class import Flock


def test_invalid_type_N_birds_in_constructor():
    """Test that the Flock class constructor raises an error when an invalid type as N_birds is provided.

    GIVEN: An invalid type for N_birds in the class constructor

    WHEN: The constructor is called to create an object Flock

    THEN: A TypeError is raised 
    """

    with pytest.raises(TypeError,
                       match = 'Number of birds must be an integer number',
                ): 
                    flock = Flock('mille', 100, 1999)


def test_invalid_value_N_birds_in_constructor():
    """Test that the Flock class constructor raises an error when a negative number of birds is provided.

    GIVEN: A negative number for N_birds in the class constructor

    WHEN: The constructor is called to create an object Flock

    THEN: A ValueError is raised 
    """

    with pytest.raises(ValueError,
                       match = 'Number of birds must be > 0',
                ): 
                    flock = Flock(-1, 100, 1999)