# -*- coding: utf-8 -*-
'''
Author: Mario Massimo
Date: March 2025
'''

import argparse
from flock_class import Flock
from animation import animate


def main():
    ''' Main function to parse command-line arguments, then simulate a flock dynamic

    Command-line settings:

    Parameters
    ----------
    N_birds : int, optional
        Number of birds in the simulation, default is 200

    space_length : float, optional
        Length of the side of the square containing the birds, default is 100

    seed : int, optional
            Integer initializing the random seed for reproducibility, default is 1999

    separation : float, optional
        Value of the separation parameter, default is 10

    alignment : float, optional
        Value of the alignment parameter, default is 2.2

    coherence : float, optional
        Value of the coherence parameter, default is 2.2

    avoidance : float, optional
        Value of the avoidance parameter, default is 10
    
    visual_range : float, optional
        Radius of a circle with which a bird can see other birds, default is 30

    avoid_range : float, optional
        Radius of a circle with which a bird sees the simulation edges, default is 40

    dt : float, optional
        Time step of the simulation, default is 0.1

    num_time_steps : int, optional
        Total number of time steps, default is 300

    save : bool, optional
        Bool variable to save or not the gif produce, default is False


    Returns
    -------
    None
    '''

    parser = argparse.ArgumentParser(
        description = "Simulate and animate a flock dynamic")
    
    parser.add_argument(
        "--N",
        type = int,
        help = "Number of birds",
        default = 200
    )


    parser.add_argument(
        "--space_length",
        type = float,
        help = "Simulation space length",
        default = 100
    )


    parser.add_argument(
        "--seed",
        type = int,
        help = "Random seed",
        default = 1999
    )


    parser.add_argument(
        "--separation",
        type = float,
        help = "Separation coefficient",
        default = 10.
    )


    parser.add_argument(
        "--alignment",
        type = float,
        help = "Aligment coefficient",
        default = 2.2
    )


    parser.add_argument(
        "--coherence",
        type = float,
        help = "Coherence coefficient",
        default = 2.2
    )


    parser.add_argument(
        "--avoidance",
        type = float,
        help = "Avoidance coefficient",
        default = 10.
    )


    parser.add_argument(
        "--visual_range",
        type = float,
        help = "Visual range",
        default = 30.
    )


    parser.add_argument(
        "--avoid_range",
        type = float,
        help = "Avoid range",
        default = 40.
    )


    parser.add_argument(
        "--dt",
        type = float,
        help = "Time step of the simulation",
        default = 0.1
    )


    parser.add_argument(
        "--num_t_steps",
        type = int,
        help = "Total number of time steps",
        default = 300
    )


    parser.add_argument(
        "--save",
        type = bool,
        help = "Save choice",
        default = False
    )


    args = parser.parse_args()


    flock = Flock(N_birds = args.N, 
                  space_length = args.space_length,
                  seed = args.seed)
    

    birds_positions_per_time_step, birds_velocities_per_time_step = flock.simulate(separation = args.separation, 
                                                                                    alignment = args.alignment, 
                                                                                    coherence = args.coherence, 
                                                                                    avoidance = args.avoidance, 
                                                                                    dt = args.dt, 
                                                                                    num_time_steps = args.num_t_steps, 
                                                                                    visual_range = args.visual_range, 
                                                                                    avoid_range = args.avoid_range)
    


    animate(birds_positions_per_time_step = birds_positions_per_time_step, 
            birds_velocities_per_time_step = birds_velocities_per_time_step,
            space_length = args.space_length,
            num_time_steps = args.num_t_steps,
            save = args.save)




if __name__ == '__main__':
    main()