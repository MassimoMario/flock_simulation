# -*- coding: utf-8 -*-
'''
Author: Mario Massimo
Date: March 2025
'''

import argparse
import configparser
import sys
import numpy as np
from scripts.flock_class import Flock
from scripts.utils import animate
from scripts.utils import set_type


def main():
    ''' Main function to parse command-line arguments, then simulate a flock dynamic

    Command-line settings:

    Parameters
    ----------
    N_birds : int, optional
        Number of birds in the simulation, default is 200

    space_length : float, optional
        Length of the side of the square containing the birds, default is 100

    positions_i : list, optional
        2d list with initial positions (will be casted into np.ndarray), shape [N,2]

    velocities_i: list, optional
        2d list with initial velocities (will be casted into np.ndarray), shape [N,2]

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

    save_anim : bool, optional
        Bool variable to save or not the animation in gif format, default is False

    save_ps_vs : bool, optional
        Bool variable to save or not the arrays produces by the simulation in .npy format, default is False


    Returns
    -------
    None
    '''

    parser = argparse.ArgumentParser(
        description = "Simulate and animate a flock dynamic")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (.ini). Command-line arguments override config file values"
    )

    parser.add_argument(
        "--N",
        type = int,
        help = "Number of birds (int)",
        default = 200
    )


    parser.add_argument(
        "--space_length",
        type = float,
        help = "Simulation space length (float)",
        default = 100
    )


    parser.add_argument(
        "--positions_i",
        action = 'append',
        nargs = '+',
        type = float,
        help = "Initial positions array (must be given calling --positions_i with 2 numbers as many times as N)"
    )


    parser.add_argument(
        "--velocities_i",
        action = 'append',
        nargs = '+',
        type = float,
        help = "Initial velocities array (must be given calling --velocities_i with 2 numbers as many times as N)"
    )



    parser.add_argument(
        "--positions_file",
        type = str,
        help = "File .npy containing the initial positions array (must have shape (N,2))"
    )



    parser.add_argument(
        "--velocities_file",
        type = str,
        help = "File .npy containing the initial velocities array (must have shape (N,2))"
    )


    parser.add_argument(
        "--seed",
        type = int,
        help = "Random seed (int)",
        default = None
    )


    parser.add_argument(
        "--separation",
        type = float,
        help = "Separation coefficient (float)",
        default = 10.
    )


    parser.add_argument(
        "--alignment",
        type = float,
        help = "Aligment coefficient (float)",
        default = 2.2
    )


    parser.add_argument(
        "--coherence",
        type = float,
        help = "Coherence coefficient (float)",
        default = 2.2
    )


    parser.add_argument(
        "--avoidance",
        type = float,
        help = "Avoidance coefficient (float)",
        default = 10.
    )


    parser.add_argument(
        "--visual_range",
        type = float,
        help = "Visual range (float)",
        default = 30.
    )


    parser.add_argument(
        "--avoid_range",
        type = float,
        help = "Avoid range (float)",
        default = 40.
    )


    parser.add_argument(
        "--dt",
        type = float,
        help = "Time step of the simulation (float)",
        default = 0.1
    )


    parser.add_argument(
        "--num_t_steps",
        type = int,
        help = "Total number of time steps (int)",
        default = 300
    )


    parser.add_argument(
        "--save_anim",
        type = str,
        help = "Save animation choice (True/False or Yes/No)",
        default = 'False'
    )

    parser.add_argument(
        "--save_ps_vs",
        type = str,
        help = "Save birds positions and velocities per time step in a numpy array (True/False or Yes/No)",
        default = 'False'
    )


    args = parser.parse_args()
    setattr(args, 'save_anim', set_type(args.save_anim))
    setattr(args, 'save_ps_vs', set_type(args.save_ps_vs))

    if args.config:
        config = configparser.ConfigParser()
        config.read(args.config)

        for key, value in vars(args).items():
            if key in config['Default']:
                setattr(args, key, value if f'--{key}' in sys.argv else set_type(config['Default'][key]))
            else: 
                pass


    flock = Flock(N_birds = args.N, 
                  space_length = args.space_length,
                  seed = args.seed)
    


    if args.positions_file:
        init_positions = np.load(args.positions_file)
        flock.init_given_positions(init_positions)

    elif args.positions_i:
        init_positions = np.array(args.positions_i)
        flock.init_given_positions(init_positions)



    if args.velocities_file:
        init_velocities = np.load(args.velocities_file)
        flock.init_given_velocities(init_velocities)

    elif args.velocities_i:
        init_velocities = np.array(args.velocities_i)
        flock.init_given_velocities(init_velocities)


    print('\nRandom seed of the simulation: ', args.seed, '\n')


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
            save = args.save_anim)


    if args.save_ps_vs:
        np.save('birds_positions_per_time_step.npy', birds_positions_per_time_step)
        np.save('birds_velocities_per_time_step.npy', birds_velocities_per_time_step)


    if args.seed == None:
        print("\nThe simulation random seed is", None, "Do you want to save initial positions and velocities to restart the same simulation? [Yes/No]\n")
        
        save_pi_vi = set_type(input())

        if save_pi_vi == True:
            np.save('initial_positions.npy', birds_positions_per_time_step[0])
            np.save('initial_velocities.npy', birds_velocities_per_time_step[0])


if __name__ == '__main__':
    main()