#
# The script to run the orientation consensus experiment
#

import random
import time
import copy
import os
import argparse
import numpy as np

import consensus_environment as env
import agent
import consensus_visualize as visualize
import utils

import neat

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')
out_dir = os.path.join(out_dir, 'consensus_objective')

class ConsensusSimulationTrial:
    """
    The class to hold consensus orientation simulator execution parameters and results.
    """
    def __init__(self, consensus_env, population):
        """
        Creates new instance and initialize fileds.
        Arguments:
            consensus_env:   The environment as loaded (randomly placed robots)
            population:      The population for this trial run
        """
        # The initial simulation environment
        self.orig_consensus_environment = consensus_env

        # The record store for evaluated solver agents
        # self.record_store = agent.AgentRecordStore()

        # The NEAT population object
        self.population = population

def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list.
    Arguments:
        genomes: The list of genomes from population in the 
                 current generation
        config:  The configuration settings with algorithm
                 hyper-parameters
    """
    for genome_id, genome in genomes:
        genome.fitness = eval_fitness(genome_id, genome, config)

def eval_fitness(genome_id, genome, config, time_steps=600):
    """
    Evaluates fitness of the provided genome.
    Arguments:
        genome_id:  The ID of genome.
        genome:     The genome to evaluate.
        config:     The NEAT configuration holder.
        time_steps: The number of time steps to execute for consensus solver simulation.
    Returns:
        The phenotype fitness score in range (0, 1]
    """
    # run the simulation
    maze_env = copy.deepcopy(trialSim.orig_consensus_environment)

    # create the net with feed-forward neat class or Recurent Network
    if config.genome_config.feed_forward:
        control_net = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        control_net = neat.nn.RecurrentNetwork.create(genome, config)

    epochs_fitness = []
    for i in range(evaluate_epochs):
        epochs_fitness.append(env.consensus_simulation_evaluate(
                                        env=maze_env, 
                                        net=control_net, 
                                        time_steps=time_steps))
    fitness_array=np.array(epochs_fitness)
    fitness = fitness_array.mean()

    # Store simulation results into the agent record
    #record = agent.AgentRecord(
    #    generation=trialSim.population.generation,
    #    agent_id=genome_id)
    #record.fitness = fitness
    #record.x = maze_env.agent.location.x
    #record.y = maze_env.agent.location.y
    #record.hit_exit = maze_env.exit_found
    #record.species_id = trialSim.population.species.get_species_id(genome_id)
    #record.species_age = record.generation - trialSim.population.species.get_species(genome_id).created
    # add record to the store
    #trialSim.record_store.add_record(record)

    return fitness

def run_experiment(config_file, consensus_env, trial_out_dir, n_generations=100, silent=False):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file:    The path to the file with experiment configuration
        consensus_env:  The environment to use in simulation.
        trial_out_dir:  The directory to store outputs for this trial
        n_generations:  The number of generations to execute.
        silent:         If True than no intermediary outputs will be
                        presented until solution is found.
        args:           The command line arguments holder.
    Returns:
        True if experiment finished with successful solver found. 
    """

    # set random seed
    seed = int(time.time())
    random.seed(seed)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Create the trial simulationf
    global trialSim
    trialSim = ConsensusSimulationTrial(consensus_env=consensus_env, population=p)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='%s/consensus-neat-checkpoint-' % trial_out_dir))

    # Run for up to N generations.
    start_time = time.time()
    best_genome = p.run(eval_genomes, n=n_generations)

    elapsed_time = time.time() - start_time

    # Display the best genome among generations.
    print('\nBest genome:\n%s' % (best_genome))

    solution_found = (best_genome.fitness >= config.fitness_threshold)
    if solution_found:
        print("SUCCESS: The orientation consensus solver was found !!!")
    else:
        print("FAILURE: Failed to find the orientation consensus solver !!!")

    # write the record store data
    # rs_file = os.path.join(trial_out_dir, "data.pickle")
    # trialSim.record_store.dump(rs_file)

    # print("Record store file: %s" % rs_file)
    print("Random seed:", seed)
    print("Trial elapsed time: %.3f sec" % (elapsed_time))

    # Visualize the experiment results
    if not silent or solution_found:
        node_names =   {-1:'MODE', -2:'TETA_TX', -3:'TETA_RX', -4:'RCV_1', -5:'RCV_2',
                        0:'MODE', 1:'ANG_VEL', 2:'SEN_1', 3:'SEN_2'}
        visualize.draw_net(config, best_genome, True, node_names=node_names, directory=trial_out_dir, fmt='svg')
        #if args is None:
        #    visualize.draw_maze_records(maze_env, trialSim.record_store.records, view=True)
        #else:
        #    visualize.draw_maze_records(maze_env, trialSim.record_store.records, 
        #                                view=True, 
        #                                width=args.width,
        #                                height=args.height,
        #                                filename=os.path.join(trial_out_dir, 'maze_records.svg'))
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(trial_out_dir, 'avg_fitness.svg'))
        visualize.plot_species(stats, view=False, filename=os.path.join(trial_out_dir, 'speciation.svg'))

    # create the best genome simulation path and visualize it
    consensus_env = copy.deepcopy(trialSim.orig_consensus_environment)

    # create the best genome net with feed-forward or recurrent neat class
    if config.genome_config.feed_forward:
        control_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    else:
        control_net = neat.nn.RecurrentNetwork.create(best_genome, config)

    #best_fitness = 0
    #for i in range(evaluate_epochs):
    #    robot_orientation_list = [[] for i in range(len(consensus_env.agent_list))]
    #    evaluate_fitness = env.consensus_simulation_evaluate(consensus_env, control_net,
    #                                                     robot_orientation_list=robot_orientation_list)
    #    if evaluate_fitness > best_fitness:
    #        best_fitness = evaluate_fitness
    #        best_robot_orientation_list = robot_orientation_list

    #print("Evaluated fitness of best agent: %f" % best_fitness)
    #visualize.animate_experiment(consensus_env, best_robot_orientation_list, best_genome, trial_out_dir)

    # try the experiment with the best genome until one successful run is found
    robot_orientation_list = [[] for i in range(len(consensus_env.agent_list))]
    tries = 0
    #loop until a succesful run is found
    if solution_found:
        fitness = env.consensus_simulation_evaluate(consensus_env, control_net,
                                                                robot_orientation_list=robot_orientation_list)
        while tries<evaluate_epochs and fitness < config.fitness_threshold:
        
            print("Run n°%d unsuccessful. Fitness : %f" % (tries, fitness))
            robot_orientation_list = [[] for i in range(len(consensus_env.agent_list))]
            fitness = env.consensus_simulation_evaluate(consensus_env, control_net,
                                                                robot_orientation_list=robot_orientation_list)
            tries +=1

        if tries<evaluate_epochs:
            print("Successful run found after %d tries. Fitness = %f" % (tries, fitness))
        else:
            print("No successful run was found in %d tries with the best genome. Visualization of the last try." % evaluate_epochs)

        visualize.animate_experiment(consensus_env, robot_orientation_list, best_genome, trial_out_dir)
        visualize.plot_headings(robot_orientation_list, best_genome, dirname=trial_out_dir, view=True)

    return solution_found


if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The maze experiment runner.")
    parser.add_argument('-g', '--generations', default=500, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('-e', '--epochs', default=5, type=int,
                        help='The number of epochs used to evaluate the fitness of one genome.')
    #parser.add_argument('--width', type=int, default=100, help='The width of the records subplot')
    #parser.add_argument('--height', type=int, default=100, help='The height of the records subplot')
    args = parser.parse_args()

    # create variable evaluate_epoch and set it global
    global evaluate_epochs
    evaluate_epochs=args.epochs

    # Determine path to configuration file.
    config_path = os.path.join(local_dir, 'consensus_config.ini')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir)

    # Run the experiment
    # maze_env_config = os.path.join(local_dir, '%s_maze.txt' % args.maze)
    # maze_env = env.read_environment(maze_env_config)

    consensus_env = env.Environment()

    # visualize.draw_maze_records(maze_env, None, view=True)

    print("Starting the experiment")
    run_experiment( config_file=config_path, 
                    consensus_env=consensus_env, 
                    trial_out_dir=out_dir,
                    n_generations=args.generations)