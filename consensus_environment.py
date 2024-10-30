#
# This is a definition of the orientation consensus environment simulation engine.
# It provides the initialization of a random environment and methods of communication between the different agents.
#

import random
import geometry
import math

import agent

import neat

class Environment:
    """
    The instance holding every agents and methods of communication
    """

    def __init__(self, length=100, height=100, N=10):
        """
        Creates a new random environment of length and height given, with N agents placed with random locations and headings, and angular velocity of zero
        """
 
        self.length=length
        self.height=height

        self.agent_list=[]

        for i in range(N):
        
            x = random.random() * length
            y = random.random() * height
            location = geometry.Point(x,y)

            heading = random.random() * 360

            robot = agent.Agent(location, i, heading, angular_vel=0)
            self.agent_list.append(robot)

    def communication(self):
        """
        Gives each robot on the map a new message in one of their radar
        """
        for robot in self.agent_list:

            # The list of all robots in range
            robots_in_range = []
            for other_robot in self.agent_list:
                if other_robot.agent_id != robot.agent_id and robot.location.distance(other_robot.location) <= 80:
                    robots_in_range.append(other_robot)

            # Randomly selects one robot from which the message will be received
            random_id = random.randint(0, len(robots_in_range)-1)
            selected_robot = robots_in_range[random_id]

            robot.update_radar(selected_robot)

    def consensus_verified(self):
        """
        Returns : True if all robots in the environment are heading the same way, with an error tolerated of 5°.
        """
        min_heading = self.agent_list[0].heading
        max_heading = self.agent_list[0].heading

        for robot in self.agent_list:


            if robot.heading < min_heading:
                min_heading = robot.heading

            elif robot.heading > max_heading:
                max_heading = robot.heading
            
        return max_heading-min_heading < 5    # doesn't consider the fact that it can be around 360°
    
    def avg_heading(self):
        """
        Returns the average angle of heading of every robots in the environment
        """
        r0 = self.agent_list[0]
        sum_point = geometry.Point(math.cos(geometry.deg_to_rad(r0.heading)), math.sin(geometry.deg_to_rad(r0.heading)))
        for r in self.agent_list[1:]:
            new_point = geometry.Point(math.cos(geometry.deg_to_rad(r.heading)), math.sin(geometry.deg_to_rad(r.heading)))
            sum_point = geometry.sum_points(sum_point, new_point)
        
        return sum_point.angle()
    
    
    def fitness(self):
        """
        Return the fitness score of the environment
        """
        R = len(self.agent_list)

        sum_r = 0
        avg_heading = self.avg_heading()

        for r in self.agent_list:
            sum_r += r.individual_fitness(avg_heading)

        return (1/R)*sum_r


def consensus_simulation_evaluate(env, net, time_steps=600, robot_orientation_list = None):
    """
    The function to evaluate simulation for specific environment
    and controll ANN provided. The results will be saved into provided
    agent record holder.
    Arguments:
        env: The configuration environment.
        net: The agent's control ANN.
        time_steps: The number of time steps for maze simulation.
    """
    for i in range(time_steps):
        if consensus_simulation_step(env, net, robot_orientation_list):
            print("Consensus reached in %d steps" % (i + 1))
            return 1.0
            
    # Calculate the fitness score based on distance from exit
    fitness = env.fitness()
    if fitness <= 0.01:
        fitness = 0.01

    return fitness
    
def consensus_simulation_step(env, net, robot_orientation_list):
    """
    The function to perform one step of consensus orientation simulation.
    Arguments:
    env: The maze configuration environment.
       net: The maze solver agent's control ANN
    Returns:
        The True if every robots are heading the same way, with a 5° error tolerated
    """
    
    # Activate/update communication for this step
    env.communication()

    for i, robot in enumerate(env.agent_list):
        # create inputs from the current state of the robot in environment
        inputs = robot.create_net_inputs()
        # load inputs into controll ANN and get results
        output = net.activate(inputs)
        # apply control signal to the environment and update
        robot.apply_outputs(output)

        if robot_orientation_list != None:
            robot_orientation_list[i].append(robot.heading)

    return env.consensus_verified()


        