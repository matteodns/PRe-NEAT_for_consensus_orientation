#
# This is the definition of an orientation consensus agent.
# It  provides initialization and methods to simulate the behavior of the agents through the interactions with its radars
#

import math
import geometry
import numpy as np

MAX_ANGULAR_VEL = 3.0

class Agent:
    """
    The instance that holds every information of one robot in the consensus map and methods of mobility of the agent
    """

    def __init__(self, location, agent_id, heading=0.0, angular_vel=0.0, radius=8.0, mode=1, radar_range=80):
        """
        Creates new Agent with specified parameters.
        Arguments:
            location:               The agent initial position within maze
            heading:                The heading direction in degrees.
            angular_vel:            The angular velocity of the agent.
            radius:                 The agent's body radius.
            range_finder_range:     The maximal detection range for range finder sensors.
        """
        self.agent_id = agent_id
        self.heading = heading
        self.angular_vel = angular_vel
        self.radius = radius
        self.location = location
        self.mode = 1 # 0=transmission, 1=emission
        self.radar_range=radar_range
        self.msg_rcv_1=None
        self.msg_rcv_2=None
        self.msg_sen_1=0
        self.msg_sen_2=0

        # This variable contains the angle of radars (here is an EPUCK robot)
        self.radar_angle = [[345.0, 375.0], [15.0, 45.0], [45.0, 90.0], [90.0, 150.0], [150.0, 210.0], [210.0, 270.0], [270.0, 315.0], [315.0, 345.0]]

        #This variable contains the angular position of radars (here is an EPUCk robot)
        self.radar_position = [0.0]
        for i in range(1,len(self.radar_angle)):
            self.radar_position.append((self.radar_angle[i][0]+self.radar_angle[i][1])/2)

        # This variable contains the message received by each radar
        # Later, only one element of the list can be other than None, because the robots must receive only one message at each time step
        # For now, the information is the angle with which the sender sent the message to this agent
        self.radar = [None] * len(self.radar_angle)


    def create_net_inputs(self):
        """
        The function to return the ANN input values from the agent.
        """

        for i,msg in enumerate(self.radar):
            if msg != None:
                angle_of_reception = self.radar_position[i]
                angle_of_emission = msg
                break

        self.msg_rcv_1

        inputs = [self.mode, angle_of_reception/360.0, angle_of_emission/360.0, self.msg_rcv_1, self.msg_rcv_2]        
        return inputs
    

    def apply_outputs(self, outputs):
        """
        The function to change behaviour of agent according to the outputs of the ANN
        """

        # Change mode of the agent
        if outputs[0] < 0.5:
            self.mode = 0
        else:
            self.mode = 1

        # Change angular velocity of the agent
        new_ang_vel = outputs[1]-0.5
        if new_ang_vel >= MAX_ANGULAR_VEL:
            self.angular_vel = MAX_ANGULAR_VEL
        elif new_ang_vel < -MAX_ANGULAR_VEL:
            self.angular_vel = -MAX_ANGULAR_VEL
        else:
            self.angular_vel = new_ang_vel

        # Change heading of the agent
        self.heading += 10* self.angular_vel # an angular velocity of 1 corresponds to 10 degrees per step
        if self.heading < 0:
            self.heading += 360
        elif self.heading >= 360:
            self.heading -= 360

        if outputs[2]<1/6:
            self.msg_sen_1=0
        elif outputs[2]<3/6:
            self.msg_sen_1=1/3
        elif outputs[2]<5/6:
            self.msg_sen_1=2/3
        else:
            self.msg_sen_1=1

        if outputs[3]<1/6:
            self.msg_sen_2=0
        elif outputs[3]<3/6:
            self.msg_sen_2=1/3
        elif outputs[3]<5/6:
            self.msg_sen_2=2/3
        else:
            self.msg_sen_2=1

    def update_radar(self, sender):
        """
        Updates the list radar, by initializing it and putting the msg in the right element of the list
        """
        # Initializing radar list
        for radar in self.radar:
            radar = None

        # Calculating geometric angle from sender to receiver
        vect = geometry.Point(self.location.x-sender.location.x, self.location.y-sender.location.y)
        angle = vect.angle()

        rel_angle = self.heading-angle

        self.radar[self.find_radar_index(rel_angle)] = sender.calculate_msg_to(self)

        # Transmission of the two-digits message
        if sender.mode == 0:
            self.msg_rcv_1 = sender.msg_rcv_1
            self.msg_rcv_2 = sender.msg_rcv_2
        elif sender.mode == 1:
            self.msg_rcv_1 = sender.msg_sen_1
            self.msg_rcv_2 = sender.msg_sen_2
        else:
            print("ERROR: mode not equal 1 or 0")
        


    def calculate_msg_to(self, aimed_robot):
        """
        Calculates the position of the radar used by sender to send a message to an aimed robot
        """

        # Calculating geometric angle from sender to receiver
        vect = geometry.Point(aimed_robot.location.x-self.location.x, aimed_robot.location.y-self.location.y)
        angle = vect.angle()

        # Substract heading of sender to have the relative angle
        rel_angle = angle - self.heading
        if rel_angle <0:
            rel_angle += 360
        elif rel_angle >360:
            rel_angle -= 360

        return self.radar_position[self.find_radar_index(rel_angle)]

    def find_radar_index(self, angle):
        """
        Gives the index of the radar covering this angle
        """  

        for i,span in enumerate(self.radar_angle):
            if angle > span[0] and angle < span[1]:
                return i
        else:
            return 0

        print("ERROR : find_radar_index") 


    def individual_fitness(self, avg_heading):
        """
        Returns the individual fitness of the agent, in an environment with a specific avg_heading
        Argument : avg_heading, average angle of the heading in the evironment (in degrees)
        """

        tetaR = self.heading

        # Computing absolute value term
        abs_term = abs(geometry.deg_to_rad(tetaR)-geometry.deg_to_rad(avg_heading))

        # Computing min term
        min_term = min(2*math.pi - abs_term, abs_term)

        # Computing left term
        left_term = 1-min_term/math.pi

        # Computing right term
        right_term = 1-abs(self.angular_vel)

        return left_term * right_term