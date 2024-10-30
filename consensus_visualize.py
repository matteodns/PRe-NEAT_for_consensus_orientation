#Copyright (c) 2007-2011, cesar.gomes and mirrorballu2
#Copyright (c) 2015-2017, CodeReclaimers, LLC
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
#following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#disclaimer in the documentation and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
#derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import print_function

import copy
import warnings
import random
import argparse
import os
import math
import imageio

import graphviz
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

import geometry
# import agent
# import consensus_environment as env

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, directory=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, directory, view=view)

    return dot


def animate_experiment(consensus_env, robot_orientation_list, genome, dirname, width=100, height=100, fig_height=5):
    """
    The function to create an animation of the experiment
    /!\ robot_orientation_list and genome must be related
    """

    arrow_size = 10
    dir_images = os.path.join(dirname, 'images_gif')
    os.makedirs(dir_images, exist_ok=True)

    # Animate path
    for i in range(len(robot_orientation_list[0])):
        
        # Initialize plotting
        fig, ax = plt.subplots()
        fig.set_dpi(100)
        fig_width = fig_height * (float(width)/float(height)) - 0.5
        fig.set_size_inches(fig_width, fig_height)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

        ax.set_title('Image n° %03d / %03d' % (i , len(robot_orientation_list[0])))

        # Draw image
        for r, robot_heading in enumerate(robot_orientation_list):
            robot_location = consensus_env.agent_list[r].location
            arrow = plt.arrow(robot_location.x,
                              robot_location.y,
                              arrow_size * math.cos(geometry.deg_to_rad(robot_heading[i])),
                              arrow_size * math.sin(geometry.deg_to_rad(robot_heading[i])),
                              width=1.5,
                              length_includes_head=True)
            ax.add_patch(arrow)

        # Draw env
        _draw_env_(consensus_env, ax)

        # Invert Y axis to have coordinates origin at the top left and turn off axis rendering
        ax.invert_yaxis()
        ax.axis('off')
        

        #save_image
        filename_image = os.path.join(dir_images, 'image %03d' % i)
        plt.savefig(filename_image)

        plt.close()

    # Create gif
    path_gif = os.path.join(dirname, 'genome_%d_animation.gif' % genome.key)
    with imageio.get_writer(path_gif, mode='I') as writer:
        for filename in os.listdir(dir_images):
            path_image = os.path.join(dir_images, filename)
            image = imageio.imread(path_image)
            writer.append_data(image)

            #os.remove(path_image)

    #os.removedirs(dir_images)

def plot_headings(robot_orientation_list, genome, dirname=None, view=False):
    """
    Plots the difference to average heading of every robots step by step
    """
    steps = range(len(robot_orientation_list[0]))
    avg_heading = []

    for i in steps:
        headings_in_this_step = np.array([robot_orientation_list[j][i] for j in range(len(robot_orientation_list))])
        avg_heading.append(headings_in_this_step.mean())

    avg_heading_array = np.array(avg_heading)
    #plt.plot(steps, avg_heading, 'r-', label="average heading")


    for i,robot_heading in enumerate(robot_orientation_list):
        robot_heading_array = np.array(robot_heading)
        plt.plot(steps, robot_heading_array - avg_heading_array, 'b-', label="robot_%d" % i)

    plt.title("Robots headings for genome n°%d" % genome.key)
    plt.xlabel("Robots headings difference to average")
    plt.ylabel("Steps")
    plt.grid()
    
    if dirname!=None:
        filename = os.path.join(dirname, 'robots_headings.svg')
        plt.savefig(filename)
    if view:
        plt.show()
    
    plt.close()


def _draw_env_(env, ax):
    """
    The function to draw the walls of environment and the points representing robots
    """

    #draw env walls
    line = plt.Line2D((0,0),(0,env.length), lw=1.5)
    ax.add_line(line)
    line = plt.Line2D((0,0),(env.height,0), lw=1.5)
    ax.add_line(line)
    line = plt.Line2D((env.height,0),(env.height,env.length), lw=1.5)
    ax.add_line(line)
    line = plt.Line2D((env.height,env.length),(0,env.length), lw=1.5)
    ax.add_line(line)

    for robot in env.agent_list:
        circle = plt.Circle((robot.location.x, robot.location.y), radius=2.5)
        ax.add_patch(circle)