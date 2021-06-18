from matplotlib import pyplot as plt
import moviepy.video.io.ImageSequenceClip as Movie_maker
from shapely.geometry import Point, Polygon, LineString, box
from environment import Environment, plot_environment, plot_line, plot_poly
from math import sqrt
import os


def draw_results(algo_name, path, V, E, env, bounds, object_radius, resolution, start_pose, goal_region, elapsed_time, iteration):
    """
    Plots the path from start node to goal region as well as the graph (or tree) searched with the Sampling Based Algorithms.

    Args:
        algo_name (str): The name of the algorithm used (used as title of the plot)
        path (list<(float,float), (float,float)>): The sequence of coordinates traveled to reach goal from start node
        V (set<(float, float)>): All nodes in the explored graph/tree
        E (set<(float,float), (float, float)>): The set of all edges considered in the graph/tree
        env (yaml environment): 2D yaml environment for the path planning to take place
        bounds (int, int int int): min x, min y, max x, max y of the coordinates in the environment.
        object_radius (float): radius of our object.
        resolution (int): Number of segments used to approximate a quarter circle around a point.
        start_pose(float,float): Coordinates of initial point of the path.
        goal_region (Polygon): A polygon object representing the end goal.
        elapsed_time (float): Time it took for the algorithm to run

    Return:
        None

    Action:
        Plots a path using the environment module.
    """
    originalPath, pruningPath = path
    graph_size = len(V)
    path_size = len(originalPath)
    # Calculate path length
    path_length1 = 0.0
    path_length2 = 0.0
    for i in range(len(originalPath)-1):
        path_length1 += euclidian_dist(originalPath[i], originalPath[i+1])
    for i in range(len(pruningPath)-1):
        path_length2 += euclidian_dist(pruningPath[i], pruningPath[i+1])

    # Create title with descriptive information based on environment, path length, and elapsed_time
    title = algo_name + "\nPath Length: " + \
        str([path_length1, path_length2]) + \
        "\n Runtime(s)= " + str(elapsed_time)

    # Plot environment
    fig = plt.figure()
    env_plot = plot_environment(fig, env, bounds)
    # Add title
    env_plot.set_title(title)
    # Plot goal
    plot_poly(env_plot, goal_region, 'green')
    # Plot start
    buffered_start_vertex = Point(start_pose).buffer(object_radius, resolution)
    plot_poly(env_plot, buffered_start_vertex, 'red')

    # Plot Edges explored by ploting lines between each edge
    for edge in E:
        line = LineString([edge[0], edge[1]])
        plot_line(env_plot, line)

    # Plot path
    plot_path(env_plot, originalPath, object_radius, 'black', iteration)
    #plot_path(env_plot, pruningPath, object_radius, 'red')


def euclidian_dist(point1, point2):
    return sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def plot_path(env_plot, path, object_radius, colorset, iteration):
    # Plots path by taking an enviroment plot and ploting in red the edges that form part of the path
    line = LineString(path)
    x, y = line.xy
    env_plot.plot(x[:iteration], y[:iteration],
                  color=colorset,
                  linewidth=3,
                  solid_capstyle='round',
                  zorder=1)


def draw_animation(algo_name, path, V, E, env, bounds, object_radius, resolution, start_pose, goal_region, elapsed_time):
    """
    Plots the path from start node to goal region as well as the graph (or tree) searched with the Sampling Based Algorithms.

    Args:
        algo_name (str): The name of the algorithm used (used as title of the plot)
        path (list<(float,float), (float,float)>): The sequence of coordinates traveled to reach goal from start node
        V (set<(float, float)>): All nodes in the explored graph/tree
        E (set<(float,float), (float, float)>): The set of all edges considered in the graph/tree
        env (yaml environment): 2D yaml environment for the path planning to take place
        bounds (int, int int int): min x, min y, max x, max y of the coordinates in the environment.
        object_radius (float): radius of our object.
        resolution (int): Number of segments used to approximate a quarter circle around a point.
        start_pose(float,float): Coordinates of initial point of the path.
        goal_region (Polygon): A polygon object representing the end goal.
        elapsed_time (float): Time it took for the algorithm to run

    Return:
        None

    Action:
        Plots a path using the environment module.
    """
    originalPath, pruningPath = path
    graph_size = len(V)
    path_size = len(originalPath)
    # Calculate path length
    path_length1 = 0.0
    path_length2 = 0.0
    for i in range(len(originalPath)-1):
        path_length1 += euclidian_dist(originalPath[i], originalPath[i+1])
    for i in range(len(pruningPath)-1):
        path_length2 += euclidian_dist(pruningPath[i], pruningPath[i+1])
    fig = plt.figure()
    for iteration in range(1, path_size+2):
        # Plot environment
        env_plot = plot_environment(fig, env, bounds)
        # Plot goal
        plot_poly(env_plot, goal_region, 'green')
        # Plot start
        buffered_start_vertex = Point(
            start_pose).buffer(object_radius, resolution)
        plot_poly(env_plot, buffered_start_vertex, 'red')

        # Plot Edges explored by ploting lines between each edge
        for edge in E:
            line = LineString([edge[0], edge[1]])
            plot_line(env_plot, line)

        # Plot path
        plot_path(env_plot,
                  originalPath,
                  object_radius,
                  'black',
                  iteration)
        #plot_path(env_plot, pruningPath, object_radius, 'red')
        iteration = str(iteration).zfill(len(str(graph_size)))
        fig.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1)
        env_plot.axis("off")
        fig.savefig("{}.png".format(iteration))
        fig.clf()
    create_movie()


def create_movie():
    """
    Funcion que crea el gif a partir de las graficas diarias
    """
    filenames = sorted(os.listdir())
    filenames = [file for file in filenames if file.endswith(".png")]

    output_file = "animation.mp4"
    movie = Movie_maker.ImageSequenceClip(filenames,
                                          fps=20)
    movie.write_videofile(output_file)
    os.system("rm *.png")
