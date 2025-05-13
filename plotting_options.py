import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from numpy.ma import masked_array
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon, LineString
import numpy as np
from numpy.ma import masked_array
from scipy.interpolate import griddata
from collections import namedtuple, deque
import statistics

 
import numpy as np

RC      = 10  #mu m 
RN      = 4   #mu m 


def plot_reward_evolution(reward_history):
    """
    Plot the evolution of the reward over episodes for multiple agents and their mean.

    Args:
        reward_history (list of dict): A list of dictionaries, each containing rewards for different agents.
    """
    plt.figure(figsize=(10, 5))

    agents = list(reward_history[0].keys())
    episode_range = range(1, len(reward_history) + 1)

    episode_rewards = np.array([[episode[agent] for agent in agents] for episode in reward_history])
    episode_means = np.mean(episode_rewards, axis=1)

    for agent in agents:
        agent_rewards = episode_rewards[:, agents.index(agent)]
        plt.plot(episode_range, agent_rewards, label=agent)

    plt.plot(episode_range, episode_means, label='Mean', color='r', linestyle='-', linewidth=2)

    plt.title('Reward Evolution Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

    

def cell_trajectory_pressure(cell_pos,reward,file_path): 
    
    # Initialize empty lists to store data
    xp = [] 
    yp = [] 
    pressure = []

    # Read data from the file
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split(',')
            xp.append(float(parts[1]) * 1e6)  # Convert to micrometers
            yp.append(float(parts[2]) * 1e6)  # Convert to micrometers
            pressure.append(float(parts[3]))

    # Convert data to NumPy arrays as float32
    xp = np.array(xp, dtype=np.float32)
    yp = np.array(yp, dtype=np.float32)
    pressure = np.array(pressure, dtype=np.float32)

    ax = [-15, 15, 15, 36, 36, 51, 51, 36, 36, 15, 15, -15]
    ay = [-10.5, -10.5, -25.5, -25.5, -10.5, -10.5, 10.5, 10.5, 25.5, 25.5, 10.5, 10.5]


    # Create a meshgrid of coordinates within the irregular boundary
    coarse_X, coarse_Y = np.meshgrid(np.linspace(min(ax), max(ax), 500), np.linspace(min(ay), max(ay), 500))

    # Initialize an array for the interpolated pressure
    coarse_pressure = np.zeros_like(coarse_X)

    # Interpolate pressure values only within the irregular boundary
    points_within_hull = np.column_stack((xp, yp))
    pressure_within_hull = pressure
    coarse_pressure_within_boundary = griddata(points_within_hull, pressure_within_hull, (coarse_X, coarse_Y), method='linear', fill_value=0)

    # Create a mask for points outside the irregular boundary
    boundary_polygon = Polygon(list(zip(ax, ay)), closed=True)
    mask = ~boundary_polygon.contains_points(np.column_stack((coarse_X.ravel(), coarse_Y.ravel()))).reshape(coarse_X.shape)

    # Create a masked array to set values outside the boundary to np.nan
    coarse_pressure = masked_array(coarse_pressure_within_boundary, mask=mask)

    # Create a contour plot of the interpolated pressure within the irregular boundary
    plt.figure(figsize=(10, 5))
    contours = plt.contourf(coarse_X, coarse_Y, coarse_pressure, cmap='viridis', levels=100)
    plt.colorbar(contours)
    plt.title('Pressure Contours within Irregular Boundary')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')


    # plt.plot(ax + [ax[0]], ay + [ay[0]], linestyle='-', color='b')

    # Extract x and y coordinates for plotting
    x_pos, y_pos = zip(*cell_pos)
    # Plot the evolution of the agent's position
    plt.plot(x_pos, y_pos, linestyle='-', color='black')
    # Add a circle at the final position
    final_position = cell_pos[-1]
    circle = plt.Circle(final_position, RC, color='black', fill=False, label='Final Position')
    plt.gca().add_patch(circle)
    plt.scatter([x_pos[0]], [y_pos[0]], color='red', marker='x', s=100, label='Initial Position')  # Add initial position marker
    plt.title('Reward: '+str(round(reward,4)))
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.show()    
    
    
    

def plot_training_geoms(cell_pos, reward):
    # Geometries
    ax = [-15, 15, 15, 36, 36, 51, 51, 36, 36, 15, 15, -15]
    ay = [-10.5, -10.5, -25.5, -25.5, -10.5, -10.5, 10.5, 10.5, 25.5, 25.5, 10.5, 10.5]

    # Get the number of geoms
    num_geoms = len(cell_pos)

    # Create subplots with 'num_geoms' rows and 1 column
    fig, axs = plt.subplots(num_geoms, 1, figsize=(8, 4 * num_geoms))

    # Iterate over each geom and plot position evolution
    for i, geom in enumerate(cell_pos):
        # Pressure representation
        # Initialize empty lists to store data
        x_pos = []
        y_pos = []
        xp    = [] 
        yp    = [] 
        pressure = []

        # Read data from the file
        file_path = f'Data\Pressure-{geom}.txt'
        with open(file_path, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                parts = line.strip().split(',')
                xp.append(float(parts[1]) * 1e6)  # Convert to micrometers
                yp.append(float(parts[2]) * 1e6)  # Convert to micrometers
                pressure.append(float(parts[3]))

        # Convert data to NumPy arrays as float32
        xp = np.array(xp, dtype=np.float32)
        yp = np.array(yp, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)

        # Create a meshgrid of coordinates within the irregular boundary
        coarse_X, coarse_Y = np.meshgrid(np.linspace(min(ax), max(ax), 500), np.linspace(min(ay), max(ay), 500))

        # Initialize an array for the interpolated pressure
        coarse_pressure = np.zeros_like(coarse_X)

        # Interpolate pressure values only within the irregular boundary
        points_within_hull = np.column_stack((xp, yp))
        pressure_within_hull = pressure
        coarse_pressure_within_boundary = griddata(points_within_hull, pressure_within_hull, (coarse_X, coarse_Y),
                                                   method='linear', fill_value=0)

        # Create a mask for points outside the irregular boundary
        boundary_polygon = Polygon(list(zip(ax, ay)), closed=True)
        mask = ~boundary_polygon.contains_points(np.column_stack((coarse_X.ravel(), coarse_Y.ravel()))).reshape(
            coarse_X.shape)

        # Create a masked array to set values outside the boundary to np.nan
        coarse_pressure = np.ma.masked_array(coarse_pressure_within_boundary, mask=mask)

        contours = axs[i].contourf(coarse_X, coarse_Y, coarse_pressure, cmap='viridis', levels=100)
        fig.colorbar(contours, ax=axs[i])

        x_pos = cell_pos[geom]['x']
        y_pos = cell_pos[geom]['y']

        title = f"{geom} - reward: {round(reward[geom], 4)}"
        axs[i].set_title(title)
        # Evolution of the position
        axs[i].plot(x_pos, y_pos, linestyle='-', color='black')
        # Add a circle at the final position
        circle = plt.Circle([x_pos[-1], y_pos[-1]], RC, color='black', fill=False, label='Final Position')
        axs[i].add_patch(circle)
        # Add initial position marker
        axs[i].scatter([x_pos[0]], [y_pos[0]], color='red', marker='x', s=100, label='Initial Position')
        axs[i].set_xlabel('X ($\mu m$)')
        axs[i].set_ylabel('Y ($\mu m$)')

    # Adjust subplot spacing
    plt.tight_layout()

    # Show the plots
    plt.show()



def plot_validation(cell_pos, reward):
    geom = list(cell_pos.keys())[0]
    
    # Geometries
    if geom == 'straight' or geom == 'bot' or geom == 'top':
        ax = [-15, 15, 15, 36, 36, 51, 51, 36, 36, 15, 15, -15]
        ay = [-10.5, -10.5, -25.5, -25.5, -10.5, -10.5, 10.5, 10.5, 25.5, 25.5, 10.5, 10.5]
    elif geom == 'deadend':
        ax = [-15, 15, 15, -47, -47, 14, 14, -26, -26, 36, 36, 112.21, 112.21,
              174.21, 174.21, 134.21, 134.21, 195.21, 195.21, 133.21, 133.21,
              163.21, 163.21, 133.21, 133.21, 15, 15, -15, -15]
        ay = [-10.5, -10.5, -35.5, -35.5, -97.5, -97.5, -76.5, -76.5, -56.5,
              -56.5, 14.5, 14.5, -56.5, -56.5, -76.5, -76.5, -97.5, -97.5, -35.5,
              -35.5, -10.5, -10.5, 10.5, 10.5, 35.5, 35.5, 10.5, 10.5, -10.5]
    elif geom == 'shortdeadend':    
        ax = [-15, 15, 15, 36, 36, 112.21, 112.21, 133.21, 133.21, 163.21, 163.21,
              133.21, 133.21, 15, 15, -15, -15]
        ay = [-10.5, -10.5, -35.5, -35.5, 14.5, 14.5, -35.5, -35.5, -10.5, -10.5, 10.5,
              10.5, 35.5, 35.5, 10.5, 10.5, -10.5]
    elif geom == 'square':
        ax = [-21,21,21,147,147,189,189,147,147,21,21,-21,-21, 42,126,126,42,42]
        ay = [-10.5,-10.5,-63,-63,-10.5,-10.5,10.5,10.5,63,63,10.5,10.5,-10.5, -42,-42,42,42,-42]
    elif geom == 'twisted':
        ax = [-21,21,21,-147,-147,-84,-84,-63,-63,0,0,21,21,84,84,105,105,168,168,189,189,252,252,\
            273,273,336,336,168,168,210,210,168,168,21,21,-21,-21,42,-126,-126,-105,-105,-42,-42,-21,\
            -21,42,42,63,63,126,126,147,147,210,210,231,231,294,294,315,315,147,147,42,42]
        ay = [-10.5,-10.5,-52.5,-52.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,\
            -199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,\
            -52.5,-52.5,-10.5,-10.5,10.5,10.5,52.5,52.5,10.5,10.5,-10.5,-73.5,-73.5,-178.5,-178.5,-115.5,\
            -115.5,-178.5,-178.5,-115.5,-115.5,-178.5,-178.5,-115.5,-115.5,-178.5,-178.5,-115.5,-115.5,-178.5,\
            -178.5,-115.5,-115.5,-178.5,-178.5,-73.5,-73.5,31.5,31.5,-73.5]
    elif geom == 'curved':  
        ax = [-21,21,21,105,105,147,147,231,231,273,273,357,357,399,399,357,
                357,346.8684,317.6955,273,218.1729,159.8271,105,60.3045,31.1316,21,21,-21,-21,
                42,84,84,168,168,210,210,294,294,336,337,328.1348,302.6085,263.5,215.5263,164.4737,116.5,77.3915,51.8652,43,42]
        ay = [-10.5,-10.5,-105,-105,-63,-63,-105,-105,-63,-63,-105,-105,-10.5,-10.5,10.5,10.5,136,193.4594,243.9883, 
                281.4923,301.4477,301.4477,281.4923,243.9883,193.4594,136,10.5,10.5,-10.5,
                -84,-84,-42,-42,-84,-84,-42,-42,-84,-84,136.5,186.7770,230.9898,263.8057,281.2667,281.2667,263.8057,230.9898,186.7770,136.5,-84]
    
    else: 
        return print('No geometry found')   
    
    # Create subplots with 'num_geoms' rows and 1 column
    plt.figure(figsize=(10, 5))

    # Pressure representation
    xp, yp, pressure = [], [], []

    # Read data from the file
    file_path = f'Data\Pressure-{geom}.txt'
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split(',')
            xp.append(float(parts[1]) * 1e6)  # Convert to micrometers
            yp.append(float(parts[2]) * 1e6)  # Convert to micrometers
            pressure.append(float(parts[3]))

    # Convert data to NumPy arrays as float32
    xp = np.array(xp, dtype=np.float32)
    yp = np.array(yp, dtype=np.float32)
    pressure = np.array(pressure, dtype=np.float32)
    # Create a meshgrid of coordinates within the irregular boundary
    coarse_X, coarse_Y = np.meshgrid(np.linspace(min(ax), max(ax), 500), np.linspace(min(ay), max(ay), 500))

    # Interpolate pressure values only within the irregular boundary
    points_within_hull = np.column_stack((xp, yp))
    coarse_pressure_within_boundary = griddata(points_within_hull, pressure, (coarse_X, coarse_Y),
                                               method='linear', fill_value=0)

    # Create a polygon from the boundary points
    boundary_polygon = Polygon(list(zip(ax, ay)))
    mask = ~np.array([
        boundary_polygon.contains(Point(x, y))
        for x, y in zip(coarse_X.ravel(), coarse_Y.ravel())
    ]).reshape(coarse_X.shape)

    # Create a masked array to set values outside the boundary to np.nan
    coarse_pressure = np.ma.masked_array(coarse_pressure_within_boundary, mask=mask)
    coarse_pressure = 2 * ((coarse_pressure - np.min(coarse_pressure)) / (np.max(coarse_pressure) - np.min(coarse_pressure))) - 1
    contours = plt.contourf(coarse_X, coarse_Y, coarse_pressure, cmap='viridis', levels=100, vmin=-1, vmax=1)
    cbar = plt.colorbar(contours)
    cbar.set_ticks([-1, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, 1]) 
    # Plot the cell trajectory
    x_pos = cell_pos[geom]['x']
    y_pos = cell_pos[geom]['y']
    title = f"{geom} - reward: {round(reward[geom], 4)}"
    plt.title(title)
    plt.plot(x_pos, y_pos, linestyle='-', color='black')

    # Add a circle at the final position
    plt.scatter([x_pos[0]], [y_pos[0]], color='red', marker='x', s=100, label='Initial Position')
    circle = plt.Circle([x_pos[-1], y_pos[-1]], RC, color='black', fill=False, label='Final Position')
    plt.gca().add_patch(circle)
    circle = plt.Circle([x_pos[-1], y_pos[-1]], RN, color='black', fill=False)
    plt.gca().add_patch(circle)
    plt.xlabel('X ($\mu m$)')
    plt.ylabel('Y ($\mu m$)')

    plt.tight_layout()
    plt.show()

    
    
    
    
def plot_geometry(geom):

    """
    Plot the geometry

    Args:
        geom (string): name of the geometry
    """
    
    # Geometries
    if geom == 'straight' or geom == 'bot' or geom == 'top':
        ax = [-15, 15, 15, 36, 36, 51, 51, 36, 36, 15, 15, -15]
        ay = [-10.5, -10.5, -25.5, -25.5, -10.5, -10.5, 10.5, 10.5, 25.5, 25.5, 10.5, 10.5]
    elif geom == 'deadend':
        ax=[-15,15,15,-47,-47,14,14,-26,-26,36,36,112.21,112.21,\
            174.21,174.21,134.21,134.21,195.21,195.21,133.21,133.21,\
            163.21,163.21,133.21,133.21,15,15,-15,-15]
        ay=[-10.5,-10.5,-35.5,-35.5,-97.5,-97.5,-76.5,-76.5,-56.5,\
            -56.5,14.5,14.5,-56.5,-56.5,-76.5,-76.5,-97.5,-97.5,-35.5,\
            -35.5,-10.5,-10.5,10.5,10.5,35.5,35.5,10.5,10.5,-10.5]
    elif geom == 'shortdeadend':    
        ax=[-15,15,15,36,36,112.21,112.21,133.21,133.21,163.21,163.21,\
            133.21,133.21,15,15,-15,-15]
        ay=[-10.5,-10.5,-35.5,-35.5,14.5,14.5,-35.5,-35.5,-10.5,-10.5,10.5,\
            10.5,35.5,35.5,10.5,10.5,-10.5]
    elif geom == 'twisted': 
        ax=[-21,21,21,-147,-147,-84,-84,-63,-63,0,0,21,21,84,84,105,105,168,168,189,189,252,252,\
            273,273,336,336,168,168,210,210,168,168,21,21,-21,-21,42,-126,-126,-105,-105,-42,-42,-21,\
            -21,42,42,63,63,126,126,147,147,210,210,231,231,294,294,315,315,147,147,42,42]
        ay=[-10.5,-10.5,-52.5,-52.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,\
            -199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,\
            -52.5,-52.5,-10.5,-10.5,10.5,10.5,52.5,52.5,10.5,10.5,-10.5,-73.5,-73.5,-178.5,-178.5,-115.5,\
            -115.5,-178.5,-178.5,-115.5,-115.5,-178.5,-178.5,-115.5,-115.5,-178.5,-178.5,-115.5,-115.5,-178.5,\
            -178.5,-115.5,-115.5,-178.5,-178.5,-73.5,-73.5,31.5,31.5,-73.5]

    else: 
        return print('No geometry found')   
    
    # Create subplots with 'num_geoms' rows and 1 column
    plt.figure(figsize=(10, 5))

    # Iterate over each geom and plot position evolution
    # Pressure representation
    # Initialize empty lists to store data
    xp    = [] 
    yp    = [] 
    pressure = []

    # Read data from the file
    file_path = f'Data\Pressure-{geom}.txt'
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split(',')
            xp.append(float(parts[1]) * 1e6)  # Convert to micrometers
            yp.append(float(parts[2]) * 1e6)  # Convert to micrometers
            pressure.append(float(parts[3]))

    # Convert data to NumPy arrays as float32
    xp = np.array(xp, dtype=np.float32)
    yp = np.array(yp, dtype=np.float32)
    pressure = np.array(pressure, dtype=np.float32)

    # Create a meshgrid of coordinates within the irregular boundary
    coarse_X, coarse_Y = np.meshgrid(np.linspace(min(ax), max(ax), 500), np.linspace(min(ay), max(ay), 500))

    # Initialize an array for the interpolated pressure
    coarse_pressure = np.zeros_like(coarse_X)

    # Interpolate pressure values only within the irregular boundary
    points_within_hull = np.column_stack((xp, yp))
    pressure_within_hull = pressure
    coarse_pressure_within_boundary = griddata(points_within_hull, pressure_within_hull, (coarse_X, coarse_Y),
                                                method='linear', fill_value=0)

    # Create a mask for points outside the irregular boundary
    boundary_polygon = Polygon(list(zip(ax, ay)), closed=True)
    mask = ~boundary_polygon.contains_points(np.column_stack((coarse_X.ravel(), coarse_Y.ravel()))).reshape(
        coarse_X.shape)

    # Create a masked array to set values outside the boundary to np.nan
    coarse_pressure = np.ma.masked_array(coarse_pressure_within_boundary, mask=mask)

    contours = plt.contourf(coarse_X, coarse_Y, coarse_pressure, cmap='viridis', levels=100)
    plt.colorbar(contours)

    # Adjust subplot spacing
    plt.tight_layout()

    # Show the plots
    plt.show()
    
    
    
def plot_validation_deformed_cell(cell_pos, reward):
    geom = list(cell_pos.keys())[0]
    
    # Define geometry boundaries
    if geom in ['straight', 'bot', 'top']:
        ax = [-15, 15, 15, 36, 36, 51, 51, 36, 36, 15, 15, -15]
        ay = [-10.5, -10.5, -25.5, -25.5, -10.5, -10.5, 10.5, 10.5, 25.5, 25.5, 10.5, 10.5]
    elif geom == 'deadend':
        ax = [-15, 15, 15, -47, -47, 14, 14, -26, -26, 36, 36, 112.21, 112.21,
              174.21, 174.21, 134.21, 134.21, 195.21, 195.21, 133.21, 133.21,
              163.21, 163.21, 133.21, 133.21, 15, 15, -15, -15]
        ay = [-10.5, -10.5, -35.5, -35.5, -97.5, -97.5, -76.5, -76.5, -56.5,
              -56.5, 14.5, 14.5, -56.5, -56.5, -76.5, -76.5, -97.5, -97.5, -35.5,
              -35.5, -10.5, -10.5, 10.5, 10.5, 35.5, 35.5, 10.5, 10.5, -10.5]
    elif geom == 'shortdeadend':
        ax = [-15, 15, 15, 36, 36, 112.21, 112.21, 133.21, 133.21, 163.21, 163.21,
              133.21, 133.21, 15, 15, -15, -15]
        ay = [-10.5, -10.5, -35.5, -35.5, 14.5, 14.5, -35.5, -35.5, -10.5, -10.5, 10.5,
              10.5, 35.5, 35.5, 10.5, 10.5, -10.5]
    else:
        return print('No geometry found')

    # Create subplots
    plt.figure(figsize=(10, 5))

    # Read pressure data
    file_path = f'Data\Pressure-{geom}.txt'
    xp, yp, pressure = [], [], []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            parts = line.strip().split(',')
            xp.append(float(parts[1]) * 1e6)
            yp.append(float(parts[2]) * 1e6)
            pressure.append(float(parts[3]))

    xp = np.array(xp, dtype=np.float32)
    yp = np.array(yp, dtype=np.float32)
    pressure = np.array(pressure, dtype=np.float32)

    # Create meshgrid and interpolate pressure
    coarse_X, coarse_Y = np.meshgrid(np.linspace(min(ax), max(ax), 500), np.linspace(min(ay), max(ay), 500))
    coarse_pressure_within_boundary = griddata(
        (xp, yp), pressure, (coarse_X, coarse_Y), method='linear', fill_value=0
    )
    boundary_polygon = Polygon(list(zip(ax, ay)))

    mask = ~np.array([
        boundary_polygon.contains(Point(x, y))
        for x, y in zip(coarse_X.ravel(), coarse_Y.ravel())
    ]).reshape(coarse_X.shape)

    coarse_pressure = np.ma.masked_array(coarse_pressure_within_boundary, mask=mask)

    # Plot pressure field
    contours = plt.contourf(coarse_X, coarse_Y, coarse_pressure, cmap='viridis', levels=100)
    plt.colorbar(contours)

    # Plot cell positions
    x_pos = cell_pos[geom]['x']
    y_pos = cell_pos[geom]['y']
    plt.plot(x_pos, y_pos, linestyle='-', color='black')
    plt.scatter([x_pos[0]], [y_pos[0]], color='red', marker='x', s=100)

    # Plot deformed circle within the boundary
    circle_x = RC * np.cos(np.linspace(0, 2 * np.pi, 100)) + x_pos[-1]
    circle_y = RC * np.sin(np.linspace(0, 2 * np.pi, 100)) + y_pos[-1]

    # Filter points within the boundary
    circle_points = np.column_stack((circle_x, circle_y))
    inside_mask = np.array([boundary_polygon.contains(Point(p)) for p in circle_points])
    filtered_circle = circle_points[inside_mask]

    if len(filtered_circle) > 0:
        # Create a patch from the filtered circle
        path = Path(filtered_circle, closed=False)
        patch = PathPatch(path, facecolor='none', edgecolor='blue', lw=2)
        plt.gca().add_patch(patch)
        
    circle = plt.Circle([x_pos[-1], y_pos[-1]], RN, color='black', fill=False, label='Final Position')
    plt.gca().add_patch(circle)
    # Add labels and finalize plot
    title = f"{geom} - reward: {round(reward[geom], 4)}"
    plt.title(title)
    plt.xlabel('X ($\mu m$)')
    plt.ylabel('Y ($\mu m$)')
    plt.tight_layout()
    plt.show()
    
    
    
def plot_validation_wall_deformed_cell(cell_pos, reward):
    geom = list(cell_pos.keys())[0]
    
    # Define geometry boundaries
    if geom in ['straight', 'bot', 'top']:
        ax = [-15, 15, 15, 36, 36, 51, 51, 36, 36, 15, 15, -15]
        ay = [-10.5, -10.5, -25.5, -25.5, -10.5, -10.5, 10.5, 10.5, 25.5, 25.5, 10.5, 10.5]
    elif geom == 'deadend':
        ax = [-15, 15, 15, -47, -47, 14, 14, -26, -26, 36, 36, 112.21, 112.21,
              174.21, 174.21, 134.21, 134.21, 195.21, 195.21, 133.21, 133.21,
              163.21, 163.21, 133.21, 133.21, 15, 15, -15, -15]
        ay = [-10.5, -10.5, -35.5, -35.5, -97.5, -97.5, -76.5, -76.5, -56.5,
              -56.5, 14.5, 14.5, -56.5, -56.5, -76.5, -76.5, -97.5, -97.5, -35.5,
              -35.5, -10.5, -10.5, 10.5, 10.5, 35.5, 35.5, 10.5, 10.5, -10.5]
    elif geom == 'shortdeadend':
        ax = [-15, 15, 15, 36, 36, 112.21, 112.21, 133.21, 133.21, 163.21, 163.21,
              133.21, 133.21, 15, 15, -15, -15]
        ay = [-10.5, -10.5, -35.5, -35.5, 14.5, 14.5, -35.5, -35.5, -10.5, -10.5, 10.5,
              10.5, 35.5, 35.5, 10.5, 10.5, -10.5]
    else:
        return print('No geometry found')

    # Create subplots
    plt.figure(figsize=(10, 5))

    # Read pressure data
    file_path = f'Data\Pressure-{geom}.txt'
    xp, yp, pressure = [], [], []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            parts = line.strip().split(',')
            xp.append(float(parts[1]) * 1e6)
            yp.append(float(parts[2]) * 1e6)
            pressure.append(float(parts[3]))

    xp = np.array(xp, dtype=np.float32)
    yp = np.array(yp, dtype=np.float32)
    pressure = np.array(pressure, dtype=np.float32)

    # Create meshgrid and interpolate pressure
    coarse_X, coarse_Y = np.meshgrid(np.linspace(min(ax), max(ax), 500), np.linspace(min(ay), max(ay), 500))
    coarse_pressure_within_boundary = griddata(
        (xp, yp), pressure, (coarse_X, coarse_Y), method='linear', fill_value=0
    )
    boundary_polygon = Polygon(list(zip(ax, ay)))

    mask = ~np.array([
        boundary_polygon.contains(Point(x, y))
        for x, y in zip(coarse_X.ravel(), coarse_Y.ravel())
    ]).reshape(coarse_X.shape)

    coarse_pressure = np.ma.masked_array(coarse_pressure_within_boundary, mask=mask)

    # Plot pressure field
    contours = plt.contourf(coarse_X, coarse_Y, coarse_pressure, cmap='viridis', levels=100)
    plt.colorbar(contours)

    # Plot cell positions
    x_pos = cell_pos[geom]['x']
    y_pos = cell_pos[geom]['y']
    plt.plot(x_pos, y_pos, linestyle='-', color='black')
    plt.scatter([x_pos[0]], [y_pos[0]], color='red', marker='x', s=100)

    # Generate deformed circle
    circle_x = RC * np.cos(np.linspace(0, 2 * np.pi, 100)) + x_pos[-1]
    circle_y = RC * np.sin(np.linspace(0, 2 * np.pi, 100)) + y_pos[-1]

    # Project points outside the boundary onto the nearest boundary segment
    boundary_coords = np.array(list(zip(ax, ay)))
    projected_circle_x = []
    projected_circle_y = []
    
    for cx, cy in zip(circle_x, circle_y):
        point = Point(cx, cy)
        if boundary_polygon.contains(point):
            projected_circle_x.append(cx)
            projected_circle_y.append(cy)
        else:
            # Find the nearest segment
            distances = []
            projections = []
            for i in range(len(boundary_coords)):
                p1 = boundary_coords[i]
                p2 = boundary_coords[(i + 1) % len(boundary_coords)]  # Wrap around
                line = LineString([p1, p2])
                projected_point = line.interpolate(line.project(point))
                distances.append(projected_point.distance(point))
                projections.append(projected_point)
            
            # Use the closest projection
            nearest_projection = projections[np.argmin(distances)]
            projected_circle_x.append(nearest_projection.x)
            projected_circle_y.append(nearest_projection.y)

    # Plot the projected circle
    plt.plot(projected_circle_x, projected_circle_y, color='blue', linestyle='-', label='Deformed Circle')

    # Add final circle for RN
    circle = plt.Circle([x_pos[-1], y_pos[-1]], RN, color='black', fill=False, label='Final Position')
    plt.gca().add_patch(circle)

    # Add labels and finalize plot
    title = f"{geom} - reward: {round(reward[geom], 4)}"
    plt.title(title)
    plt.xlabel('X ($\mu m$)')
    plt.ylabel('Y ($\mu m$)')
    plt.legend()
    plt.tight_layout()
    plt.show()




def plot_validation_movie(cell_pos, reward, output_movie_path="cell_migration_movie.mp4"):
    geom = list(cell_pos.keys())[0]
    
    # Geometries
    if geom == 'straight' or geom == 'bot' or geom == 'top':
        ax = [-15, 15, 15, 36, 36, 51, 51, 36, 36, 15, 15, -15]
        ay = [-10.5, -10.5, -25.5, -25.5, -10.5, -10.5, 10.5, 10.5, 25.5, 25.5, 10.5, 10.5]
    elif geom == 'deadend':
        ax = [-15, 15, 15, -47, -47, 14, 14, -26, -26, 36, 36, 112.21, 112.21,
              174.21, 174.21, 134.21, 134.21, 195.21, 195.21, 133.21, 133.21,
              163.21, 163.21, 133.21, 133.21, 15, 15, -15, -15]
        ay = [-10.5, -10.5, -35.5, -35.5, -97.5, -97.5, -76.5, -76.5, -56.5,
              -56.5, 14.5, 14.5, -56.5, -56.5, -76.5, -76.5, -97.5, -97.5, -35.5,
              -35.5, -10.5, -10.5, 10.5, 10.5, 35.5, 35.5, 10.5, 10.5, -10.5]
    # (The rest of the geometries follow the same structure, no need to repeat them here)
    else:
        return print('No geometry found')

    # Create a figure for the animation
    fig, ax_plot = plt.subplots(figsize=(10, 5))

    # Pressure representation
    xp, yp, pressure = [], [], []

    # Read data from the file
    file_path = f'Data\Pressure-{geom}.txt'
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split(',')
            xp.append(float(parts[1]) * 1e6)  # Convert to micrometers
            yp.append(float(parts[2]) * 1e6)  # Convert to micrometers
            pressure.append(float(parts[3]))

    # Convert data to NumPy arrays as float32
    xp = np.array(xp, dtype=np.float32)
    yp = np.array(yp, dtype=np.float32)
    pressure = np.array(pressure, dtype=np.float32)

    # Create a meshgrid of coordinates within the irregular boundary
    coarse_X, coarse_Y = np.meshgrid(np.linspace(min(ax), max(ax), 500), np.linspace(min(ay), max(ay), 500))

    # Interpolate pressure values only within the irregular boundary
    points_within_hull = np.column_stack((xp, yp))
    coarse_pressure_within_boundary = griddata(points_within_hull, pressure, (coarse_X, coarse_Y),
                                               method='linear', fill_value=0)

    # Create a polygon from the boundary points
    boundary_polygon = Polygon(list(zip(ax, ay)))
    mask = ~np.array([
        boundary_polygon.contains(Point(x, y))
        for x, y in zip(coarse_X.ravel(), coarse_Y.ravel())
    ]).reshape(coarse_X.shape)

    # Create a masked array to set values outside the boundary to np.nan
    coarse_pressure = np.ma.masked_array(coarse_pressure_within_boundary, mask=mask)
    coarse_pressure = 2 * ((coarse_pressure - np.min(coarse_pressure)) / (np.max(coarse_pressure) - np.min(coarse_pressure))) - 1
    contours = ax_plot.contourf(coarse_X, coarse_Y, coarse_pressure, cmap='viridis', levels=100, vmin=-1, vmax=1)
    cbar = plt.colorbar(contours)
    cbar.set_ticks([-1, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, 1]) 

    # Initialize the plot
    x_pos = cell_pos[geom]['x']
    y_pos = cell_pos[geom]['y']
    title = f"{geom} - reward: {round(reward[geom], 4)}"
    ax_plot.set_title(title)

    # Add a circle at the initial and final positions
    initial_pos = plt.scatter([x_pos[0]], [y_pos[0]], color='red', marker='x', s=100, label='Initial Position')
    final_circle = plt.Circle([x_pos[-1], y_pos[-1]], 1.0, color='black', fill=False, label='Final Position')
    ax_plot.add_patch(final_circle)

    # Setup the animation
    def update_frame(frame_num):
        ax_plot.clear()
        ax_plot.set_title(title)
        contours = ax_plot.contourf(coarse_X, coarse_Y, coarse_pressure, cmap='viridis', levels=100, vmin=-1, vmax=1)
        ax_plot.scatter(x_pos[:frame_num], y_pos[:frame_num], color='black')
        ax_plot.scatter([x_pos[0]], [y_pos[0]], color='red', marker='x', s=100, label='Initial Position')
        final_circle = plt.Circle([x_pos[frame_num - 1], y_pos[frame_num - 1]], 1.0, color='black', fill=False)
        ax_plot.add_patch(final_circle)

    # Set up the writer and the animation
    writer = FFMpegWriter(fps=30, metadata={'title': 'Cell Migration Animation', 'artist': 'Matplotlib'})
    ani = animation.FuncAnimation(fig, update_frame, frames=len(x_pos), interval=100, repeat=False)

    # Save the animation to a file
    ani.save(output_movie_path, writer='ffmpeg',fps=30)

    plt.close(fig)


def deform_cell_to_geometry_closed(x_pos, y_pos, RC, RN, boundaries):
    """
    Adjusts the cell shape if it intersects the boundary (with interior holes).

    Parameters:
        x_pos, y_pos: Coordinates of the cell's center.
        RC: Cell radius.
        RN: Nucleus radius.
        boundaries: Tuple ((exterior_x, exterior_y), (interior_x, interior_y)) 
                    representing the geometry boundary.

    Returns:
        modified_cell: Shapely geometry of the deformed cell.
    """
    (exterior_x, exterior_y), (interior_x, interior_y) = boundaries  # Extract boundaries
    
    # Construct the Shapely Polygon with an interior hole
    exterior = list(zip(exterior_x, exterior_y))
    interior = list(zip(interior_x, interior_y))
    
    # Create a Polygon with a hole
    boundary_polygon = Polygon(shell=exterior, holes=[interior])

    # Create a circular buffer (cell shape)
    cell = Point(x_pos, y_pos).buffer(RC, resolution=50)  

    # Check intersection with the boundary (exterior & interior)
    if cell.intersects(boundary_polygon):
        # Get the valid region inside the geometry
        deformed_cell = cell.intersection(boundary_polygon)
    else:
        deformed_cell = cell  # If no intersection, keep original shape

    return deformed_cell

def deform_cell_to_geometry(x_pos, y_pos, RC, RN, geometry_boundary):
    """
    Adjusts the cell shape if it intersects the boundary.

    Parameters:
        x_pos, y_pos: Coordinates of the cell's center.
        RC: Cell radius.
        RN: Nucleus radius.
        geometry_boundary: (ax, ay) tuple representing the geometry boundary.

    Returns:
        modified_cell: Shapely geometry of the deformed cell.
    """
    ax, ay = geometry_boundary  # Extract boundary points
    boundary_polygon = Polygon(zip(ax, ay))  # Convert to Shapely polygon

    # Create a circular buffer (cell shape)
    cell = Point(x_pos, y_pos).buffer(RC, resolution=50)  

    # Check intersection with the boundary
    if cell.intersects(boundary_polygon):
        # Get the overlapping region
        deformed_cell = cell.intersection(boundary_polygon)
    else:
        deformed_cell = cell  # If no intersection, keep original shape

    return deformed_cell


def plot_validation_results(cell_pos):
    plt.figure(figsize=(10, 5))

    geom = list(cell_pos.keys())[0]
    
    # Define geometry boundaries
    if geom in ["straight","top","bot"]:
        ax = [-15, 15, 15, 36, 36, 51, 51, 36, 36, 15,15,-15,-15]
        ay = [-10.5, -10.5, -25.5, -25.5,-10.5, -10.5, 10.5, 10.5,\
            25.5, 25.5, 10.5,10.5,-10.5]
    elif geom in ["shortdeadend"]:
        ax = [-15,15,15,36,36,112.21,112.21,133.21,133.21,163.21,163.21,\
            133.21,133.21,15,15,-15,-15]
        ay = [-10.5,-10.5,-35.5,-35.5,14.5,14.5,-35.5,-35.5,-10.5,-10.5,10.5,\
            10.5,35.5,35.5,10.5,10.5,-10.5]    
    elif geom in ["deadend"]:  
        ax = [-15,15,15,-47,-47,14,14,-26,-26,36,36,112.21,112.21,\
            174.21,174.21,134.21,134.21,195.21,195.21,133.21,133.21,\
            163.21,163.21,133.21,133.21,15,15,-15,-15]
        ay = [-10.5,-10.5,-35.5,-35.5,-97.5,-97.5,-76.5,-76.5,-56.5,\
            -56.5,14.5,14.5,-56.5,-56.5,-76.5,-76.5,-97.5,-97.5,-35.5,\
            -35.5,-10.5,-10.5,10.5,10.5,35.5,35.5,10.5,10.5,-10.5]
    elif geom in ["square"]:
        ax = [-21,21,21,147,147,189,189,147,147,21,21,-21,-21, 42,126,126,42,42]
        ay = [-10.5,-10.5,-63,-63,-10.5,-10.5,10.5,10.5,63,63,10.5,10.5,-10.5, -42,-42,42,42,-42]
    elif geom in ["twisted"]:
        ax = [-21,21,21,-147,-147,-84,-84,-63,-63,0,0,21,21,84,84,105,105,168,168,189,189,252,252,\
            273,273,336,336,168,168,210,210,168,168,21,21,-21,-21,42,-126,-126,-105,-105,-42,-42,-21,\
            -21,42,42,63,63,126,126,147,147,210,210,231,231,294,294,315,315,147,147,42,42]
        ay = [-10.5,-10.5,-52.5,-52.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,\
            -199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,\
            -52.5,-52.5,-10.5,-10.5,10.5,10.5,52.5,52.5,10.5,10.5,-10.5,-73.5,-73.5,-178.5,-178.5,-115.5,\
            -115.5,-178.5,-178.5,-115.5,-115.5,-178.5,-178.5,-115.5,-115.5,-178.5,-178.5,-115.5,-115.5,-178.5,\
            -178.5,-115.5,-115.5,-178.5,-178.5,-73.5,-73.5,31.5,31.5,-73.5]
        
            #twisted boundary
        N= 36
        # ax.insert(N, ax[-1])  # Close exterior
        # ay.insert(N, ay[0])

        ax.append(ax[N])  # Close interior
        ay.append(ay[N])

        # Define exterior boundary (first N points)
        exterior_x = ax[:N]  # Adjust N to match the number of points in the exterior
        exterior_y = ay[:N]

        # Define interior holes (remaining points)
        interior_x = ax[N+1:-2]  # The rest of the points belong to the interior
        interior_y = ay[N+1:-2]

        # Ensure closure of each boundary
        exterior_x.append(exterior_x[0])
        exterior_y.append(exterior_y[0])

        interior_x.append(interior_x[0])
        interior_y.append(interior_y[0])



        # Plot exterior boundary
        plt.plot(exterior_x, exterior_y, linestyle='-', color='black')


        plt.plot(interior_x, interior_y, linestyle='-', color='black')
        
    elif geom in ["curved"]:
        ax = [-21,21,21,105,105,147,147,231,231,273,273,357,357,399,399,357,
                357,346.8684,317.6955,273,218.1729,159.8271,105,60.3045,31.1316,21,21,-21,-21,
                42,84,84,168,168,210,210,294,294,336,337,328.1348,302.6085,263.5,215.5263,164.4737,116.5,77.3915,51.8652,43,42]
        ay = [-10.5,-10.5,-105,-105,-63,-63,-105,-105,-63,-63,-105,-105,-10.5,-10.5,10.5,10.5,136,193.4594,243.9883, 
                281.4923,301.4477,301.4477,281.4923,243.9883,193.4594,136,10.5,10.5,-10.5,
                -84,-84,-42,-42,-84,-84,-42,-42,-84,-84,136.5,186.7770,230.9898,263.8057,281.2667,281.2667,263.8057,230.9898,186.7770,136.5,-84]
                #curved
        N= 29
        # ax.insert(N, ax[-1])  # Close exterior
        # ay.insert(N, ay[0])

        ax.append(ax[N])  # Close interior
        ay.append(ay[N])
        # Define exterior boundary (first N points)
        exterior_x = ax[:N]  # Adjust N to match the number of points in the exterior
        exterior_y = ay[:N]

        # Define interior holes (remaining points)
        interior_x = ax[N:-1]  # The rest of the points belong to the interior
        interior_y = ay[N:-1]

        # Ensure closure of each boundary
        exterior_x.append(exterior_x[0])
        exterior_y.append(exterior_y[0])

        interior_x.append(interior_x[0])
        interior_y.append(interior_y[0])

        # Plot exterior boundary
        plt.plot(exterior_x, exterior_y, linestyle='-', color='black')

        plt.plot(interior_x, interior_y, linestyle='-', color='black')
           
    else:
        return print('No geometry found')


    if geom in ["straight","top","bot","deadend"]:
        
        plt.figure(figsize=(10, 5))

        # Plot boundary
        plt.plot(ax, ay, linestyle='-', color='black', label='Geometry Boundary')

        # plt.gca().set_aspect('equal', adjustable='box')
        # Pressure representation
        xp, yp, pressure = [], [], []

        # Read data from the file
        file_path = f'Data\Pressure-{geom}.txt'
        with open(file_path, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                parts = line.strip().split(',')
                xp.append(float(parts[1]) * 1e6)  # Convert to micrometers
                yp.append(float(parts[2]) * 1e6)  # Convert to micrometers
                pressure.append(float(parts[3]))

        # Convert data to NumPy arrays as float32
        xp = np.array(xp, dtype=np.float32)
        yp = np.array(yp, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)
        # Create a meshgrid of coordinates within the irregular boundary
        coarse_X, coarse_Y = np.meshgrid(np.linspace(min(ax), max(ax), 500), np.linspace(min(ay), max(ay), 500))

        # Interpolate pressure values only within the irregular boundary
        points_within_hull = np.column_stack((xp, yp))
        coarse_pressure_within_boundary = griddata(points_within_hull, pressure, (coarse_X, coarse_Y),
                                                    method='linear', fill_value=0)

        # Create a polygon from the boundary points
        boundary_polygon = Polygon(list(zip(ax, ay)))
        mask = ~np.array([
            boundary_polygon.contains(Point(x, y))
            for x, y in zip(coarse_X.ravel(), coarse_Y.ravel())
        ]).reshape(coarse_X.shape)

        # Create a masked array to set values outside the boundary to np.nan
        coarse_pressure = np.ma.masked_array(coarse_pressure_within_boundary, mask=mask)
        coarse_pressure = 2 * ((coarse_pressure - np.min(coarse_pressure)) / (np.max(coarse_pressure) - np.min(coarse_pressure))) - 1
        contours = plt.contourf(coarse_X, coarse_Y, coarse_pressure, cmap='viridis', levels=100, vmin=-1, vmax=1)
        cbar = plt.colorbar(contours)
        cbar.set_ticks([-1, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, 1]) 


        # Plot cell trajectory
        x_pos = cell_pos[geom]['x']
        y_pos = cell_pos[geom]['y']
        plt.plot(x_pos, y_pos, linestyle='-', color='black', label="Trajectory")

        # Plot initial position
        plt.scatter([x_pos[0]], [y_pos[0]], color='red', marker='x', s=100, label='Initial Position')

        # Deform final cell shape
        deformed_cell = deform_cell_to_geometry(x_pos[-1], y_pos[-1], RC, RN, (ax, ay))

        # Plot deformed cell
        x, y = deformed_cell.exterior.xy
        plt.fill(x, y, color='black', alpha=0.2, label='Deformed Cell')
        plt.plot(x, y, color='black', alpha=0.3,linewidth=1)

        # Plot nucleus
        nucleus = Point(x_pos[-1], y_pos[-1]).buffer(RN, resolution=30)
        x_n, y_n = nucleus.exterior.xy
        plt.fill(x_n, y_n, color='black', alpha=0.6, label='Nucleus')
        plt.plot(x_n, y_n, color='black', alpha=0.7, linewidth=1)

        plt.xlabel("X (µm)")
        plt.ylabel("Y (µm)")
        # plt.legend()
        plt.axis("off")  # Hide axis lines, ticks, and labels
        plt.xticks([])   # Remove x-axis ticks
        plt.yticks([])   # Remove y-axis ticks
        plt.gca().spines["top"].set_visible(False)  
        plt.gca().spines["right"].set_visible(False)  
        plt.gca().spines["left"].set_visible(False)  
        plt.gca().spines["bottom"].set_visible(False)  
        # plt.savefig("dead-end_trajectory_v3.svg", format="svg", dpi=600)
        plt.show()  
         
    elif geom in ["twisted","curved"]:
        xp, yp, pressure = [], [], []

        # Read data from the file
        file_path = f'Data\Pressure-{geom}.txt'
        with open(file_path, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                parts = line.strip().split(',')
                xp.append(float(parts[1]) * 1e6)  # Convert to micrometers
                yp.append(float(parts[2]) * 1e6)  # Convert to micrometers
                pressure.append(float(parts[3]))

        # Convert data to NumPy arrays as float32
        xp = np.array(xp, dtype=np.float32)
        yp = np.array(yp, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)
        # Create a meshgrid of coordinates within the irregular boundary
        coarse_X, coarse_Y = np.meshgrid(np.linspace(min(ax), max(ax), 500), np.linspace(min(ay), max(ay), 500))

        # Interpolate pressure values only within the irregular boundary
        points_within_hull = np.column_stack((xp, yp))
        coarse_pressure_within_boundary = griddata(points_within_hull, pressure, (coarse_X, coarse_Y),
                                                    method='linear', fill_value=0)

        # Create a polygon from the boundary points
        boundary_polygon = Polygon(list(zip(ax, ay)))
        mask = ~np.array([
            boundary_polygon.contains(Point(x, y))
            for x, y in zip(coarse_X.ravel(), coarse_Y.ravel())
        ]).reshape(coarse_X.shape)

        # Create a masked array to set values outside the boundary to np.nan
        coarse_pressure = np.ma.masked_array(coarse_pressure_within_boundary, mask=mask)
        coarse_pressure = 2 * ((coarse_pressure - np.min(coarse_pressure)) / (np.max(coarse_pressure) - np.min(coarse_pressure))) - 1
        contours = plt.contourf(coarse_X, coarse_Y, coarse_pressure, cmap='viridis', levels=100, vmin=-1, vmax=1)
        cbar = plt.colorbar(contours)
        cbar.set_ticks([-1, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, 1]) 


        # Plot cell trajectory
        x_pos = cell_pos[geom]['x']
        y_pos = cell_pos[geom]['y']
        plt.plot(x_pos, y_pos, linestyle='-', color='black', label="Trajectory")

        # Plot initial position
        plt.scatter([x_pos[0]], [y_pos[0]], color='red', marker='x', s=100, label='Initial Position')

        # Deform final cell shape
        deformed_cell = deform_cell_to_geometry_closed(x_pos[-1], y_pos[-1], RC, RN, ((exterior_x, exterior_y), (interior_x, interior_y)))

        # Plot deformed cell
        x, y = deformed_cell.exterior.xy
        plt.fill(x, y, color='black', alpha=0.2, label='Deformed Cell')
        plt.plot(x, y, color='black', alpha=0.3,linewidth=1)

        # Plot nucleus
        nucleus = Point(x_pos[-1], y_pos[-1]).buffer(RN, resolution=30)
        x_n, y_n = nucleus.exterior.xy
        plt.fill(x_n, y_n, color='black', alpha=0.6, label='Nucleus')
        plt.plot(x_n, y_n, color='black', alpha=0.7, linewidth=1)

        plt.xlabel("X (µm)")
        plt.ylabel("Y (µm)")
        # plt.legend()
        plt.axis("off")  # Hide axis lines, ticks, and labels
        plt.xticks([])   # Remove x-axis ticks
        plt.yticks([])   # Remove y-axis ticks
        plt.gca().spines["top"].set_visible(False)  
        plt.gca().spines["right"].set_visible(False)  
        plt.gca().spines["left"].set_visible(False)  
        plt.gca().spines["bottom"].set_visible(False)  
        plt.show()

    return




















