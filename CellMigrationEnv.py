import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from collections import namedtuple, deque
import random
from matplotlib.path import Path
from shapely.geometry import Point, Polygon, LineString

#MODEL PARAMETERS 
TSIM    = 3*5*300 #min  # Reduce the simulated time for training
DT      = 10    #min
VC      = 0.2   #mu m/min
DELTA   = 0.0   #mu m
RC      = 10    #mu m 
RN      = 4     #mu m


class CellMigrationEnv(gym.Env):
    def __init__(self, N_obs,geometry):
        super(CellMigrationEnv, self).__init__()
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-1, high=1, dtype=np.float32)

        self.action_space = spaces.Discrete(16)
        self.N_obs        = N_obs  # Define N_obs as an instance attribute
        
        #...variables of the computational model 
        self.t       = 0
        self.x_c     = 0
        self.y_c     = 0
        self.e_x     = 1
        self.e_y     = 0
        self._create_geometry(geometry)
        self.geom    = geometry

    def reset(self):
        # Initialize the environment
        self.t       = 0
        self.x_c     = 0
        self.y_c     = 0
        self.e_x     = 1
        self.e_y     = 0
        obs_pressure = self._interp_pressure_from_file() #OBTAIN THE INITIAL STATE
        
        return np.array(self.state, dtype=np.float32), {"pressure": obs_pressure}


    def step(self, action):
        terminated = False 
        # truncated  = False 
        
        #Determine the direction from the action
        th = action*(2*np.pi/self.action_space.n)
        xC = RC * np.cos(th) + self.x_c
        yC = RC * np.sin(th) + self.y_c

        self.e_x = (xC - self.x_c)/np.sqrt((self.x_c - xC)**2 + (self.y_c - yC)**2)
        self.e_y = (yC - self.y_c)/np.sqrt((self.x_c - xC)**2 + (self.y_c - yC)**2)

        #Wall interaction 
        e_cell_wall = self._wall_interaction()
         
        #Update position
        self.x_c += DT*(VC*self.e_x + VC*e_cell_wall[0])
        self.y_c += DT*(VC*self.e_y + VC*e_cell_wall[1])
        
        #Detect collision 
        # terminated  = self.is_collision()
        
        #Obtain reward
        reward  = self._get_reward()  
        
        #Update state to make the next decision 
        obs_pressure = self._interp_pressure_from_file()      
        
        #Check if the simulation has finished 
        if self.t >= TSIM or reward == 1:
            terminated = True
        
        self.t += DT 

        return np.array(self.state), reward, terminated, obs_pressure, {}


    def _get_reward(self):
        
        if self.g_s == "straight":
            goal = [40.5, 0]
            if self.x_c < goal[0]: 
                # reward = 10**(-(goal[0]-self.x_c)**2/(2*(goal[0])**2))
                D_c = np.sqrt((goal[0] - self.x_c)**2 + (goal[1] - self.y_c)**2)
                D_g = np.sqrt((goal[0])**2 + (goal[1])**2) 
                # reward = 1 - D_c/D_g
                reward = 1 - 0.5*(D_c/D_g)**2
            else:
                reward = 1
            
        elif self.g_s == "top":
            goal = [25.5, 15]
            if self.y_c < goal[1]: 
                # reward = 10**(-(goal[1]-self.y_c)**2/(2*(goal[1])**2))
                D_c = np.sqrt((goal[0] - self.x_c)**2 + (goal[1] - self.y_c)**2)
                D_g = np.sqrt((goal[0])**2 + (goal[1])**2) 
                # reward = 1 - D_c/D_g
                reward = 1 - 0.5*(D_c/D_g)**2
            else:
                reward = 1
                
        elif self.g_s == "bot":
            goal = [25.5, -15]
            if self.y_c > goal[1]: 
                # reward = 10**(-(goal[1]-self.y_c)**2/(2*(goal[1])**2))
                D_c = np.sqrt((goal[0] - self.x_c)**2 + (goal[1] - self.y_c)**2)
                D_g = np.sqrt((goal[0])**2 + (goal[1])**2) 
                # reward = 1 - D_c/D_g
                reward = 1 - 0.5*(D_c/D_g)**2
            else:
                reward = 1
        
        elif self.g_s == "deadend" or self.g_s == 'shortdeadend' or self.g_s == 'square':
            goal = [140, 0]
            if self.x_c < goal[0]: 
                # reward = 10**(-(goal[1]-self.y_c)**2/(2*(goal[1])**2))
                D_c = np.sqrt((goal[0] - self.x_c)**2 + (goal[1] - self.y_c)**2)
                D_g = np.sqrt((goal[0])**2 + (goal[1])**2) 
                # reward = 1 - D_c/D_g
                reward = 1 - 0.5*(D_c/D_g)**2
            else:
                reward = 1
                
        elif self.g_s == 'twisted':
            goal = [190, 0]
            if self.x_c < goal[0]: 
                # reward = 10**(-(goal[1]-self.y_c)**2/(2*(goal[1])**2))
                D_c = np.sqrt((goal[0] - self.x_c)**2 + (goal[1] - self.y_c)**2)
                D_g = np.sqrt((goal[0])**2 + (goal[1])**2) 
                # reward = 1 - D_c/D_g
                reward = 1 - 0.5*(D_c/D_g)**2
            else:
                reward = 1 
                          
        elif self.g_s == 'curved':
            goal = [380, 0]
            if self.x_c < goal[0]: 
                # reward = 10**(-(goal[1]-self.y_c)**2/(2*(goal[1])**2))
                D_c = np.sqrt((goal[0] - self.x_c)**2 + (goal[1] - self.y_c)**2)
                D_g = np.sqrt((goal[0])**2 + (goal[1])**2) 
                # reward = 1 - D_c/D_g
                reward = 1 - 0.5*(D_c/D_g)**2
            else:
                reward = 1     
                    
        return reward
    
    
    
    def check_feasible_actions(self): 

        polygon_vertices = np.column_stack((self.ver_ax, self.ver_ay))  # Combine x and y coordinates

        # Create a Path object for the polygon
        polygon_path = Path(polygon_vertices)

        # Points to check (circle points)
        th = np.linspace(0, 2 * np.pi, self.action_space.n, endpoint=False)
        xC = RC * np.cos(th) + self.x_c
        yC = RC * np.sin(th) + self.y_c

        # Combine the points into an array of shape (N, 2)
        points_to_check = np.column_stack((xC, yC))

        # Check which points are inside the polygon
        inside = polygon_path.contains_points(points_to_check)

        # Points outside the polygon
        outside_points = points_to_check[~inside]
                
        # Indices of points outside the polygon
        inside_indices  = np.where(inside)[0]
        outside_indices = np.where(~inside)[0]
        
        return outside_indices 
    
    

    
    
    def _interp_pressure_from_file(self): #Complete then with path 
        
        file_path = f'Data\Pressure-{self.g_s}.txt'
        
        # Read the CSV file using pandas
        data = pd.read_csv(file_path, skiprows=1, header=None, names=["nodenumber", "xcoordinate", "ycoordinate", "pressure"])
    
        # Circle formation
        th = np.linspace(0, 2 * np.pi, self.N_obs, endpoint=False)
        xC = RC * np.cos(th) + self.x_c
        yC = RC * np.sin(th) + self.y_c
    
        # Interpolation
        x_0      = data["xcoordinate"] * 1e6  # x-vector in micrometers
        y_0      = data["ycoordinate"] * 1e6  # y-vector in micrometers
        pressure = data["pressure"]
        
        Input_p = griddata((x_0, y_0), data["pressure"], (xC, yC), method='nearest')
    
        if np.isnan(Input_p).any():
            raise ValueError('Simulation error: Function output contains NaN values.')
    
        # Normalization, -1 to 1
        self.state = 2 * ((Input_p - np.min(Input_p)) / (np.max(Input_p) - np.min(Input_p))) - 1
    
        # Pressure observed from the environment 
        xp = np.array(x_0, dtype=np.float32)
        yp = np.array(y_0, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)
        pressure_norm =  2 * ((pressure - np.min(pressure)) / (np.max(pressure) - np.min(pressure))) - 1           
        obs_pressure = griddata((xp, yp), pressure_norm, (xC, yC), method='nearest')  
    
        return  obs_pressure
            
    
    def _pressure_field_normalized(self,data,xC,yC):
        
        # data
        x_0 = data["xcoordinate"] * 1e6  # x-vector in micrometers
        y_0 = data["ycoordinate"] * 1e6  # y-vector in micrometers
        pressure = data["pressure"]
        
        # Convert data to NumPy arrays as float32
        xp = np.array(x_0, dtype=np.float32)
        yp = np.array(y_0, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)
        
        pressure_norm =  2 * ((pressure - np.min(pressure)) / (np.max(pressure) - np.min(pressure))) - 1   
            
        obs_pressure = griddata((xp, yp), pressure_norm, (xC, yC), method='nearest')    
            
        # obs_pressure = griddata((coarse_X, coarse_Y), coarse_pressure, (xC, yC), method='nearest')
        
        return obs_pressure
    
    
       
    def _create_geometry(self,geometry):
        #Geometry 
        #...vertices
        if geometry[0] in ["straight","top","bot"]:
            ax = [-15, 15, 15, 36, 36, 51, 51, 36, 36, 15,15,-15,-15]
            ay = [-10.5, -10.5, -25.5, -25.5,-10.5, -10.5, 10.5, 10.5,\
                25.5, 25.5, 10.5,10.5,-10.5]
        elif geometry[0] in ["shortdeadend"]:
            ax = [-15,15,15,36,36,112.21,112.21,133.21,133.21,163.21,163.21,\
                133.21,133.21,15,15,-15,-15]
            ay = [-10.5,-10.5,-35.5,-35.5,14.5,14.5,-35.5,-35.5,-10.5,-10.5,10.5,\
                10.5,35.5,35.5,10.5,10.5,-10.5]    
        elif geometry[0] in ["deadend"]:  
            ax = [-15,15,15,-47,-47,14,14,-26,-26,36,36,112.21,112.21,\
                174.21,174.21,134.21,134.21,195.21,195.21,133.21,133.21,\
                163.21,163.21,133.21,133.21,15,15,-15,-15]
            ay = [-10.5,-10.5,-35.5,-35.5,-97.5,-97.5,-76.5,-76.5,-56.5,\
                -56.5,14.5,14.5,-56.5,-56.5,-76.5,-76.5,-97.5,-97.5,-35.5,\
                -35.5,-10.5,-10.5,10.5,10.5,35.5,35.5,10.5,10.5,-10.5]
        elif geometry[0] in ["square"]:
            ax = [-21,21,21,147,147,189,189,147,147,21,21,-21,-21, 42,126,126,42,42]
            ay = [-10.5,-10.5,-63,-63,-10.5,-10.5,10.5,10.5,63,63,10.5,10.5,-10.5, -42,-42,42,42,-42]
        elif geometry[0] in ["twisted"]:
            ax = [-21,21,21,-147,-147,-84,-84,-63,-63,0,0,21,21,84,84,105,105,168,168,189,189,252,252,\
                273,273,336,336,168,168,210,210,168,168,21,21,-21,-21,42,-126,-126,-105,-105,-42,-42,-21,\
                -21,42,42,63,63,126,126,147,147,210,210,231,231,294,294,315,315,147,147,42,42]
            ay = [-10.5,-10.5,-52.5,-52.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,\
                -199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,-136.5,-136.5,-199.5,-199.5,\
                -52.5,-52.5,-10.5,-10.5,10.5,10.5,52.5,52.5,10.5,10.5,-10.5,-73.5,-73.5,-178.5,-178.5,-115.5,\
                -115.5,-178.5,-178.5,-115.5,-115.5,-178.5,-178.5,-115.5,-115.5,-178.5,-178.5,-115.5,-115.5,-178.5,\
                -178.5,-115.5,-115.5,-178.5,-178.5,-73.5,-73.5,31.5,31.5,-73.5]
        elif geometry[0] in ["curved"]:
            ax = [-21,21,21,105,105,147,147,231,231,273,273,357,357,399,399,357,
                  357,346.8684,317.6955,273,218.1729,159.8271,105,60.3045,31.1316,21,21,-21,-21,
                  42,84,84,168,168,210,210,294,294,336,337,328.1348,302.6085,263.5,215.5263,164.4737,116.5,77.3915,51.8652,43,42]
            ay = [-10.5,-10.5,-105,-105,-63,-63,-105,-105,-63,-63,-105,-105,-10.5,-10.5,10.5,10.5,136,193.4594,243.9883, 
                  281.4923,301.4477,301.4477,281.4923,243.9883,193.4594,136,10.5,10.5,-10.5,
                 -84,-84,-42,-42,-84,-84,-42,-42,-84,-84,136.5,186.7770,230.9898,263.8057,281.2667,281.2667,263.8057,230.9898,186.7770,136.5,-84]
                        
        self.ver_ax = ax
        self.ver_ay = ay
        return
        
        
    def geometry_selection(self,geom):
        self.g_s  = geom 
        return
        
        
    def is_collision(self):
        shock = False
        n_p = len(self.ver_ax)
        
        # Collision convex domain
        for j in range(n_p-1):
            # Calculate the vectors representing the edges of the polygon
            edge_vector = np.array([self.ver_ax[j+1] - self.ver_ax[j], self.ver_ay[j+1] - self.ver_ay[j]])
            particle_to_vertex = np.array([self.x_c - self.ver_ax[j], self.y_c - self.ver_ay[j]])
        
            # Calculate the perpendicular distance from the particle to the line defined by the edges
            distance = np.linalg.norm(np.cross(edge_vector, particle_to_vertex)) / np.linalg.norm(edge_vector)
        
            if distance <= RC + DELTA:
                # Check if the particle is within the line segment
                if 0 <= np.dot(particle_to_vertex, edge_vector) <= np.dot(edge_vector, edge_vector):
                    shock = True
                    break 
                
        # Collision with walls (outside the loop)
        for j in range(n_p):
            # Calculate the distance from the particle to the vertex
            d = np.sqrt((self.ver_ax[j] - self.x_c)**2 + (self.ver_ay[j] - self.y_c)**2)
        
            if d <= RC + DELTA:
                shock = True
                break  # Exit the loop if a collision is detected   
         
        return shock 
    
    
    
    def _wall_interaction(self): 
        
        shock = False
        n_p         = len(self.ver_ax)
        e_cell_wall = np.array([0.0, 0.0])
        
        # Collision with convex domain
        for j in range(n_p - 1):
            # Calculate the vectors representing the edges of the polygon
            edge_vector = np.array([self.ver_ax[j+1] - self.ver_ax[j], self.ver_ay[j+1] - self.ver_ay[j]])
            particle_to_vertex = np.array([self.x_c - self.ver_ax[j], self.y_c - self.ver_ay[j]])
            
            # Calculate the perpendicular distance from the particle to the line defined by the edges
            distance = np.linalg.norm(np.cross(edge_vector, particle_to_vertex)) / np.linalg.norm(edge_vector)
            
            if distance <= RN + DELTA:
                # Check if the particle is within the line segment
                if 0 <= np.dot(particle_to_vertex, edge_vector) <= np.dot(edge_vector, edge_vector):
                    # Calculate the touching point
                    projection = np.dot(particle_to_vertex, edge_vector) / np.dot(edge_vector, edge_vector)
                    touching_point = np.array([
                        self.ver_ax[j] + projection * edge_vector[0],
                        self.ver_ay[j] + projection * edge_vector[1]
                    ])
                    
                    # Calculate the direction vector from cell center to touching point
                    direction_vector = touching_point - np.array([self.x_c, self.y_c])
                    
                    # Normalize the direction vector
                    e_cell_wall = - direction_vector / np.linalg.norm(direction_vector)
                    
                    shock = True
                    break  # Exit loop after detecting collision

        # Collision with walls (outside the loop)
        for j in range(n_p):
            # Calculate the distance from the particle to the vertex
            d = np.sqrt((self.ver_ax[j] - self.x_c)**2 + (self.ver_ay[j] - self.y_c)**2)
            
            if d <= RN + DELTA:
                # Calculate the direction vector from the cell center to the vertex (touching point)
                touching_point = np.array([self.ver_ax[j], self.ver_ay[j]])
                direction_vector = touching_point - np.array([self.x_c, self.y_c])
                
                # Normalize the direction vector
                e_cell_wall = - direction_vector / np.linalg.norm(direction_vector)
                
                shock = True
                break  # Exit the loop if a collision is detected
                
        
        return e_cell_wall
        
    
    def _wall_interaction_repulsive_strength(self): 
        shock = False
        n_p = len(self.ver_ax)
        e_cell_wall = np.array([0.0, 0.0])
        repulsion_strength = 0.0  # Default repulsion strength

        # Collision with convex domain
        for j in range(n_p - 1):
            # Calculate the vectors representing the edges of the polygon
            edge_vector = np.array([self.ver_ax[j+1] - self.ver_ax[j], self.ver_ay[j+1] - self.ver_ay[j]])
            particle_to_vertex = np.array([self.x_c - self.ver_ax[j], self.y_c - self.ver_ay[j]])
            
            # Calculate the perpendicular distance from the particle to the line defined by the edges
            distance = np.linalg.norm(np.cross(edge_vector, particle_to_vertex)) / np.linalg.norm(edge_vector)
            
            if distance <= RC + DELTA:
                # Check if the particle is within the line segment
                if 0 <= np.dot(particle_to_vertex, edge_vector) <= np.dot(edge_vector, edge_vector):
                    # Calculate the touching point
                    projection = np.dot(particle_to_vertex, edge_vector) / np.dot(edge_vector, edge_vector)
                    touching_point = np.array([
                        self.ver_ax[j] + projection * edge_vector[0],
                        self.ver_ay[j] + projection * edge_vector[1]
                    ])
                    
                    # Calculate the direction vector from cell center to touching point
                    direction_vector = touching_point - np.array([self.x_c, self.y_c])
                    distance_to_wall = np.linalg.norm(direction_vector)  # Distance to the wall
                    
                    # Normalize the direction vector
                    e_cell_wall = - direction_vector / distance_to_wall
                    
                    # Calculate repulsion strength (closer → stronger)
                    repulsion_strength = max(0.0, 1.0 - distance_to_wall / (RC + DELTA))
                    e_cell_wall *= repulsion_strength
                    
                    shock = True
                    break  # Exit loop after detecting collision

        # Collision with walls (outside the loop)
        for j in range(n_p):
            # Calculate the distance from the particle to the vertex
            distance_to_wall = np.sqrt((self.ver_ax[j] - self.x_c)**2 + (self.ver_ay[j] - self.y_c)**2)
            
            if distance_to_wall <= RC + DELTA:
                # Calculate the direction vector from the cell center to the vertex (touching point)
                touching_point = np.array([self.ver_ax[j], self.ver_ay[j]])
                direction_vector = touching_point - np.array([self.x_c, self.y_c])
                
                # Normalize the direction vector
                e_cell_wall = - direction_vector / distance_to_wall
                
                # Calculate repulsion strength (closer → stronger)
                repulsion_strength = max(0.0, 1.0 - distance_to_wall / (RC + DELTA))
                e_cell_wall *= repulsion_strength
                
                shock = True
                break  # Exit the loop if a collision is detected

        return e_cell_wall


    def render(self):
        # Implement visualization
        pass



