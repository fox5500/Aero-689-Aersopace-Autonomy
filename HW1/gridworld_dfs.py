#import cv2
import numpy as np
import heapq
import time
import json
from collections import deque
from math import *
import matplotlib.pyplot as plt
import csv


iterations =  100

with open("DFS_Random100x100_Run_Data_100runs.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Run Number", "Runtime", "Path Cost", "Nodes Generated", "Move Sequence"])

for iteration in range(iterations):
    print(f"Loop {iteration + 1}:")

    random_grid = np.zeros((100,100))
    walls = np.random.choice(random_grid.size, 2000, replace=False)
    goals = np.random.choice(random_grid.size, 1, replace=False)
    start = np.random.choice(random_grid.size, 1, replace=False)
    random_grid.ravel()[walls] = np.nan
    random_grid.ravel()[goals] = 1
    random_grid.ravel()[start] = 5
    def get_grid_cell_by_location(grid_dict,x,y):
        # This function returns a particular node given its x and y coordinates.

        grid_cell = None
        for loc in grid_dict["node_list"]:
            if loc["x"] == x and loc["y"] == y:
                grid_cell = loc
        return grid_cell

    def create_grid_dict_from_array(grid_array):
        # This function converts the numpy array to a dict with information on
        # the number of columns, the number of rows, the start location, and the
        # goal and obstacle status of each cell, as well as possible actions in each cell.

        grid_dict = {}
        grid_dict["num_rows"] = len(grid_array[:,0])
        grid_dict["num_cols"] = len(grid_array[0,:])
        for i in range(len(grid_array[0,:])):
            for j in range(len(grid_array[:,0])):
                if grid_array[i,j] == 5:
                    start_x = j
                    start_y = i
        grid_dict["start_location"] = {
            "x": start_x,
            "y": start_y
        }
        grid_dict["node_list"] = []
        for i in range(len(grid_array[0,:])):
            for j in range(len(grid_array[:,0])):
                cost = 1
                if np.isnan(grid_array[i,j]):
                    wall = True
                    cost = np.nan
                else:
                    wall = False
                if grid_array[i,j] == 1:
                    goal = True
                    cost = 0
                else:
                    goal = False
                node = {
                    "x": j,
                    "y": i,
                    "cost": cost,
                    "goal": goal,
                    "wall": wall
                }
                grid_dict["node_list"].append(node)

        for node in grid_dict["node_list"]:
            if node["wall"] == True:
                possible_actions = []
                node["possible_actions"] = possible_actions
                continue
            possible_actions = ["left","right","up","down"]
            x_location = node["x"]
            y_location = node["y"]
            if x_location-1 < 0:
                possible_actions.remove("left")
            else:
                if np.isnan(grid_array[y_location,x_location-1]):
                    possible_actions.remove("left")
            if x_location+1 >= len(grid_array[0,:]):
                possible_actions.remove("right")
            else:
                if np.isnan(grid_array[y_location,x_location+1]):
                    possible_actions.remove("right")
            if y_location-1 < 0:
                possible_actions.remove("up")
            else:
                if np.isnan(grid_array[y_location-1,x_location]):
                    possible_actions.remove("up")
            if y_location+1 >= len(grid_array[:,0]):
                possible_actions.remove("down")
            else:
                if np.isnan(grid_array[y_location+1,x_location]):
                    possible_actions.remove("down")
            node["possible_actions"] = possible_actions

        out_file = open("grid.json", "w")

        json.dump(grid_dict, out_file, indent = 4)

        out_file.close()
        return grid_dict

    #def plot_grid(grid_dict,grid):
    #    # This function plots the grid.
    #
    #    cell_height = 300
    #    cell_width = 300
    #    top_left = (int(cell_height * 0.01), int(cell_width * 0.01))
    #    bot_right = (int(cell_height * 0.99), int(cell_width * 0.99))
    #    new_pos = (int(0.5 * cell_width), int(0.5 * cell_height))
    #
    #    row_concatenation = None
    #    for row_ind in range(grid_dict["num_rows"]):
    #        column_concatenation = None
    #
    #        for col_ind in range(grid_dict["num_cols"]):
    #
    #            blank_image = np.ones((cell_height, cell_width, 3), np.uint8)
    #            colored_field = False
    #            # walls or special states or none
    #            if np.isnan(grid[row_ind,col_ind]):
    #                blank_image = blank_image * 128
    #                blank_image.astype(np.uint8)
    #            else:
    #                # white state by default
    #                blank_image = blank_image * 255
    #                blank_image = blank_image.astype(np.uint8)
    #
    #            if grid[row_ind,col_ind] == 5:
    #                # start percept has border with special color
    #                txt_size = 2
    #                txt_color = (170, 255, 255)
    #                cv2.putText(blank_image, "start", new_pos,
    #                            cv2.FONT_HERSHEY_SIMPLEX, txt_size, txt_color, 2, cv2.LINE_AA)
    #                cv2.rectangle(blank_image, top_left, bot_right, (170, 255, 255), int(cell_width * 0.06))
    #            elif grid[row_ind,col_ind] == 1:
    #                cost = 1
    #
    #                if cost > 0:
    #                    rew_color = (86, 255, 170)
    #                else:
    #                    rew_color = (86, 86, 255)
    #
    #                cv2.rectangle(blank_image, top_left, bot_right, rew_color, int(cell_width * 0.06))
    #
    #                blank_image = blank_image.astype(np.uint8)
    #
    #                if colored_field is True:
    #                    txt_color = (255, 255, 255)
    #                else:
    #                    txt_color = rew_color
    #
    #                if abs(cost) > 9:
    #                    txt_size = 1
    #                else:
    #                    txt_size = 2
    #
    #                cv2.putText(blank_image, "goal", new_pos,
    #                            cv2.FONT_HERSHEY_SIMPLEX, txt_size, txt_color, 2, cv2.LINE_AA)
    #            else:
    #                cv2.rectangle(blank_image, top_left, bot_right, (0, 0, 0), int(cell_width * 0.03))
    #            blank_image = blank_image.astype(np.uint8)
    #
    #            if column_concatenation is None:
    #                column_concatenation = blank_image
    #            else:
    #                column_concatenation = np.concatenate((column_concatenation, blank_image), axis=1)
    #
    #        if row_concatenation is None:
    #            row_concatenation = column_concatenation
    #        else:
    #            row_concatenation = np.concatenate((row_concatenation, column_concatenation), axis=0)
    #
    #    cv2.imwrite('grid.png', row_concatenation)
    #
    #def plot_path(grid_dict,grid,move_sequence):
    #    # This function plots the grid and the path given in move_sequence. 
    #    # It also evaluates move_sequence to check for validity.
    #
    #    curr_location = [grid_dict["start_location"]["x"],grid_dict["start_location"]["y"]]
    #    i = 0
    #    path_locations = []
    #    path_locations.append([curr_location[0],curr_location[1],i])
    #    for move in move_sequence:
    #        print(curr_location)
    #        curr_cell = get_grid_cell_by_location(grid_dict,curr_location[0],curr_location[1])
    #        print(curr_cell)
    #        if move not in curr_cell["possible_actions"]:
    #            print("Illegal move. Path is invalid")
    #            return
    #        else:
    #            if move == "left":
    #                curr_location = [curr_location[0]-1,curr_location[1]]
    #            elif move == "right":
    #                curr_location = [curr_location[0]+1,curr_location[1]]
    #            elif move == "up":
    #                curr_location = [curr_location[0],curr_location[1]-1]
    #            elif move == "down":
    #                curr_location = [curr_location[0],curr_location[1]+1]
    #            else:
    #                print("Move not supported. Path is invalid.")
    #                return
    #            i = i+1
    #            path_locations.append([curr_location[0],curr_location[1],i])
    #    print(path_locations)
    #        
    #    cell_height = 300
    #    cell_width = 300
    #    top_left = (int(cell_height * 0.01), int(cell_width * 0.01))
    #    bot_right = (int(cell_height * 0.99), int(cell_width * 0.99))
    #    new_pos = (int(0.5 * cell_width), int(0.5 * cell_height))
    #
    #    row_concatenation = None
    #    for row_ind in range(grid_dict["num_rows"]):
    #        column_concatenation = None
    #
    #        for col_ind in range(grid_dict["num_cols"]):
    #
    #            blank_image = np.ones((cell_height, cell_width, 3), np.uint8)
    #            colored_field = False
    #            # walls or special states or none
    #            path_ind = None
    #            on_path = False
    #            for path_loc in path_locations:
    #                if path_loc[0] == col_ind and path_loc[1] == row_ind:
    #                    path_ind = path_loc[2]
    #                    on_path = True
    #            if np.isnan(grid[row_ind,col_ind]):
    #                blank_image = blank_image * 128
    #                blank_image.astype(np.uint8)
    #            elif on_path:
    #                blank_image[:,:,0] = 255
    #                blank_image[:,:,1] = 0
    #                blank_image[:,:,2] = 255
    #                # purple?
    #            else:
    #                # white state by default
    #                blank_image = blank_image * 255
    #                blank_image = blank_image.astype(np.uint8)
    #
    #
    #            if grid[row_ind,col_ind] == 5:
    #                # start percept has border with special color
    #                txt_size = 2
    #                txt_color = (170, 255, 255)
    #                cv2.putText(blank_image, "start", new_pos,
    #                            cv2.FONT_HERSHEY_SIMPLEX, txt_size, txt_color, 2, cv2.LINE_AA)
    #                cv2.rectangle(blank_image, top_left, bot_right, (170, 255, 255), int(cell_width * 0.06))
    #
    #            elif grid[row_ind,col_ind] == 1:
    #                cost = 1
    #
    #                if cost > 0:
    #                    rew_color = (86, 255, 170)
    #                else:
    #                    rew_color = (86, 86, 255)
    #
    #                cv2.rectangle(blank_image, top_left, bot_right, rew_color, int(cell_width * 0.06))
    #
    #                blank_image = blank_image.astype(np.uint8)
    #
    #                if colored_field is True:
    #                    txt_color = (255, 255, 255)
    #                else:
    #                    txt_color = rew_color
    #
    #                if abs(cost) > 9:
    #                    txt_size = 1
    #                else:
    #                    txt_size = 2
    #
    #                cv2.putText(blank_image, "goal", new_pos,
    #                            cv2.FONT_HERSHEY_SIMPLEX, txt_size, txt_color, 2, cv2.LINE_AA)
    #            elif on_path:
    #                txt_size = 1
    #                txt_color = (0,0,0)
    #                cv2.putText(blank_image, str(path_ind), new_pos,
    #                            cv2.FONT_HERSHEY_SIMPLEX, txt_size, txt_color, 2, cv2.LINE_AA)
    #                cv2.rectangle(blank_image, top_left, bot_right, (0, 0, 0), int(cell_width * 0.03))
    #            else:
    #                cv2.rectangle(blank_image, top_left, bot_right, (0, 0, 0), int(cell_width * 0.03))
    #            blank_image = blank_image.astype(np.uint8)
    #
    #            if column_concatenation is None:
    #                column_concatenation = blank_image
    #            else:
    #                column_concatenation = np.concatenate((column_concatenation, blank_image), axis=1)
    #
    #        if row_concatenation is None:
    #            row_concatenation = column_concatenation
    #        else:
    #            row_concatenation = np.concatenate((row_concatenation, column_concatenation), axis=0)
    #
    #    cv2.imwrite('path.png', row_concatenation)
    #
    #
    ## These are the two pre-defined grids.
    grid1 = np.array([[ 0,  0,  0, np.nan,  0,  0,],
     [ 0,  0,  0,  5,  0,  0,],
     [ 0,  0,  0,  0,  0,  0,],
     [ 0, np.nan, np.nan,  0,  0,  0,],
     [ 1, np.nan,  0, np.nan,  0,  0,],
     [ 0,  0,  0, np.nan,  0,  0,]])

    grid2 = np.array([[ 0,  0,  0,  1,  0, np.nan,],
     [ 0,  0,  0,  0,  0,  0,],
     [ 0, np.nan,  0,  0,  0,  0,],
     [ 0, np.nan,  0,  0,  0,  0,],
     [ 0, np.nan, np.nan,  0, np.nan,  0,],
     [ 0,  5,  0,  0,  0,  0,]])


   # # This block creates a 100x100 random grid, with 100 walls
   # random_grid = np.zeros((100,100))
   # walls = np.random.choice(random_grid.size, 2000, replace=False)
   # goals = np.random.choice(random_grid.size, 1, replace=False)
   # start = np.random.choice(random_grid.size, 1, replace=False)
   # random_grid.ravel()[walls] = np.nan
   # random_grid.ravel()[goals] = 1
   # random_grid.ravel()[start] = 5
#
   #print(random_grid)
    #grid = random_grid # or grid 2 or random_grid
    #
    grid_dict = create_grid_dict_from_array(random_grid)

    # Comment out the plotting utilities for large grids.
    # plot_grid(grid_dict,grid)

    start = time.time()
    ### REPLACE BELOW CODE WITH YOUR ALGORITHMS ###
    move_sequence = ["right","down","left","left","left","left","down","down"]
    ### REPLACE ABOVE CODE WITH YOUR ALGORITHMS ###
    #end = time.time()
    #print("Runtime: "+str(end-start))

    #plot_path(grid_dict,grid,move_sequence)

    ##################
    #DFS
    class StateNode:
        def __init__(self, state, parent=None, action=None, cost=0):
            self.state = state
            self.parent = parent
            self.action = action
            self.cost = cost

    def dfs_search(grid):
        # Function to check if a given (x, y) position is within the grid boundaries
        def is_within_bounds(x, y):
            return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and not np.isnan(grid[x, y])

        # Get valid neighbors (up, down, left, right) of the current state
        def get_valid_neighbors(x, y):
            neighbors = []
            for dx, dy, action in directions:
                new_x, new_y = x + dx, y + dy
                if is_within_bounds(new_x, new_y):
                    neighbors.append((new_x, new_y, action))
            return neighbors

        # Possible directions for movement: (dx, dy, action)
        directions = [(0, -1, 'left'), (0, 1, 'right'), (-1, 0, 'up'), (1, 0, 'down')]

        # Find the starting and goal positions in the grid
        start_x, start_y = np.argwhere(grid == 5)[0]
        goal_x, goal_y = np.argwhere(grid == 1)[0]

        # Create the initial node representing the starting state
        initial_node = StateNode(state=(start_x, start_y))

        # Initialize the frontier (stack) with the initial node
        frontier = [initial_node]

        # Dictionary to store explored states and their associated nodes
        explored = {(start_x, start_y): initial_node}

        # Initialize the count of explored nodes/cost.
        explored_nodes = 0
        search_cost = 0

        while frontier:
            # Pop the last node from the frontier (stack behavior for DFS)
            current_node = frontier.pop()
            # Extract the current (x, y) state
            x, y = current_node.state

            # Check if the goal state has been reached
            if (x, y) == (goal_x, goal_y):
                # Reconstruct the solution path
                solution_path = []
                while current_node:
                    if current_node.action is not None:
                        solution_path.insert(0, current_node.action)
                    current_node = current_node.parent
                return solution_path , explored_nodes, search_cost

            # Explore neighboring states
            for new_x, new_y, action in get_valid_neighbors(x, y):
                new_state = (new_x, new_y)
                new_cost = current_node.cost + 1  # Uniform cost for each step
                # Create a new node for the neighboring state
                new_node = StateNode(state=new_state, parent=current_node, action=action, cost=new_cost)
                search_cost += 1

                # Check if the neighboring state has not been explored or has a lower cost
                if new_state not in explored or new_cost < explored[new_state].cost:
                    explored[new_state] = new_node
                    frontier.append(new_node)  # Add the neighboring node to the frontier
                    explored_nodes += 1

        return [], explored_nodes, search_cost # Return an empty list if no valid path is found

    # Perform custom DFS search
    solution, explored_nodes, search_cost = (dfs_search(random_grid))

    if solution:
        total_cost = len(solution)   #this is only valid because each move costs 1
        print("Custom DFS Solution Path:", solution)
        print("Nodes Explored:", explored_nodes)
        print("Total Cost:", search_cost)
        print("Best Path Solution Cost:", total_cost)
    else:
        print("No valid path found.")
        print("Nodes Explored:", explored_nodes)
        print("Total Cost:", search_cost)


    move_sequence_dfs = solution
    end = time.time()
    runtime = str(end - start)
    print("Custom DFS Search Runtime: " + str(end - start) + " sec")
    #plot_path(grid_dict, grid, move_sequence_dfs)
    
    with open("DFS_Random100x100_Run_Data_100runs.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([iteration + 1, runtime, total_cost, explored_nodes, move_sequence_dfs])    
print("DONE")
