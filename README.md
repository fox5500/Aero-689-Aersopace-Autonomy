#README

HW1 - A Set of Search Algs to solve the following problem: 
Consider an autonomous rover in a scientific mission on a planetary surface such as Mars or
Europa. Imagine that the rover has a 2D grid describing the map of its surroundings. Some cells in the
grid are of special scientific interest. The goal of the agent is to autonomously find a path from its current
location to one of those scientifically interesting areas. Some of the cells have rough terrain (e.g., high
slope, large rocks, slippery conditions). These cells can be considered “obstacles” to be avoided at all
costs.

Algorithms: Implement breadth-first search, depth-first search, A* search, iterative deepening, and greedy best first. 

HW2 - Context: Consider an autonomous rover on the surface of Mars. The goal of the rover is to explore the terrain around it, find and retrieve a biological sample, and to return it to the starting position. To simplify things, we will assume the Mars rover is traversing a 6x6 grid. The agent’s state is specified by its location (x,y) and its orientation (left, right, up, down).

There are certain hazards in the environment:
Rocks (which destroy the rover on contact (they’re very sharp))
Storms (which destroy the rover on contact (they’re very powerful), but also degrade the air quality in adjacent (not diagonally adjacent) cells)
Grad students only: Sandy areas (which increase movement cost from 1 to 2)

The agent can take only certain actions:
Move forward
Rotate clockwise/counterclockwise by 90deg (CW/CCW)
Drill (to obtain the biological sample in the correct cell)
Spectrometer (to determine if a biological sample is present in the current cell)
Air quality (to determine if the air quality is degraded in a cell)
Traction (to determine if a cell is sandy)
Lidar (to determine if rocks are ahead and if so, how far)
