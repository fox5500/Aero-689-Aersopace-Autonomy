import clips
import random
import numpy as np
#from utils_grad import *
from project_utils import *


#def generate_bayes_nets(nrows,ncols,storm_prob,sandy_prob):
#    # This function constructs a Bayesian network for the storm probability in each cell using the provided PMFs.
#    # Return a dictionary of Bayes nets!
#    bayesnets = {}
#    for i in range(nrows):
#        for j in range(ncols):
#            ### YOUR CODE HERE ###
#            # Create a cellbn for each cell. Remember to handle corner cases! Use BayesNetCategorical rather than BayesNet.
#            ### YOUR CODE HERE ###
#            bayesnets[(i,j)] = cellbn
#    return bayesnets

def get_ppm_evidence(facts):
    # This function gets the ppm evidence from the facts.
    evidence_dict = {}
    for fact in facts:
        fact_dict = dict(fact)
        x = fact_dict["xloc"]
        y = fact_dict["yloc"]
        ppm = fact_dict["air_quality"]
        if ppm != "unknown":
            evidence_dict[(y,x)] = ppm
    return evidence_dict

def get_sandyground_evidence(facts):
    # This function gets the sandy ground evidence from the facts.
    evidence_dict = {}
    for fact in facts:
        fact_dict = dict(fact)
        x = fact_dict["xloc"]
        y = fact_dict["yloc"]
        sandy = fact_dict["sandy"]
        if sandy != "unknown":
            if sandy == "true":
                sandy = True
            if sandy == "false":
                sandy = False
            evidence_dict[(y,x)] = sandy
    return evidence_dict

#def compute_storm_probs(bayes_nets,ppm_evidence,sandyground_evidence,safe_cells):
#    # This function takes in the evidence and asks the Bayes net for storm predictions. You should override the safe cells with storm probability = 0
#    # Return a 6x6 grid with the storm probabilities.
#    # Remember that the grid is in y,x (y is our row index, x is our column index)
#    safe_cell_list = []
#    for safe_cell in safe_cells:
#        safe_cell_list.append((safe_cell[1],safe_cell[0]))
#
#    evidence_dict = {}
#    ### YOUR CODE HERE ###
#    # Fill in the evidence dict with the evidence provided to the function. Format it in a way that the elimination_ask function will like.
#    ### YOUR CODE HERE ###
#    
#    prob_grid = np.zeros(shape=(len(grid[:,0]),len(grid[0,:])))
#    ### YOUR CODE HERE ###
#    # Ask the bayes_nets for each cell what their storm probability is given the evidence.
#    ### YOUR CODE HERE ###
#    return prob_grid


def generate_fault_detection_bayes_net():
    # This function constructs a Bayesian network for fault detection using the provided PMFs.
    ### YOUR CODE HERE ### FOXY
    T, F = True, False

    fault_detection_bn = BayesNet([
        # Base probabilities for each fault
        ('BatteryFault', '', 0.001),
        ('SolarPanelDusty', '', 0.05),
        ('RockInWheel', '', 0.02),

        # Conditional probabilities for symptoms given faults
        ('lowBattery', 'BatteryFault SolarPanelDusty', 
            {(T, T): 0.99, (T, F): 0.95, (F, T): 0.75, (F, F): 0.01}),
        ('wheelFailure', 'RockInWheel', 
            {(T): 0.9, (F): 0.05}),
        ('SlowMovement', 'lowBattery wheelFailure', 
            {(T, T): 0.95, (T, F): 0.8, (F, T): 0.5, (F, F): 0.05}),
        ('WeakComms', 'lowBattery', 
            {(T): 0.8, (F): 0.3}),
        ('MovementDrifts', 'wheelFailure', 
            {(T): 0.8, (F): 0.1})
    ])
    ### YOUR CODE HERE ###
    return fault_detection_bn

def get_fault_state(last_fault_state):
    # input is the last fault state list, this function adds onto the fault state according to the PMFs
    unresolved_fault_state = []
    fault_info = []
    batteryFault = False
    solarPanelDusty = False
    rockInWheel = False
    lowBattery = False
    wheelFailure = False

    if "BatteryFault" in last_fault_state:
        unresolved_fault_state.append("BatteryFault")
        batteryFault = True
    else:
        if random.uniform(0,1) < 0.001:
            unresolved_fault_state.append("BatteryFault")
            batteryFault = True

    if "SolarPanelDusty" in last_fault_state:
        unresolved_fault_state.append("SolarPanelDusty")
        solarPanelDusty = True
    else:
        if random.uniform(0,1) < 0.05:
            unresolved_fault_state.append("SolarPanelDusty")
            solarPanelDusty = True

    if "RockInWheel" in last_fault_state:
        unresolved_fault_state.append("RockInWheel")
        rockInWheel = True
    else:
        if random.uniform(0,1) < 0.02:
            unresolved_fault_state.append("RockInWheel")
            rockInWheel = True

    if batteryFault and solarPanelDusty:
        if random.uniform(0,1) < 0.99:
            lowBattery = True
    elif batteryFault:
        if random.uniform(0,1) < 0.95:
            lowBattery = True
    elif solarPanelDusty:
        if random.uniform(0,1) < 0.75:
            lowBattery = True
    else:
        if random.uniform(0,1) < 0.01:
            lowBattery = True

    if rockInWheel:
        if random.uniform(0,1) < 0.9:
            wheelFailure = True
    else:
        if random.uniform(0,1) < 0.05:
            wheelFailure = True
    
    if lowBattery:
        if random.uniform(0,1) < 0.8:
            fault_info.append("WeakComms")
    else:
        if random.uniform(0,1) < 0.3:
            fault_info.append("WeakComms")

    if lowBattery and wheelFailure:
        if random.uniform(0,1) < 0.95:
            fault_info.append("SlowMovement")
    elif lowBattery:
        if random.uniform(0,1) < 0.8:
            fault_info.append("SlowMovement")
    elif wheelFailure:
        if random.uniform(0,1) < 0.5:
            fault_info.append("SlowMovement")
    else:
        if random.uniform(0,1) < 0.05:
            fault_info.append("SlowMovement")

    if wheelFailure:
        if random.uniform(0,1) < 0.8:
            fault_info.append("MovementDrifts")
    else:
        if random.uniform(0,1) < 0.1:
            fault_info.append("MovementDrifts")
    return unresolved_fault_state, fault_info
        
def get_fault_prediction(fault_info, fault_bn):
    # This function predicts the fault state for the 3 different fault root causes (BatteryFault, SolarPanelDusty, RockInWheel) and puts them in a dict.
    prediction_dict = {
        "BatteryFault": 0.0,
        "SolarPanelDusty": 0.0,
        "RockInWheel": 0.0
    }

    ### YOUR CODE HERE ### FOXY
    # Ask the fault_bn bayes net for the probability of each fault conditioned on the fault_info.
        # Convert fault_info into a dictionary format suitable for Bayesian queries
    evidence = {}
    for symptom in fault_info:
        evidence[symptom] = True  # Assuming all symptoms in fault_info are observed as True

    # Querying the Bayesian network for each fault
    for fault in prediction_dict.keys():
        query_result = elimination_ask(fault, evidence, fault_bn)
        # Store the probability of the fault being True
        prediction_dict[fault] = query_result[True]
    ### YOUR CODE HERE ### 

    return prediction_dict

def get_fault_reaction(fault_prediction):
    # This function reacts to the fault prediction by performing a repair task for any elevated probability.
    # FOR GRAD STUDENTS ONLY: replace this function with a smarter policy.
    fault_reaction = []
    if fault_prediction["BatteryFault"] > 0.09:
        fault_reaction.append("RestartPowerSystem")
    if fault_prediction["SolarPanelDusty"] > 0.4:
        fault_reaction.append("SolarPanelWipers")
    if fault_prediction["RockInWheel"] > 0.09:
        fault_reaction.append("WheelCleaning")
    return fault_reaction

def get_fault_resolution(fault_state,fault_reaction):
    # This function resolves the faults based on the first action in fault_reaction.
    resolution_action = fault_reaction[0]
    if resolution_action == "RestartPowerSystem" and "BatteryFault" in fault_state:
        fault_state.remove("BatteryFault")
    if resolution_action == "SolarPanelWipers" and "SolarPanelDusty" in fault_state:
        fault_state.remove("SolarPanelDusty")
    if resolution_action == "WheelCleaning" and "RockInWheel" in fault_state:
        fault_state.remove("RockInWheel")
    return fault_state


def get_nearest_rock(grid, x, y, orientation):
    # Based on the grid, this function determines the nearest rock for lidar measurements using x, y, and orientation
    init_x = x
    init_y = y
    if orientation == "up":
        while y >= 0:
            if (grid[y,x] == 2):
                return (init_y - y)
            y = y-1
    elif orientation == "down":
        while y < len(grid[:,0]):
            if (grid[y,x] == 2):
                return (y - init_y)
            y = y+1
    elif orientation == "left":
        while x >= 0:
            if (grid[y,x] == 2):
                return (init_x - x)
            x = x-1
    elif orientation == "right":
        while x < len(grid[0,:]):
            if (grid[y,x] == 2):
                return (x - init_x)
            x = x+1
    return -1

def check_storm_nearby(grid, x, y):
    # Based on the grid, this function checks if there is a nearby storm (for the air quality sensor)
    if y+1 < len(grid[:,0]) and grid[y+1,x] == 4:
        return "adjacent"
    elif y-1 >= 0 and grid[y-1,x] == 4:
        return "adjacent"
    elif x+1 < len(grid[0,:]) and grid[y,x+1] == 4:
        return "adjacent"
    elif x-1 >= 0 and grid[y,x-1] == 4:
        return "adjacent"
    elif y+1 < len(grid[:,0]) and x+1 < len(grid[0,:]) and grid[y+1,x+1] == 4:
        return "diagonal"
    elif y-1 >= 0 and x-1 >= 0 and grid[y-1,x-1] == 4:
        return "diagonal"
    elif x+1 < len(grid[0,:]) and y-1 >= 0 and grid[y-1,x+1] == 4:
        return "diagonal"
    elif x-1 >= 0 and y+1 < len(grid[:,0]) and grid[y+1,x-1] == 4:
        return "diagonal"
    else:
        return "false"
    
def check_safe_unvisited(x,y,facts):
    # This function should check if location (x, y) is safe and unvisited by querying the KB.
    safe_and_unvisited = False
    for fact in facts:
        fact_dict = dict(fact)
        if fact_dict["xloc"] == x and fact_dict["yloc"] == y and fact_dict["safe"] == "true" and fact_dict["visited"] == "false":
            safe_and_unvisited = True
    return safe_and_unvisited

def check_sandy_hidden_cell(x,y,facts):
    # This function should check if location (x, y) is sandy and unvisited by querying the KB.
    movement_cost = 1
    for fact in facts:
        fact_dict = dict(fact)
        if fact_dict["xloc"] == x and fact_dict["yloc"] == y and fact_dict["sandy"] == "true":
            movement_cost = 2
    return movement_cost

def ask_bio_unknown(x,y,facts):
    # This function queries the KB to see if it is unknown if there is a biosample in location (x, y).
    unknown = False
    for fact in facts:
        fact_dict = dict(fact)
        if fact_dict["xloc"] == x and fact_dict["yloc"] == y and fact_dict["biosample"] == "unknown":
            unknown = True
    return unknown

def ask_sandy_unknown(x,y,facts):
    # This function queries the KB to see if it is unknown if there is a biosample in location (x, y).
    unknown = False
    for fact in facts:
        fact_dict = dict(fact)
        if fact_dict["xloc"] == x and fact_dict["yloc"] == y and fact_dict["sandy"] == "unknown":
            unknown = True
    return unknown

def ask_stormnearby_unknown(x,y,facts):
    # This function queries the KB to see if it is unknown if there is a storm nearby in location (x, y).
    unknown = False
    for fact in facts:
        fact_dict = dict(fact)
        if fact_dict["xloc"] == x and fact_dict["yloc"] == y and fact_dict["storm_nearby"] == "unknown":
            unknown = True
    return unknown

def ask_cell_bio(x,y,facts):
    # This function queries the KB to see if there is a biosample in location (x, y).
    bio = False
    for fact in facts:
        fact_dict = dict(fact)
        if fact_dict["xloc"] == x and fact_dict["yloc"] == y and fact_dict["biosample"] == "true":
            bio = True
    return bio

def ask_current_pos(facts,environment):
    # This function queries the KB to see the current position of the agent.
    for fact in facts:
        if fact.template == environment.find_template('agent'):
            agent_dict = dict(fact)
    x = agent_dict["xloc"]
    y = agent_dict["yloc"]
    return [x,y]

def ask_current_orientation(facts,environment):
    # This function queries the KB to see the current orientation of the agent.
    for fact in facts:
        if fact.template == environment.find_template('agent'):
            agent_dict = dict(fact)
    orientation = agent_dict["orientation"]
    return orientation


def ask_rover_destroyed(facts,environment):
    # This function queries the KB to see if the agent is destroyed.
    for fact in facts:
        if fact.template == environment.find_template('agent'):
            agent_dict = dict(fact)
    destroyed = agent_dict["destroyed"]
    return destroyed

def ask_agent_sample(facts,environment):
    # This function queries the KB to see if the agent has retrieved the biosample.
    for fact in facts:
        if fact.template == environment.find_template('agent'):
            agent_dict = dict(fact)
    if agent_dict["sample_retrieved"] == "true":
        return True
    else:
        return False
    

def ask_if_lidar_this_loc(x,y,o,facts,environment):
    # This function asks if a lidar measurement has been taken in this location/orientation yet.
    lidar_this_loc = False
    action_dicts = []
    for fact in facts:
        if fact.template == environment.find_template('action'):
            action_dict = dict(fact)
            action_dicts.append(action_dict)
    for action in action_dicts:
        if action["xloc"] == x and action["yloc"] == y and action["orientation"] == o and action["type"] == "lidar":
            lidar_this_loc = True
    return lidar_this_loc

def ask_if_all_lidar_this_loc(x,y,facts,environment):
    # This function asks if a lidar measurement has been taken in this location/orientation yet.
    action_dicts = []
    for fact in facts:
        if fact.template == environment.find_template('action'):
            action_dict = dict(fact)
            action_dicts.append(action_dict)
    orientations = ["left","right","up","down"]
    count = 0
    for orientation in orientations:
        for action in action_dicts:
            if action["xloc"] == x and action["yloc"] == y and action["orientation"] == orientation and action["type"] == "lidar":
                count += 1
    if count == 4:
        return True
    else:
        return False


def plan_route(current, goal, allowed, sandy, game_size, sandy_flag):
    # This function tries to create a route from current to goal based on the allowed spaces provided for travel.
    if sandy_flag:
        problem = PlanRouteSandy(current, goal, allowed, sandy, game_size)
        search_result = a_star_graph_search(problem)
    else:
        problem = PlanRoute(current, goal, allowed, game_size)
        search_result = breadth_first_graph_search(problem)
    
    ### GRAD STUDENTS REPLACE BREADTH FIRST WITH BEST FIRST OR ASTAR ###
    if search_result is not None:
        return search_result.solution()
    else:
        return None

### TEMPLATES ###

cell_template = """
(deftemplate cell 
    (slot xloc)
    (slot yloc)
    (slot rock)
    (slot biosample)
    (slot storm)
    (slot storm_nearby)
    (slot sandy)
    (slot safe)
    (slot visited)
    (slot up_cell)
    (slot down_cell)
    (slot right_cell)
    (slot left_cell)
    (slot time_checked)
)
"""        

hidden_cell_template = """
(deftemplate hidden-cell 
    (slot xloc)
    (slot yloc)
    (slot rock)
    (slot biosample)
    (slot storm)
    (slot storm_nearby)
    (slot sandy)
    (slot lidar_right)
    (slot lidar_up)
    (slot lidar_down)
    (slot lidar_left)
)
"""

agent_template = """
(deftemplate agent
    (slot xloc)
    (slot yloc)
    (slot orientation)
    (slot batt_soc)
    (slot time)
    (slot sample_retrieved)
    (slot destroyed)
    (multislot loc_history)
    (multislot action_history)
)
"""

lidar_measurement_template = """
(deftemplate lidar_measurement
    (slot xloc)
    (slot yloc)
    (slot orientation)
    (slot distance)
    (slot rock_xloc)
    (slot rock_yloc)
)
"""

aq_measurement_template = """
(deftemplate aq_measurement
    (slot xloc)
    (slot yloc)
    (slot air_quality)
)
"""

bio_measurement_template = """
(deftemplate bio_measurement
    (slot xloc)
    (slot yloc)
    (slot organic)
)
"""

traction_measurement_template = """
(deftemplate traction_measurement
    (slot xloc)
    (slot yloc)
    (slot traction)
)
"""

action_template = """
(deftemplate action
    (slot type)
    (slot time)
    (slot xloc)
    (slot yloc)
    (slot orientation)
)
"""
templates = [action_template,traction_measurement_template,aq_measurement_template,bio_measurement_template,lidar_measurement_template,action_template,agent_template,
             cell_template,hidden_cell_template]

### RULES ###

lidar_rule = """
(defrule lidar_rule
    ; This rule creates a lidar_measurement fact when a lidar action is taken.
    ?act <- (action (type ?type&:(eq ?type "lidar")) (time ?action_time))
    ?ag <- (agent (xloc ?x) (yloc ?y) (orientation ?o) (time ?agent_time&:(eq ?agent_time ?action_time))(loc_history $?lh) (action_history $?ah))
    ?hidden-cell <- (hidden-cell (xloc ?cx&:(eq ?cx ?x)) (yloc ?cy&:(eq ?cy ?y)) (lidar_up ?lidar_up) (lidar_down ?lidar_down) (lidar_left ?lidar_left) (lidar_right ?lidar_right))
    =>
    (if (eq ?o "up") then (assert (lidar_measurement (xloc ?x) (yloc ?y) (orientation ?o) (distance ?lidar_up) (rock_xloc ?x) (rock_yloc (- ?y ?lidar_up)))))
    (if (eq ?o "down") then (assert (lidar_measurement (xloc ?x) (yloc ?y) (orientation ?o) (distance ?lidar_down) (rock_xloc ?x) (rock_yloc (+ ?y ?lidar_down)))))
    (if (eq ?o "left") then (assert (lidar_measurement (xloc ?x) (yloc ?y) (orientation ?o) (distance ?lidar_left) (rock_xloc (- ?x ?lidar_left)) (rock_yloc ?y))))
    (if (eq ?o "right") then (assert (lidar_measurement (xloc ?x) (yloc ?y) (orientation ?o) (distance ?lidar_right) (rock_xloc (+ ?x ?lidar_right)) (rock_yloc ?y))))
    (bind ?t (+ ?agent_time 1))
    (modify ?ag (time ?t) (action_history (create$ $?ah ?type)))
)
"""
rotate_ccw_rule = """
(defrule rotate_ccw_rule
    ; This rule rotates the agent counterclockwise when a rotate_ccw action is taken.
    ?act <- (action (type ?type&:(eq ?type "rotate_ccw")) (time ?action_time))
    ?ag <- (agent (xloc ?x) (yloc ?y) (orientation ?o) (time ?agent_time&:(eq ?agent_time ?action_time)) (loc_history $?lh) (action_history $?ah))
    =>
    (if (eq ?o "up") then (bind ?new_o "left"))
    (if (eq ?o "down") then (bind ?new_o "right"))
    (if (eq ?o "left") then (bind ?new_o "down"))
    (if (eq ?o "right") then (bind ?new_o "up"))
    (bind ?t (+ ?agent_time 1))
    (modify ?ag (time ?t) (orientation ?new_o) (action_history (create$ $?ah ?type)))
)
"""

rotate_cw_rule = """
(defrule rotate_cw_rule
    ; This rule rotates the agent clockwise when a rotate_cw action is taken.
    ?act <- (action (type ?type&:(eq ?type "rotate_cw")) (time ?action_time))
    ?ag <- (agent (xloc ?x) (yloc ?y) (orientation ?o) (time ?agent_time&:(eq ?agent_time ?action_time)) (loc_history $?lh) (action_history $?ah))
    =>
    (if (eq ?o "up") then (bind ?new_o "right"))
    (if (eq ?o "down") then (bind ?new_o "left"))
    (if (eq ?o "left") then (bind ?new_o "up"))
    (if (eq ?o "right") then (bind ?new_o "down"))
    (bind ?t (+ ?agent_time 1))
    (modify ?ag (time ?t) (orientation ?new_o) (action_history (create$ $?ah ?type)))
)
"""

forward_rule = """
(defrule forward_rule
    ; This rule moves the agent forward when a forward action is taken.
    ?act <- (action (type ?type&:(eq ?type "forward")) (time ?action_time))
    ?ag <- (agent (xloc ?x) (yloc ?y) (orientation ?o) (time ?agent_time&:(eq ?agent_time ?action_time)) (loc_history $?lh) (action_history $?ah))
    =>
    (if (eq ?o "up") then (bind ?new_x ?x) (bind ?new_y (- ?y 1)))
    (if (eq ?o "down") then (bind ?new_x ?x) (bind ?new_y (+ ?y 1)))
    (if (eq ?o "left") then (bind ?new_x (- ?x 1)) (bind ?new_y ?y))
    (if (eq ?o "right") then (bind ?new_x (+ ?x 1)) (bind ?new_y ?y))
    (bind ?t (+ ?agent_time 1))
    (modify ?ag (time ?t) (xloc ?new_x) (yloc ?new_y) (action_history (create$ $?ah ?type)))
)
"""

unvisited_cell_rule = """
(defrule unvisited_cell_rule
    ; This rule sets the cell to visited if it was previously unvisited. It also checks for agent destruction.
    ?ag <- (agent (xloc ?x) (yloc ?y) (orientation ?o) (loc_history $?lh) (action_history $?ah))
    ?hidden-cell <- (hidden-cell (xloc ?hcx&:(eq ?hcx ?x)) (yloc ?hcy&:(eq ?hcy ?y)) (rock ?rock) (storm ?storm))
    ?cell <- (cell (xloc ?cx&:(eq ?cx ?x)) (yloc ?cy&:(eq ?cy ?y))  (visited "false"))
    =>
    (bind ?destroyed "false")
    (if (eq ?rock "true") then (bind ?destroyed "rock"))
    (if (eq ?storm "true") then (bind ?destroyed "storm"))
    (modify ?cell (rock ?rock) (storm ?storm) (visited "true") (safe "true"))
    (modify ?ag (destroyed ?destroyed))
)
"""

air_quality_rule="""
(defrule air_quality_rule
    ; This rule creates an aq_measurement when an air_quality action is taken.
    ?act <- (action (type ?type&:(eq ?type "air_quality")) (time ?action_time))
    ?ag <- (agent (xloc ?x) (yloc ?y) (orientation ?o) (time ?agent_time&:(eq ?agent_time ?action_time))(loc_history $?lh) (action_history $?ah))
    ?hidden-cell <- (hidden-cell (xloc ?cx&:(eq ?cx ?x)) (yloc ?cy&:(eq ?cy ?y)) (storm_nearby ?storm_nearby) (sandy ?sandy))
    =>
    (bind ?roll (/ (random 0 1000000) 1000000))
    (if (and (eq ?storm_nearby "adjacent") (eq ?sandy "true") (>= ?roll 0.05)) then (bind ?ppm "High"))
    (if (and (eq ?storm_nearby "adjacent") (eq ?sandy "true") (< ?roll 0.05) (>= ?roll 0.01)) then (bind ?ppm "Medium"))
    (if (and (eq ?storm_nearby "adjacent") (eq ?sandy "true") (< ?roll 0.01)) then (bind ?ppm "Low"))
    (if (and (eq ?storm_nearby "adjacent") (eq ?sandy "false") (>= ?roll 0.1)) then (bind ?ppm "High"))
    (if (and (eq ?storm_nearby "adjacent") (eq ?sandy "false") (< ?roll 0.1) (>= ?roll 0.01)) then (bind ?ppm "Medium"))
    (if (and (eq ?storm_nearby "adjacent") (eq ?sandy "false") (< ?roll 0.01)) then (bind ?ppm "Low"))
    (if (and (eq ?storm_nearby "diagonal") (eq ?sandy "true") (>= ?roll 0.7)) then (bind ?ppm "High"))
    (if (and (eq ?storm_nearby "diagonal") (eq ?sandy "true") (< ?roll 0.7) (>= ?roll 0.2)) then (bind ?ppm "Medium"))
    (if (and (eq ?storm_nearby "diagonal") (eq ?sandy "true") (< ?roll 0.2)) then (bind ?ppm "Low"))
    (if (and (eq ?storm_nearby "diagonal") (eq ?sandy "false") (>= ?roll 0.75)) then (bind ?ppm "High"))
    (if (and (eq ?storm_nearby "diagonal") (eq ?sandy "false") (< ?roll 0.75) (>= ?roll 0.25)) then (bind ?ppm "Medium"))
    (if (and (eq ?storm_nearby "diagonal") (eq ?sandy "false") (< ?roll 0.25)) then (bind ?ppm "Low"))
    (if (and (eq ?storm_nearby "false") (eq ?sandy "true") (>= ?roll 0.99)) then (bind ?ppm "High"))
    (if (and (eq ?storm_nearby "false") (eq ?sandy "true") (< ?roll 0.99) (>= ?roll 0.75)) then (bind ?ppm "Medium"))
    (if (and (eq ?storm_nearby "false") (eq ?sandy "true") (< ?roll 0.75)) then (bind ?ppm "Low"))
    (if (and (eq ?storm_nearby "false") (eq ?sandy "false") (>= ?roll 0.99)) then (bind ?ppm "High"))
    (if (and (eq ?storm_nearby "false") (eq ?sandy "false") (< ?roll 0.99) (>= ?roll 0.90)) then (bind ?ppm "Medium"))
    (if (and (eq ?storm_nearby "false") (eq ?sandy "false") (< ?roll 0.90)) then (bind ?ppm "Low"))
    (assert (aq_measurement (xloc ?x) (yloc ?y) (air_quality ?ppm)))
    (bind ?t (+ ?agent_time 1))
    (modify ?ag (time ?t) (action_history (create$ $?ah ?type)))
)
"""

bio_rule="""
(defrule bio_rule
    ; This rule creates a bio_measurement when a bio action is taken.
    ?act <- (action (type ?type&:(eq ?type "spectrometer")) (time ?action_time))
    ?ag <- (agent (xloc ?x) (yloc ?y) (orientation ?o) (time ?agent_time&:(eq ?agent_time ?action_time))(loc_history $?lh) (action_history $?ah))
    ?hidden-cell <- (hidden-cell (xloc ?cx&:(eq ?cx ?x)) (yloc ?cy&:(eq ?cy ?y)) (biosample ?biosample))
    =>
    
    (if (eq ?biosample "true") then (assert (bio_measurement (xloc ?x) (yloc ?y) (organic "true"))) 
     else (assert (bio_measurement (xloc ?x) (yloc ?y) (organic "false"))))
    (bind ?t (+ ?agent_time 1))
    (modify ?ag (time ?t) (action_history (create$ $?ah ?type)))
)
"""

traction_rule="""
(defrule traction_rule
    ; This rule creates a traction_measurement when a traction action is taken.
    ?act <- (action (type ?type&:(eq ?type "traction")) (time ?action_time))
    ?ag <- (agent (xloc ?x) (yloc ?y) (orientation ?o) (time ?agent_time&:(eq ?agent_time ?action_time))(loc_history $?lh) (action_history $?ah))
    ?hidden-cell <- (hidden-cell (xloc ?cx&:(eq ?cx ?x)) (yloc ?cy&:(eq ?cy ?y)) (sandy ?sandy))
    =>
    (if (eq ?sandy "true") then (assert (traction_measurement (xloc ?x) (yloc ?y) (traction "poor"))) 
     else (assert (traction_measurement (xloc ?x) (yloc ?y) (traction "good"))))
    (bind ?t (+ ?agent_time 1))
    (modify ?ag (time ?t) (action_history (create$ $?ah ?type)))
)
"""

storm_nearby_rule="""
(defrule storm_nearby_rule
    ; This rule checks to see if a storm is nearby a cell based on the air quality measurements.
    ?meas <- (aq_measurement (xloc ?x) (yloc ?y) (air_quality ?aq))
    ?cell <- (cell (xloc ?cx&:(eq ?cx ?x)) (yloc ?cy&:(eq ?cy ?y)))
    =>
    (bind ?sn "known")
    (modify ?cell (storm_nearby ?sn))
)
"""

traction_meas_rule="""
(defrule traction_meas_rule
    ; This rule checks to see if a cell is sandy based on the traction measurements.
    ?meas <- (traction_measurement (xloc ?x) (yloc ?y) (traction ?tr))
    ?cell <- (cell (xloc ?cx&:(eq ?cx ?x)) (yloc ?cy&:(eq ?cy ?y)))
    =>
    (bind ?s "unknown")
    (if (eq ?tr "good") then (bind ?s "false"))
    (if (eq ?tr "poor") then (bind ?s "true"))
    (modify ?cell (sandy ?s))
)
"""

biosample_meas_rule="""
(defrule biosample_meas_rule
    ; This rule checks to see if a cell has a biosample based on the bio measurements.
    ?meas <- (bio_measurement (xloc ?x) (yloc ?y) (organic ?or))
    ?cell <- (cell (xloc ?cx&:(eq ?cx ?x)) (yloc ?cy&:(eq ?cy ?y)) (biosample "unknown"))
    =>
    (bind ?bio "unknown")
    (if (eq ?or "true") then (bind ?bio "true"))
    (if (eq ?or "false") then (bind ?bio "false"))
    (modify ?cell (biosample ?bio))
)
"""

lidar_update_rule="""
(defrule lidar_update_rule
    ; This rule checks to see if a cell has a rock based on the lidar measurements.
    ?meas <- (lidar_measurement (distance ?d&:(neq ?d -1)) (rock_xloc ?rx) (rock_yloc ?ry))
    ?cell <- (cell (xloc ?cx&:(eq ?cx ?rx)) (yloc ?cy&:(eq ?cy ?ry)))
    =>
    (modify ?cell (rock "true") (safe "false"))
)
"""

lidar_y_clear_rule="""
(defrule lidar_y_clear_rule
    ; This cell infers that there is no rock based on lidar measurements returning a -1 value.
    ?meas <- (lidar_measurement (xloc ?x) (yloc ?y) (orientation ?o&:(or (eq ?o "up") (eq ?o "down"))) (distance ?d&:(eq ?d -1)))
    ?cell <- (cell (xloc ?cx&:(eq ?cx ?x)) (yloc ?cy) (rock "unknown"))
    =>
    (bind ?rock "unknown")
    (if (eq ?o "up")
     then (if (< ?cy ?y)
           then (bind ?rock "false")
          )
    )
    (if (eq ?o "down")
     then (if (> ?cy ?y)
           then (bind ?rock "false")
          )
    )
    (modify ?cell (rock ?rock))
)
"""

lidar_x_clear_rule="""
(defrule lidar_x_clear_rule
    ; This cell infers that there is no rock based on lidar measurements returning a -1 value.
    ?meas <- (lidar_measurement (xloc ?x) (yloc ?y) (orientation ?o&:(or (eq ?o "left") (eq ?o "right"))) (distance ?d&:(eq ?d -1)))
    ?cell <- (cell (xloc ?cx) (yloc ?cy&:(eq ?cy ?y)) (rock "unknown"))
    =>
    (bind ?rock "unknown")
    (if (eq ?o "left")
     then (if (< ?cx ?x)
           then (bind ?rock "false")
          )
    )
    (if (eq ?o "right")
     then (if (> ?cx ?x)
           then (bind ?rock "false")
          )
    )
    (modify ?cell (rock ?rock))
)
"""

drill_rule="""
(defrule drill_rule
    ; This rule retrieves a sample if a drill action is taken in a cell with a biosample.
    ?act <- (action (type ?type&:(eq ?type "drill")) (time ?action_time))
    ?ag <- (agent (xloc ?x) (yloc ?y) (orientation ?o) (time ?agent_time&:(eq ?agent_time ?action_time)) (loc_history $?lh) (action_history $?ah))
    ?hidden-cell <- (hidden-cell (xloc ?cx&:(eq ?cx ?x)) (yloc ?cy&:(eq ?cy ?y)) (biosample ?bio))
    =>

    (bind ?sample "false")
    (if (eq ?bio "true") then (bind ?sample "true"))
    (bind ?t (+ ?agent_time 1))
    (modify ?ag (sample_retrieved ?sample) (time ?t) (action_history (create$ $?ah ?type)))
)
"""

up_cell_rule="""
(defrule up_cell_rule
    ; This rule sets the cell that is above the cell at location (x, y).
    ?cell <- (cell (xloc ?x) (yloc ?y) (up_cell nil))
    ?up-cell <- (cell (xloc ?cx&:(eq ?x ?cx)) (yloc ?cy&:(eq (- ?y 1) ?cy)))
    =>
    (modify ?cell (up_cell $?up-cell))
)
"""

down_cell_rule="""
(defrule down_cell_rule
    ; This rule sets the cell that is below the cell at location (x, y).
    ?cell <- (cell (xloc ?x) (yloc ?y) (down_cell nil))
    ?down-cell <- (cell (xloc ?cx&:(eq ?x ?cx)) (yloc ?cy&:(eq (+ ?y 1) ?cy)))
    =>
    (modify ?cell (down_cell ?down-cell))
)
"""

left_cell_rule="""
(defrule left_cell_rule
    ; This rule sets the cell that is to the left of the cell at location (x, y).
    ?cell <- (cell (xloc ?x) (yloc ?y) (left_cell nil))
    ?left-cell <- (cell (xloc ?cx&:(eq (- ?x 1) ?cx)) (yloc ?cy&:(eq ?y ?cy)))
    =>
    (modify ?cell (left_cell ?left-cell))
)
"""

right_cell_rule="""
(defrule right_cell_rule
    ; This rule sets the cell that is to the right of the cell at location (x, y).
    ?cell <- (cell (xloc ?x) (yloc ?y) (right_cell nil))
    ?right-cell <- (cell (xloc ?cx&:(eq (+ ?x 1) ?cx)) (yloc ?cy&:(eq ?y ?cy)))
    =>
    (modify ?cell (right_cell ?right-cell))
)
"""

storm_safe_rule="""
(defrule storm_safe_rule
    ; This rule infers that there is no storm in a cell based on two adjacent conflicting air quality measurements.
    ?ag <- (agent (xloc ?x) (yloc ?y) (orientation ?o) (time ?agent_time) (loc_history $?lh) (action_history $?ah))
    ?cell <- (cell (xloc ?cx) (yloc ?cy) (right_cell ?rc) (left_cell ?lc) (up_cell ?uc) (down_cell ?dc) (storm "unknown") (time_checked ?t&:(neq ?agent_time ?t)))
    =>
    (bind ?storm "unknown")
    (bind ?safe "unknown")
    (bind ?lcsnb "unknown")
    (bind ?rcsnb "unknown")
    (bind ?ucsnb "unknown")
    (bind ?dcsnb "unknown")
    (if (neq ?lc nil)
     then (if (eq (fact-slot-value ?lc storm_nearby) "false")
           then (bind ?lcsnb "false")
          )
    )
    (if (neq ?rc nil)
     then (if (eq (fact-slot-value ?rc storm_nearby) "false")
           then (bind ?rcsnb "false")
          )
    )
    (if (neq ?uc nil)
     then (if (eq (fact-slot-value ?uc storm_nearby) "false")
           then (bind ?ucsnb "false")
          )
    )
    (if (neq ?dc nil)
     then (if (eq (fact-slot-value ?dc storm_nearby) "false")
           then (bind ?dcsnb "false")
          )
    )
    (if (or (eq ?lcsnb "false") (eq ?rcsnb "false") (eq ?ucsnb "false") (eq ?dcsnb "false"))
     then (bind ?storm "false")
    )
    (bind ?new_t ?agent_time)
    (modify ?cell (storm ?storm) (time_checked ?new_t))
)
"""

safe_cell_rule="""
(defrule safe_cell_rule
    ; This rule checks if the cell has no rock or storm and if so, sets the cell to safe.
    ?cell <- (cell (storm "false") (rock "false") (safe "unknown"))
    =>
    (modify ?cell (safe "true"))
)
"""
rules = [lidar_rule,rotate_ccw_rule,rotate_cw_rule,forward_rule,air_quality_rule,traction_rule,bio_rule,storm_nearby_rule,traction_meas_rule,biosample_meas_rule,
         unvisited_cell_rule,lidar_update_rule,drill_rule,left_cell_rule,right_cell_rule,up_cell_rule,down_cell_rule,storm_safe_rule,lidar_x_clear_rule,lidar_y_clear_rule,
         safe_cell_rule]

def new_game(templates,rules):
    # This function creates a new game, which consists of a random 6x6 grid with 3 rocks, 3 sandy cells, 2 storms, 1 goal and 1 start cell.
    # It also loads all of the rules and templates into CLIPS.
    environment = clips.Environment()
    sandy_environment = clips.Environment()
    for template in templates:
        environment.build(template)
        sandy_environment.build(template)
    for rule in rules:
        environment.build(rule)
        sandy_environment.build(rule)

    # Generate random grid.
    random_grid = np.zeros((6,6))
    stuff_cells = np.random.choice(random_grid.size, 10, replace=False) # 10 = 3 rocks + 3 sandy + 2 storms + 1 goal + 1 start
    # 1 is start, 2 is rocks, 3 is sandy, 4 is storm, 5 is organic sample
    random_grid.ravel()[stuff_cells[0]] = 1
    random_grid.ravel()[stuff_cells[1:4]] = 2
    random_grid.ravel()[stuff_cells[4:7]] = 3
    random_grid.ravel()[stuff_cells[7:9]] = 4
    random_grid.ravel()[stuff_cells[9]] = 5
    start_x = None
    start_y = None
    print(random_grid)

    # populate hidden cells (the real environment if the problem was fully observable)
    for i in range(len(random_grid[:,0])):
        for j in range(len(random_grid[0,:])):
            x = j
            y = i
            # row i, column j
            # print("Row "+str(i)+", column "+str(j)+" is "+str(random_grid[i,j]))
            lidar_up = get_nearest_rock(random_grid,x,y,"up")
            lidar_down = get_nearest_rock(random_grid,x,y,"down")
            lidar_left = get_nearest_rock(random_grid,x,y,"left")
            lidar_right = get_nearest_rock(random_grid,x,y,"right")
            storm_nearby = check_storm_nearby(random_grid,x,y)
            
            if random_grid[y,x] == 4:
                storm = "true"
            else:
                storm = "false"
            if random_grid[y,x] == 3:
                sandy = "true"
            else:
                sandy = "false"
            if random_grid[y,x] == 2:
                rock = "true"
            else:
                rock = "false"
            if random_grid[y,x] == 5:
                biosample = "true"
            else:
                biosample = "false"
            if random_grid[y,x] == 1:
                start_x = x
                start_y = y
            # assert a new fact through its template
            hidden_cell_template = environment.find_template('hidden-cell')
            fact = hidden_cell_template.assert_fact(xloc=x,
                                        yloc=y,
                                        rock=rock,
                                        biosample=biosample,
                                        storm=storm,
                                        storm_nearby=storm_nearby,
                                        sandy=sandy,
                                        lidar_up=lidar_up,
                                        lidar_down=lidar_down,
                                        lidar_left=lidar_left,
                                        lidar_right=lidar_right
                                        )
            sandy_hidden_cell_template = sandy_environment.find_template('hidden-cell')
            sandy_fact = sandy_hidden_cell_template.assert_fact(xloc=x,
                                        yloc=y,
                                        rock=rock,
                                        biosample=biosample,
                                        storm=storm,
                                        storm_nearby=storm_nearby,
                                        sandy=sandy,
                                        lidar_up=lidar_up,
                                        lidar_down=lidar_down,
                                        lidar_left=lidar_left,
                                        lidar_right=lidar_right
                                        )

    # populate agent's cell KB (what the agent knows)
    for i in range(len(random_grid[:,0])):
        for j in range(len(random_grid[0,:])):
            x = j
            y = i
            if not (x == start_x and y == start_y):
                storm = "unknown"
                storm_nearby = "unknown"
                sandy = "unknown"
                rock = "unknown"
                safe = "unknown"
                biosample = "unknown"
                visited = "false"
            else:
                storm = "false"
                sandy = "false"
                storm_nearby = "unknown"
                rock = "false"
                biosample = "false"
                safe = "true"
                visited = "true"
            cell_template = environment.find_template('cell')
            cell_fact = cell_template.assert_fact(xloc=x,
                                        yloc=y,
                                        rock=rock,
                                        biosample=biosample,
                                        storm=storm,
                                        storm_nearby=storm_nearby,
                                        sandy=sandy,
                                        safe=safe,
                                        visited=visited,
                                        time_checked=0
                                        )
            sandy_cell_template = sandy_environment.find_template('cell')
            sandy_cell_fact = sandy_cell_template.assert_fact(xloc=x,
                                        yloc=y,
                                        rock=rock,
                                        biosample=biosample,
                                        storm=storm,
                                        storm_nearby=storm_nearby,
                                        sandy=sandy,
                                        safe=safe,
                                        visited=visited,
                                        time_checked=0
                                        )

    # Populate initial agent state.
    agent_template = environment.find_template('agent')
    agent = agent_template.assert_fact(xloc=start_x,
                                    yloc=start_y,
                                    orientation="right",
                                    batt_soc=1,
                                    time=0,
                                    sample_retrieved="false",
                                    destroyed="false",
                                    loc_history=[],
                                    action_history=[]
                                    )
    sandy_agent_template = sandy_environment.find_template('agent')
    sandy_agent = sandy_agent_template.assert_fact(xloc=start_x,
                                    yloc=start_y,
                                    orientation="right",
                                    batt_soc=1,
                                    time=0,
                                    sample_retrieved="false",
                                    destroyed="false",
                                    loc_history=[],
                                    action_history=[]
                                    )
    return environment, sandy_environment, random_grid, start_x, start_y

def hybrid_agent(environment,grid,start_x,start_y,sandy):
    # Grad students will need to augment this agent with a way to track the movement actions taken in sandy vs. non-sandy areas.
    sim_length = 1000
    rover_time = 0 # rover_time is for the rules to function properly
    sim_time = 0 # sim_time is so we can keep track of fault detection performance
    total_movement_cost = 0
    #cell_bayes_nets = generate_bayes_nets(len(grid[:,0]),len(grid[0,:]),2/36,3/36)
    fault_bayes_net = generate_fault_detection_bayes_net()
    last_fault_state = []

    while sim_time < sim_length:
        # Get all of the facts associated with cells.
        cell_facts = []
        for fact in environment.facts():
            if fact.template == environment.find_template('cell'):
                cell_facts.append(fact)
        # Get all of the facts associated with hidden cells. (Only used for sandy check)
        hidden_cell_facts = []
        for fact in environment.facts():
            if fact.template == environment.find_template('hidden-cell'):
                hidden_cell_facts.append(fact)
        # Get all facts.
        facts = list(environment.facts())
        plan = []

        # Get current rover conditions from KB.
        current_pos = ask_current_pos(facts,environment)
        x = current_pos[0]
        y = current_pos[1]
        destroyed = ask_rover_destroyed(facts,environment)
        orientation = ask_current_orientation(facts,environment)
        rover_pos = RoverPosition(x,y,orientation)

        # Check if the rover has been destroyed, end the game if so.
        if destroyed == "rock" or destroyed == "storm":
            print("Game over.")
            print("Destroyed by: "+destroyed)
            print(storm_probs)
            print("^ storm probs")
            if destroyed == "rock":
                print("Location"+ str((y,x)))
                print("Orientation: "+orientation)
                print("Final grid observations: ")
                for fact in env.facts():
                    if fact.template == env.find_template('cell'):
                        d = dict(fact)
                        exclude_keys = ['up_cell', 'down_cell','right_cell','left_cell']
                        new_d = {k: d[k] for k in set(list(d.keys())) - set(exclude_keys)}
                        print(", ".join([key+": "+str(value) for key, value in sorted(new_d.items(), key=lambda x: x[0])]))
                exit()

            return 1, total_movement_cost, sim_time

        # Check if the agent has retrieved the sample and is back in the start position.
        if ask_agent_sample(facts,environment) and x == start_x and y == start_y:
            print("You win!")
            return 0, total_movement_cost, sim_time
        
        # Handle faults

        unresolved_fault_state, fault_info = get_fault_state(last_fault_state) # fault state is the root cause and symptom, fault info is the observed issue
        # if len(unresolved_fault_state) != 0:
        #     print("Fault state: "+str(unresolved_fault_state))
        #     print("Fault info: "+str(fault_info))

        fault_prediction = get_fault_prediction(fault_info, fault_bayes_net) # Predicts fault states using Bayes net
        # print(fault_prediction)

        fault_reaction = get_fault_reaction(fault_prediction) # Reacts to prediction
        # print(fault_reaction)
        if len(fault_reaction) != 0:
            resolved_fault_state = get_fault_resolution(unresolved_fault_state, fault_reaction) # Tries to resolve faults using reaction
            #print("Trying to resolve faults, action: " + fault_reaction[0])
        else:
            resolved_fault_state = unresolved_fault_state
        last_fault_state = resolved_fault_state

        _, current_fault_conditions = get_fault_state(resolved_fault_state)
        
        # Get a list of safe cells that we can traverse without worry.
        safe_points = list()
        for cell in cell_facts:
            cell_dict = dict(cell)
            if cell_dict["safe"] == "true":
                safe_points.append([cell_dict["xloc"],cell_dict["yloc"]])

        # Get a list of sandy cells.
        sandy_points = list()
        for cell in cell_facts:
            cell_dict = dict(cell)
            if cell_dict["sandy"] == "true":
                sandy_points.append([cell_dict["xloc"],cell_dict["yloc"]])

        # Check if agent has retrieved the sample.
        if len(plan) == 0:
            if ask_agent_sample(facts,environment):
                #print("Sample acquired!")
                goals = list()
                goals.append([start_x, start_y])
                actions = plan_route(rover_pos, goals[0], safe_points, sandy_points, len(grid[:,0]),sandy)
                plan.extend(actions)

        # Check if the agent is in a cell with the biosample. If so, drill.
        if len(plan) == 0:
            if ask_cell_bio(x,y,cell_facts):
                action = "drill"
                plan.append(action)

        # Check if lidar has been done in this cell/orientation. If not, lidar.
        if len(plan) == 0:
            if not ask_if_lidar_this_loc(x,y,ask_current_orientation(facts,environment),facts,environment):
                action = "lidar"
                plan.append(action)
            
        if len(plan) == 0:
            if not ask_if_all_lidar_this_loc(x,y,facts,environment):
                action = "rotate_ccw"
                plan.append(action)
        
        # Check if traction has been measured in this cell. If not, traction.
        if len(plan) == 0:
            if ask_sandy_unknown(x,y,cell_facts):
                action = "traction"
                plan.append(action)

        # Check if air quality/nearby storm status has been measured in this cell. If not, air_quality.
        if len(plan) == 0:
            if ask_stormnearby_unknown(x,y,cell_facts):
                action = "air_quality"
                plan.append(action)

        # Check if the spectrometer has been used in this cell. If not, spectrometer.
        if len(plan) == 0:
            if ask_bio_unknown(x,y,cell_facts):
                action = "spectrometer"
                plan.append(action)

        # Using air quality measurements, travel to the safest cell.
        if len(plan) == 0:
            aq_meas_facts = []
            for fact in environment.facts():
                if fact.template == environment.find_template('aq_measurement'):
                    aq_meas_facts.append(fact)
            ppm_evidence = get_ppm_evidence(aq_meas_facts) # ppm_evidence is a dictionary where keys are locations and values are ppms
            sandyground_evidence = get_sandyground_evidence(cell_facts) # sandyground_evidence is a dictionary where keys are locations and values are true/false
            #storm_probs = compute_storm_probs(cell_bayes_nets,ppm_evidence,sandyground_evidence, safe_points) # storm_probs is a 2D array matching the grid
            not_unsafe = list()
            for cell in cell_facts:
                cell_dict = dict(cell)
                if not cell_dict["safe"] == "false" and cell_dict["visited"] == "false" and not (cell_dict["xloc"] == x and cell_dict["yloc"] == y):
                    not_unsafe.append([cell_dict["xloc"],cell_dict["yloc"]])
            while len(not_unsafe) > 1:
                # find safest cell by looping, if safest cell is unreachable then remove from not_unsafe list
                minimum = 1.0
                safest_point = not_unsafe[0]
                for point in not_unsafe:
                    storm_prob = ppm_evidence]
                    if storm_prob < minimum:
                        safest_point = point
                        minimum = storm_prob
                safe_points.append(safest_point)
                temp = plan_route(rover_pos, safest_point, safe_points, sandy_points, len(grid[:,0]),sandy)
                safe_points.remove(safest_point)
                if temp is not None:
                    plan.extend(temp)
                    break
                else:
                    not_unsafe.remove(safest_point)

        #NOTE: You can uncomment this code if you want to test your fault detection code without having your storm_probs working yet.
        if len(plan) == 0:
            not_unsafe = list()
            for cell in cell_facts:
                cell_dict = dict(cell)
                if not cell_dict["safe"] == "false" and cell_dict["visited"] == "false" and not (cell_dict["xloc"] == x and cell_dict["yloc"] == y):
                    not_unsafe.append([cell_dict["xloc"],cell_dict["yloc"]])
            if len(not_unsafe) > 0:
                valid_temps = []
                for not_unsafe_point in not_unsafe:
                    safe_points.append(not_unsafe_point)
                    temp = plan_route(rover_pos, not_unsafe_point, safe_points, sandy_points, len(grid[:,0]),sandy)
                    safe_points.remove(not_unsafe_point)
                    if temp is not None:
                        valid_temps.append(temp)
                if len(valid_temps) > 0:
                    temp = random.choice(valid_temps)
                    plan.extend(temp)

        # If there are no unknown spots to explore, all of the unvisited spots are known to be unsafe, the goal must be blocked off.
        if len(plan) == 0:
            print("No actions to take!")
            print("Game over.")
            return 1, total_movement_cost, sim_time
        
        # Execute the first action in the plan.
        action = plan[0]

        # Translating from plan_route terminology to clips rule terminology.
        if action == "Forward":
            action = "forward"
        elif action == "Turnright":
            action = "rotate_cw"
        elif action == "Turnleft":
            action = "rotate_ccw"
        if "WeakComms" in current_fault_conditions and (action == "air_quality" or action == "spectrometer" or action == "traction" or action == "lidar"):
            sim_time += 2
        elif "MovementDrifts" in current_fault_conditions and (action == "forward"):
            if random.uniform(0,1) < 0.5:
                action = "rotate_ccw"
            else:
                action = "rotate_cw"
            sim_time += 1
        elif "SlowMovement" in current_fault_conditions and (action == "forward" or action == "rotate_ccw" or action == "rotate_cw"):
            sim_time += 2
        else:
            sim_time += 1

        #print("Current time: "+str(t))
        #print("Current position: "+str(current_pos))
        #print("Action taken: "+str(action))

        plan = plan[1:]

        # Update the KB with the action taken. Run the rule environment so that any new firings can occur.
        action_template = environment.find_template('action')
        action_template.assert_fact(type=action,time=rover_time,xloc=x,yloc=y,orientation=ask_current_orientation(facts,environment))
        if action == "forward" or action == "rotate_cw" or action == "rotate_ccw":
            movement_cost = check_sandy_hidden_cell(x,y,hidden_cell_facts)
            total_movement_cost += movement_cost
        environment.run()
        rover_time += 1
    print("Ran out of time.")
    return 1, total_movement_cost, sim_time


# This code runs 100 trials and tracks the number of victories and the average movement cost.
sum = 0
sims = 100
mvmt_cost_sum = 0
sandy_mvmt_cost_sum = 0
sim_time_sum = 0
for i in range(sims):
    env, sandy_env, grid, start_x, start_y = new_game(templates,rules)
    result, mvmt_cost, elapsed_sim_time = hybrid_agent(env,grid,start_x,start_y,False)
    #result, sandy_mvmt_cost = hybrid_agent(sandy_env,grid,start_x,start_y,True)

    sum += result
    mvmt_cost_sum += mvmt_cost
    sim_time_sum += elapsed_sim_time
    #sandy_mvmt_cost_sum += sandy_mvmt_cost
print("Number of victories: "+str(sims-sum))
print("Movement cost average: "+str(mvmt_cost_sum/sims))
print("Elapsed sim time average: "+str(sim_time_sum/sims))
#print("Sandy movement cost average: "+str(sandy_mvmt_cost_sum/sims))

# This prints out the working memory for the last trial run.
print("Final grid observations: ")
for fact in env.facts():
    if fact.template == env.find_template('cell'):
        d = dict(fact)
        exclude_keys = ['up_cell', 'down_cell','right_cell','left_cell']
        new_d = {k: d[k] for k in set(list(d.keys())) - set(exclude_keys)}
        print(", ".join([key+": "+str(value) for key, value in sorted(new_d.items(), key=lambda x: x[0])]))