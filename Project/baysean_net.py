from project_utils import *

T, F = True, False

rover_faults = BayesNet([
    # Base probabilities for each fault
    ('BatterySystemFault', '', 0.001),
    ('SolarPanelDust', '', 0.05),
    ('RocksInWheels', '', 0.02),

    # Conditional probabilities for symptoms given faults
    ('LowPower', 'BatterySystemFault SolarPanelDust', 
        {(T, T): 0.99, (T, F): 0.95, (F, T): 0.75, (F, F): 0.01}),
    ('WheelSubsystemFailure', 'RocksInWheels', 
        {(T): 0.9, (F): 0.05}), 
    ('SlowMovement', 'LowPower WheelSubsystemFailure', 
        {(T, T): 0.95, (T, F): 0.8, (F, T): 0.5, (F, F): 0.05}),
    ('WeakCommsSignal', 'LowPower', 
        {(T): 0.8, (F): 0.3}),
    ('MovementDrift', 'WheelSubsystemFailure', 
        {(T): 0.8, (F): 0.1})
])

# Part A
print("The probability of solar panels covered in dust given slow movement, enumeration method:",enumeration_ask('SolarPanelDust', {'SlowMovement': T}, rover_faults).show_approx())
print("The probability of solar panels covered in dust given slow movement, elimination method:",elimination_ask('SolarPanelDust', {'SlowMovement': T}, rover_faults).show_approx())

# Part B
print("The probability of a battery system fault given weak comms, enumeration method:",enumeration_ask('BatterySystemFault', {'WeakCommsSignal': T}, rover_faults).show_approx())
print("The probability of a battery system fault given weak comms, elimination method:",elimination_ask('BatterySystemFault', {'WeakCommsSignal': T}, rover_faults).show_approx())

# Part C
print("The probability of rock stuck in wheel given slow movement and low battery power, enumeration method:",enumeration_ask('WheelSubsystemFailure', {'MovementDrift': T, 'LowPower': T }, rover_faults).show_approx())
print("The probability of rock stuck in wheel given slow movement and low battery power, elimination method:",elimination_ask('WheelSubsystemFailure', {'MovementDrift': T, 'LowPower': T }, rover_faults).show_approx())



##### FIGURE OUT HOW TO ANSWER PART C AND YOU FIX YOUR OTHER ISSUE
