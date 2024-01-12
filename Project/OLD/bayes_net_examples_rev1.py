from project_utils import *

T, F = True, False

burglary = BayesNet([
    ('Burglary', '', 0.001),
    ('Earthquake', '', 0.002),
    ('Alarm', 'Burglary Earthquake',
     {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
])

print(enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
print(elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
# returns False: 0.716, True: 0.284
node_specs = []
node_specs.append(('GoodCoach', '', 1/50, [True,False]))
node_specs.append(('Lucky', '', 1/12, [True,False]))
node_specs.append(('Salary','GoodCoach Lucky', {(True,True): [0.7,0.2,0.1], (True,False): [0.9,0.1,0], (False,True): [0.1,0.5,0.4], (False,False): [0.5,0.4,0.1]}, ["High","Medium","Low"]))

catbn = BayesNetCategorical(node_specs)

print(enumeration_ask('GoodCoach', {"Lucky": False, "Salary": "High"}, catbn).show_approx())
print(elimination_ask('GoodCoach', {"Lucky": False, "Salary": "High"}, catbn).show_approx())
# returns False: 0.965, True: 0.0354
