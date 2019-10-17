#!/usr/bin/python
from gurobipy import *
import numpy as np

distances = {(1,1):0,
    (1,2): 1,
    (1,3):2,
    (1,4):3,
    (1,5):4,
    (1,6):3,
    (1,7):3,
    (1,8):2,
    (1,9):1,
    (1,10):1,
    (1,11):2,
    (1,12):2,
    (1,13):2,
    (1,14):3,
    (1,15):3,
    (2,2):0,
    (2,3):1,
    (2,4):2,
    (2,5):3,
    (2,6):3,
    (2,7):3,
    (2,8):2,
    (2,9):2,
    (2,10):1,
    (2,11):2,
    (2,12):1,
    (2,13):2,
    (2,14):2,
    (2,15):2,
    (3,3):0,
    (3,4):1,
    (3,5):2,
    (3,6):2,
    (3,7):3,
    (3,8):3,
    (3,9):3,
    (3,10):2,
    (3,11):1,
    (3,12):2,
    (3,13):2,
    (3,14):1,
    (3,15):1,
    (4,4):0,
    (4,5):1,
    (4,6):2,
    (4,7):3,
    (4,8):3,
    (4,9):3,
    (4,10):2,
    (4,11):1,
    (4,12):1,
    (4,13):2,
    (4,14):1,
    (4,15):2,
    (5,5):0,
    (5,6):1,
    (5,7):2,
    (5,8):3,
    (5,9):4,
    (5,10):3,
    (5,11):2,
    (5,12):2,
    (5,13):2,
    (5,14):1,
    (5,15):2,
    (6,6):0,
    (6,7):1,
    (6,8):2,
    (6,9):3,
    (6,10):2,
    (6,11):1,
    (6,12):2,
    (6,13):1,
    (6,14):2,
    (6,15):3,
    (7,7):0,
    (7,8):1,
    (7,9):2,
    (7,10):2,
    (7,11):2,
    (7,12):3,
    (7,13):1,
    (7,14):3,
    (7,15):4,
    (8,8):0,
    (8,9):1,
    (8,10):1,
    (8,11):2,
    (8,12):2,
    (8,13):1,
    (8,14):4,
    (8,15):4,
    (9,9):0,
    (9,10):1,
    (9,11):2,
    (9,12):2,
    (9,13):2,
    (9,14):4,
    (9,15):4,
    (10,10):0,
    (10,11):1,
    (10,12):1,
    (10,13):1,
    (10,14):3,
    (10,15):3,
    (11,11):0,
    (11,12):1,
    (11,13):1,
    (11,14):2,
    (11,15):2,
    (12,12):0,
    (12,13):2,
    (12,14):2,
    (12,15):3,
    (13,13):0,
    (13,14):3,
    (13,15):3,
    (14,14):0,
    (14,15):1,
    (15,15):0}

for i in range(15):
    for j in range(i+1,15):
        distances[(j+1,i+1)] = distances[(i+1,j+1)]

SUPPLIES = ['Medkit', 'Food', 'Water', 'Fuel']
RESPONDER_TYPES = ['EMT','FF','POL','CW']
CARRYING_CAP = {'EMT':{'Medkit':10,'Food':1,'Water':2,'Fuel':3},
    'FF':{'Medkit':6,'Food':3,'Water':4,'Fuel':4},
    'POL':{'Medkit':2,'Food':1,'Water':2,'Fuel':5},
    'CW':{'Medkit':0,'Food':2,'Water':3,'Fuel':5}}
POST_PROCESSES = ['S0','S1','S2']
PRE_PROCESSES = ['S0','S1','S2']
POST_TIMES = {'S0':0,'S1':1,'S2':1}
POST_LOCATION = {'S1':13,'S2':14}
F = 10000
POST_PRE_DISTANCES = {}

def demo():
    n = 23
    m = 8
    parameters= {}
    parameters['locations']=[2,2,2,3,4,4,9,9,9,4,4,4,4,8,1,4,4,4,4,5,15,15,15]
    medkits = [4,4,4,4,5,5,5,1,0,0,1,0,0,1,1,4,4,2,1,1,0,0,0]
    food = [0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1]
    fuel = [0,0,0,0,3,3,3,3,3,3,3,3,3,1,1,2,2,2,2,2,1,2,2]
    water = [0,0,0,0,2,2,2,2,2,2,2,2,2,0,0,2,2,4,4,4,1,1,1]
    parameters['supply_cost'] = {}
    for id in range(n):
        parameters['supply_cost'][id] = {'Medkit':medkits[id],'Food':food[id],'Water':water[id], 'Fuel':fuel[id]}
    parameters['deadlines'] = [7,25,25,25,10,10,10,F,F,F,F,F,F,20,20,5,12,12,12,15,50,50,50]
    s0 = [0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1]
    parameters['post_process'] = {}
    for id in range(n):
        if s0[id] ==1:
            parameters['post_process'][id] = ['S0','S1','S2']
        else:
            parameters['post_process'][id] = ['S1','S2']
    emt = [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0]
    ff = [0,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0]
    pol = [0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
    cw = [0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1]
    parameters['feasible_responders'] = {}
    parameters['process_times'] = {}
    for id in range(n):
        parameters['feasible_responders'][id] = {'EMT':emt[id],'FF':ff[id],'POL':pol[id],'CW':cw[id]}
        parameters['process_times'][id] = {'EMT':1,'FF':1,'POL':1,'CW':1}


    start_locations = [13,14,6,12,4,7,10,10]
    type = ['EMT','EMT','FF','FF','POL','POL','CW','CW']
    parameters['start_locations'] = {}
    parameters['responder_type'] = {}
    for id in range(m):
        parameters['start_locations'][id] = start_locations[id]
        parameters['responder_type'][id] = type[id]

    print(distances[(2,6)])
    return parameters


def disaster_solver(parameters,n,m):
    task_ids = []
    for i in range(n):
        task_ids.append(i)
    locations = {}
    supply_cost = {}
    deadlines = {}
    feasible_responders = {}
    process_times = {}
    post_process0 = {}
    post_times = {}
    pre_times = {}
    for id in task_ids:
        locations[id] = parameters['locations'][id]
        supply_cost[id] = parameters['supply_cost'][id]
        deadlines[id] = parameters['deadlines'][id]
        post_process0[id] = parameters['post_process'][id]
        for pp in POST_PROCESSES:
            if pp in post_process0[id]:
                if pp != 'S0':
                    post_times[(id,pp)] = distances[(locations[id],POST_LOCATION[pp])]+ POST_TIMES[pp]
                else:
                    post_times[(id,'S0')] = 0
            else:
                post_times[(id,pp)] = F/2
        for pp in PRE_PROCESSES:
            pre_times[(id,pp)] = 0
        feasible_responders[id] = parameters['feasible_responders'][id]
        process_times[id] = parameters['process_times'][id]


    responder_ids = []
    for k in range(m):
        responder_ids.append(k)
    start_locations = {}
    responder_type = {}
    carry_cap ={}
    for id in responder_ids:
        start_locations[id] = parameters['start_locations'][id]
        responder_type[id] = parameters['responder_type'][id]
        carry_cap[id] = CARRYING_CAP[responder_type[id]]

    model = Model('Response')

    # X_ijk in model
    process_procession = model.addVars(task_ids,task_ids,responder_ids, vtype=GRB.BINARY)
    # Y_ik in model
    completed = model.addVars(task_ids,responder_ids,vtype=GRB.BINARY)
    # R_ijk
    pre_task_inventory = model.addVars(task_ids,SUPPLIES,responder_ids,lb=0.0)
    # r^g_ij
    restocked_inventory = model.addVars(task_ids,SUPPLIES,PRE_PROCESSES)
    # B_i
    task_start = model.addVars(task_ids, lb=0.0)
    #w_ig
    post_process = model.addVars(task_ids,POST_PROCESSES,vtype=GRB.BINARY)
    #w_i^g
    pre_process = model.addVars(task_ids,PRE_PROCESSES,vtype=GRB.BINARY)
    # New var to measure that each responder is utilized
    utilized = model.addVars(responder_ids, vtype=GRB.BINARY)

    # Artificial Objective variable for simplicity
    Z = model.addVar()

    # Constraint set (0.1), check whether responder is utilized
    model.addConstrs(
        (utilized[k] >= 1/F * quicksum(completed[i,k] for i in task_ids) for k in responder_ids), 'Utilized_1')

    model.addConstrs(
        (utilized[k] <= quicksum(completed[i,k] for i in task_ids) for k in responder_ids), 'Utilized_2')

    # Constraint set (0.3), enforce x_ijk
    model.addConstrs((
        quicksum(process_procession[i,j,k] for i in task_ids for j in task_ids) ==
        quicksum(completed[i,k] for i in task_ids) - utilized[k] for k in responder_ids), 'Utilized_3')

    # Constraint set (1), make sure Z is time last task is post-processed
    model.addConstrs(
        (task_start[i] +
        quicksum(completed[i,k]*process_times[i][responder_type[k]] for k in responder_ids)
        + quicksum(post_process[i,g] * post_times[i,g] for g in POST_PROCESSES) <= Z for i in task_ids),
        'Makespan_Completed')

    # Constraint set (2), make sure all tasks are completed exactly once
    model.addConstrs(
        (completed.sum(i,'*') == 1 for i in task_ids), 'All_Completed')

    # Constraint set (3), make sure task completed by qualified worker
    model.addConstrs(
        (completed[i,k] <= feasible_responders[i][responder_type[k]] for i in task_ids for k in responder_ids),
        'Qualified_Responders')

    # Constraint set (4), make sure a post process option is taken
    model.addConstrs(
        (post_process.sum(i,'*') == 1 for i in task_ids), 'All_PostProcessed'
    )

    # Constraint set (5), make sure a a pre process option is taken
    model.addConstrs(
        (pre_process.sum(i,'*') == 1 for i in task_ids), 'All_PreProcessed'
    )

    # Constraint set (6), enforce carrying capacity
    model.addConstrs(
        (pre_task_inventory[i,j,k] <= carry_cap[k][j] for i in task_ids for j in SUPPLIES for k in responder_ids),
        'Carry_Capacity')

    # Constraint set (7), Task completed by deadline
    model.addConstrs(
        (task_start[i] + quicksum(completed[i,k] * process_times[i][responder_type[k]] for k in responder_ids) <= deadlines[i] for i in task_ids),
        'Enforce_Deadline')

    # Constraint set (8), Tasks are not preceeded by others unless completed by the same worker
    model.addConstrs(
        (process_procession.sum('*',j,k) <= completed[j,k] for j in task_ids for k in responder_ids),
        'Enforce_Procession_1')

    # Constraint set (9), other half of set (8)
    model.addConstrs(
        (process_procession.sum(i,'*',k) <= completed[i,k] for i in task_ids for k in responder_ids),
        'Enforce_Procession_2')

    # Constraint set (10), a task cannot preceed itself
    model.addConstrs(
        (process_procession[i,i,k] == 0 for i in task_ids for k in responder_ids),
        'No_SelfProcession')

    # Constraint set (11), a task cannot begin until its preceeding task is finished along with all post and pre-processing
    # Slightly modified, as we are assuming zero transition time between pre- and post-processes
    model.addConstrs(
        ((F * (3 - process_procession[i,j,k] - pre_process[j,h] - post_process[i,g]) + task_start[j] - task_start[i] >= process_times[i][responder_type[k]]
        + post_times[i,g] + pre_times[j,h] + distances[(locations[i],locations[j])]) for i in task_ids for j in task_ids for k in responder_ids for g in POST_PROCESSES for h in PRE_PROCESSES),
        'No_Overlap_1')

    # Constraint set (11.5), transition time between tasks is at least the distance between them plus processing time
    model.addConstrs(
        (F*(3 - process_procession[i,j,k] - pre_process[j,h] - post_process[i,g]) + task_start[j] - task_start[i]
        >= process_times[i][responder_type[k]] + distances[(locations[i],locations[j])]
        for i in task_ids for j in task_ids for k in responder_ids for h in PRE_PROCESSES for g in POST_PROCESSES),
        'No_Overlap_2')

    # Constraint set (12), ensure a task is not started unless the worker has sufficient supplies, given their previous task
    model.addConstrs(
        (F*(1-process_procession[i,l,k]) + pre_task_inventory[i,j,k] - supply_cost[i][j] + restocked_inventory.sum(l,j,'*')
        >= pre_task_inventory[l,j,k] for i in task_ids for l in task_ids for j in SUPPLIES for k in responder_ids),
        'Sufficient_Supplies_1')

    # Constraint set (13) ensure that responders don't resupply with more than is possible for post-process
    # Slightly modified, as only S0 has resupply limits
    model.addConstrs(
        (restocked_inventory[i,j,'S0'] == 0 for i in task_ids for j in SUPPLIES),
        'No_S0_restock'
    )

    # Constraint set (14) ensure that pre-process supplies are only picked up if pre-process is completed
    # Slightly modified to ensure that 0 does not come up
    model.addConstrs(
        (restocked_inventory[i,j,g]/F <= pre_process[i,g] for i in task_ids for j in SUPPLIES for g in PRE_PROCESSES),
        'No_illegal_restock')

    # Constraint set (15) Ensure sufficient supplies before starting a task.
    model.addConstrs(
        (F*(1-completed[i,k]) + pre_task_inventory[i,j,k] >= supply_cost[i][j] for i in task_ids for j in SUPPLIES),
        'Sufficient_Supplies_2')

    # Constraint set (16) Ensure correct start time given initial locations. Slightly modified to ignore pre-process.
    model.addConstrs(
        (F*(1-completed[i,k]) + task_start[i] >= distances[(locations[i],start_locations[k])] for i in task_ids for k in responder_ids),
        'Initial_Locations')

    # Constraint set (21) Ensure a valid post-process is chosen
    model.addConstrs(
        (post_process[i,g] == 0 for i in task_ids for g in POST_PROCESSES if g not in post_process0[i]),
        'Feasible_Post')

    model.setObjective(Z, GRB.MINIMIZE)

    model.optimize()

    # Print solution
    if model.status == GRB.Status.OPTIMAL:
        comp = model.getAttr('x',completed)
        for i in task_ids:
            for k in responder_ids:
                if completed[i,k].x>0:
                    print((i,k,comp[i,k]))
        preceeds = model.getAttr('x',process_procession)
        for i in task_ids:
            for j in task_ids:
                for k in responder_ids:
                    if process_procession[i,j,k].x>0:
                        print((i,j,k, preceeds[i,j,k]))

    return model

def main():
    n = 23
    m = 8
    parameters = demo()
    model = disaster_solver(parameters,n,m)

if __name__ == "__main__":
    main()
