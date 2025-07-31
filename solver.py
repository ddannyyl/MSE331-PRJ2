import pulp
import pandas as pd

# Define sets
plants = ['Waterloo', 'Kingston']
DCs = ['DC1', 'DC2', 'DC3', 'DC4', 'DC5', 'DC6']
stores = ['Store' + str(i+1) for i in range(10)]

# Fixed setup costs for DCs (in CAD)
fixed_costs = {
    'DC1': 16000000,  # Hamilton
    'DC2': 11000000,  # Thunder Bay
    'DC3': 6000000,   # Forestville
    'DC4': 4500000,   # Cumberland House
    'DC5': 7000000,   # Tumbler Ridge
    'DC6': 4000000    # Indian Cabins
}

# Store demands
demands = {
    'Store1': 3000000,     # Mississauga
    'Store2': 2200000,     # Calgary
    'Store3': 1000000,     # Winnipeg
    'Store4': 940000,      # Quebec City
    'Store5': 600000,      # Halifax
    'Store6': 460000,      # Kamloops
    'Store7': 13000,       # La Ronge
    'Store8': 13000,       # Happy Valley–Goose Bay
    'Store9': 7000,        # Fort Smith
    'Store10': 1500        # Watson Lake
}

# Rail distances from plants to DCs (in km)
rail_distances = {
    ('Waterloo', 'DC1'): 70,       # Hamilton
    ('Waterloo', 'DC2'): 1100,     # Thunder Bay
    ('Waterloo', 'DC3'): 1400,     # Forestville
    ('Waterloo', 'DC4'): 1600,     # Cumberland House
    ('Waterloo', 'DC5'): 2000,     # Tumbler Ridge
    ('Waterloo', 'DC6'): 2800,     # Indian Cabins
    ('Kingston', 'DC1'): 300,      # Hamilton
    ('Kingston', 'DC2'): 750,     # Thunder Bay
    ('Kingston', 'DC3'): 2000,     # Forestville
    ('Kingston', 'DC4'): 2200,     # Cumberland House
    ('Kingston', 'DC5'): 3400,     # Tumbler Ridge
    ('Kingston', 'DC6'): 2600      # Indian Cabins
}

# Define custom distances for each DC to stores (in km)
dc_store_distances = {
    # DC1 (Hamilton, ON) to Stores
    'DC1': [
        50,     # Mississauga (ON)
        3200,   # Calgary (AB)
        2000,   # Winnipeg (MB)
        900,    # Quebec City (QC)
        1900,   # Halifax (NS)
        4200,   # Kamloops (BC)
        3000,   # La Ronge (SK)
        2100,   # Happy Valley–Goose Bay (NL)
        3700,   # Fort Smith (NT)
        4100    # Watson Lake (YT)
    ],

    # DC2 (Thunder Bay, ON) to Stores
    'DC2': [
        1400,   # Mississauga
        2000,   # Calgary
        700,    # Winnipeg
        1800,   # Quebec City
        2800,   # Halifax
        2600,   # Kamloops
        1800,   # La Ronge
        3100,   # Happy Valley–Goose Bay
        3300,   # Fort Smith
        3600    # Watson Lake
    ],

    # DC3 (Forestville, QC) to Stores
    'DC3': [
        900,   # Mississauga
        3700,   # Calgary
        2400,   # Winnipeg
        600,    # Quebec City
        1200,   # Halifax
        4500,   # Kamloops
        3000,   # La Ronge
        2200,   # Happy Valley–Goose Bay
        4100,   # Fort Smith
        4900    # Watson Lake
    ],

    # DC4 (Cumberland House, SK) to Stores
    'DC4': [
        2500,   # Mississauga
        1200,   # Calgary
        800,    # Winnipeg
        2600,   # Quebec City
        3400,   # Halifax
        1700,   # Kamloops
        500,    # La Ronge
        3500,   # Happy Valley–Goose Bay
        2200,   # Fort Smith
        3000    # Watson Lake
    ],

    # DC5 (Tumbler Ridge, BC) to Stores
    'DC5': [
        4400,   # Mississauga
        1100,   # Calgary
        2300,   # Winnipeg
        4700,   # Quebec City
        5500,   # Halifax
        800,    # Kamloops
        2100,   # La Ronge
        5400,   # Happy Valley–Goose Bay
        2500,   # Fort Smith
        3300    # Watson Lake
    ],

    # DC6 (Indian Cabins, AB) to Stores
    'DC6': [
        3400,   # Mississauga
        500,    # Calgary
        1600,   # Winnipeg
        3700,   # Quebec City
        4500,   # Halifax
        1200,   # Kamloops
        1500,   # La Ronge
        4400,   # Happy Valley–Goose Bay
        1800,   # Fort Smith
        2600    # Watson Lake
    ]
}

# Unit costs (in CAD per km - unit)
rail_unit_cost = 0.046
truck_unit_cost = 0.81

# Total Truck cost
truck_cost = {}
for dc in dc_store_distances:
    store_distances = dc_store_distances[dc]
    for i in range(len(stores)):
        store = stores[i]
        dist = store_distances[i]
        truck_cost[(dc, store)] = dist * truck_unit_cost


# Calculate rail transport costs
rail_cost = {(p, d): dist * rail_unit_cost for (p, d),
             dist in rail_distances.items()}


# Create optimization problem
prob = pulp.LpProblem("Distribution_Network_Optimization", pulp.LpMinimize)

# Decision variables
y = pulp.LpVariable.dicts("OpenDC", DCs, cat='Binary')
x = pulp.LpVariable.dicts("AssignStore", ((i, j)
                          for i in DCs for j in stores), cat='Binary')
z = pulp.LpVariable.dicts("Ship", ((p, i)
                          for p in plants for i in DCs), lowBound=0, cat='Integer')

# Objective function: rail cost + truck cost + fixed setup cost
prob += (
    pulp.lpSum(rail_cost[(p, i)] * z[(p, i)] for p in plants for i in DCs) +
    pulp.lpSum(truck_cost[(i, j)] * demands[j] * x[(i, j)] for i in DCs for j in stores) +
    pulp.lpSum(fixed_costs[i] * y[i] for i in DCs)
)

# Constraints
prob += pulp.lpSum(y[i] for i in DCs) == 2  # Open exactly 2 DCs

for j in stores:
    prob += pulp.lpSum(x[(i, j)]
                       for i in DCs) == 1  # Each store assigned to one DC

M = len(stores)
for i in DCs:
    prob += pulp.lpSum(x[(i, j)] for j in stores) <= M * y[i]

for i in DCs:
    prob += pulp.lpSum(z[(p, i)]
                       for p in plants) == pulp.lpSum(demands[j] * x[(i, j)] for j in stores)


prob.solve()

open_dcs = [i for i in DCs if y[i].varValue > 0.5]
assignments = {(i, j): x[(
    i, j)].varValue for i in DCs for j in stores if x[(i, j)].varValue > 0.5}
shipments = {
    (p, i): z[(p, i)].varValue for p in plants for i in DCs if z[(p, i)].varValue > 0}


open_dcs_df = pd.DataFrame({'Open DCs': open_dcs})
assignments_df = pd.DataFrame(assignments.items(), columns=[
                              '(DC, Store)', 'Assigned'])
shipments_df = pd.DataFrame(shipments.items(), columns=[
                            '(Plant, DC)', 'Shipped Units'])

# Calculate DC-to-Store shipment quantities (in tonnes)
dc_to_store_shipments = {
    (i, j): demands[j]
    for i in DCs for j in stores
    if x[(i, j)].varValue > 0.5
}

# Convert to DataFrame
dc_store_df = pd.DataFrame(dc_to_store_shipments.items(), columns=[
    '(DC, Store)', 'Units Shipped'])

# Output
print("\nShipments from DC to Store:")
print(dc_store_df)

# Output
print("Open DCs:")
print(open_dcs_df)

print("\nAssignments:")
print(assignments_df)

print("\nShipments:")
print(shipments_df)

print("\nTotal Cost (Objective Function Value):")
print(prob.objective.value())
