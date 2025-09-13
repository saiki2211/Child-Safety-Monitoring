from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the Bayesian Network structure
model = DiscreteBayesianNetwork([
    ('Activity', 'Danger'),
    ('Proximity', 'Danger'),
    ('Environment', 'Danger')
])

# Step 2: Define Conditional Probability Tables (CPTs)

# Prior: Activity
cpd_activity = TabularCPD(
    variable='Activity',
    variable_card=3,
    values=[[0.5], [0.3], [0.2]],  # Calm=50%, Running=30%, Jumping=20%
    state_names={'Activity': ['Calm', 'Running', 'Jumping']}
)

# Prior: Proximity
cpd_proximity = TabularCPD(
    variable='Proximity',
    variable_card=2,
    values=[[0.7], [0.3]],  # Safe=70%, NearHazard=30%
    state_names={'Proximity': ['Safe', 'NearHazard']}
)

# Prior: Environment
cpd_environment = TabularCPD(
    variable='Environment',
    variable_card=2,
    values=[[0.8], [0.2]],  # Normal=80%, Slippery=20%
    state_names={'Environment': ['Normal', 'Slippery']}
)

# CPT for Danger
cpd_danger = TabularCPD(
    variable='Danger',
    variable_card=2,
    values=[
        # Danger=No
        [0.99, 0.90, 0.85, 0.70, 0.95, 0.60, 0.40, 0.05, 0.10, 0.01, 0.20, 0.05],
        # Danger=Yes
        [0.01, 0.10, 0.15, 0.30, 0.05, 0.40, 0.60, 0.95, 0.90, 0.99, 0.80, 0.95]
    ],
    evidence=['Activity', 'Proximity', 'Environment'],
    evidence_card=[3, 2, 2],
    state_names={
        'Danger': ['No', 'Yes'],
        'Activity': ['Calm', 'Running', 'Jumping'],
        'Proximity': ['Safe', 'NearHazard'],
        'Environment': ['Normal', 'Slippery']
    }
)

# Step 3: Add CPDs to the model
model.add_cpds(cpd_activity, cpd_proximity, cpd_environment, cpd_danger)

# Step 4: Check model validity
print("Model is valid:", model.check_model())

# Step 5: Inference
infer = VariableElimination(model)

# Example 1: Child is Jumping near Hazard on Slippery surface
query = infer.query(variables=['Danger'], evidence={'Activity': 'Jumping', 'Proximity': 'NearHazard', 'Environment': 'Slippery'})
print("\nCase 1: Jumping + NearHazard + Slippery")
print(query)

# Example 2: Child is Calm and Safe in Normal environment
query = infer.query(variables=['Danger'], evidence={'Activity': 'Calm', 'Proximity': 'Safe', 'Environment': 'Normal'})
print("\nCase 2: Calm + Safe + Normal")
print(query)
