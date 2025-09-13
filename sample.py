from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random, time
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define Bayesian Network
# -------------------------------
model = DiscreteBayesianNetwork([
    ('Activity', 'Danger'),
    ('Proximity', 'Danger'),
    ('Environment', 'Danger')
])

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

# Add CPDs to the model
model.add_cpds(cpd_activity, cpd_proximity, cpd_environment, cpd_danger)

# Validate model
print("Model is valid:", model.check_model())

# Inference object
infer = VariableElimination(model)

# -------------------------------
# 2. Quick Test Cases
# -------------------------------
query = infer.query(variables=['Danger'], evidence={'Activity': 'Jumping', 'Proximity': 'NearHazard', 'Environment': 'Slippery'})
print("\nCase 1: Jumping + NearHazard + Slippery")
print(query)

query = infer.query(variables=['Danger'], evidence={'Activity': 'Calm', 'Proximity': 'Safe', 'Environment': 'Normal'})
print("\nCase 2: Calm + Safe + Normal")
print(query)

# -------------------------------
# 3. Continuous Monitoring + Plot
# -------------------------------
activities = ['Calm', 'Running', 'Jumping']
proximities = ['Safe', 'NearHazard']
environments = ['Normal', 'Slippery']

probabilities = []
steps = []

print("\n--- Starting Child Safety Monitoring ---\n")

for i in range(15):  # simulate 15 steps
    activity = random.choice(activities)
    proximity = random.choice(proximities)
    environment = random.choice(environments)

    query = infer.query(
        variables=['Danger'],
        evidence={'Activity': activity, 'Proximity': proximity, 'Environment': environment}
    )
    prob_danger = query.values[1]  # probability of Danger=Yes

    # Decision
    status = "🚨 Alarm" if prob_danger > 0.7 else "✅ Safe"
    print(f"Step {i+1}: Activity={activity}, Proximity={proximity}, Environment={environment} "
          f"=> P(Danger)={prob_danger:.2f} → {status}")

    # Save for plot
    probabilities.append(prob_danger)
    steps.append(i+1)

    time.sleep(0.5)

# -------------------------------
# 4. Plot Results
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(steps, probabilities, marker='o', color='red', label='Danger Probability')
plt.axhline(y=0.7, color='blue', linestyle='--', label='Alarm Threshold (0.7)')
plt.title("Danger Probability Over Time")
plt.xlabel("Time Step")
plt.ylabel("P(Danger = Yes)")
plt.ylim(0,1)
plt.legend()
plt.grid(True)
plt.show()
