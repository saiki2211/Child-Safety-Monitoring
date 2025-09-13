import random, time, math
import numpy as np
import skfuzzy as fuzz

# Define universe for danger probability
x = np.linspace(0, 1, 100)

# Membership functions
safe_mf = fuzz.trimf(x, [0.0, 0.0, 0.3])
caution_mf = fuzz.trimf(x, [0.2, 0.4, 0.6])
high_mf = fuzz.trimf(x, [0.5, 0.7, 0.85])
critical_mf = fuzz.trimf(x, [0.7, 1.0, 1.0])

def fuzzy_action(prob_danger):
    membership = {
        "Safe": fuzz.interp_membership(x, safe_mf, prob_danger),
        "Caution": fuzz.interp_membership(x, caution_mf, prob_danger),
        "High": fuzz.interp_membership(x, high_mf, prob_danger),
        "Critical": fuzz.interp_membership(x, critical_mf, prob_danger),
    }
    decision = max(membership, key=membership.get)
    return decision, membership

# -------------------------------
# Tuned Risk Weights (more human-like)
# -------------------------------
risk_weights = {
    "Activity": {"Calm": 0.0, "Running": 0.3, "Jumping": 0.6},
    "Proximity": {"Safe": 0.0, "NearHazard": 0.4},
    "Environment": {"Normal": 0.0, "Slippery": 0.2},   # was 0.3
    "Age": {"Teen": 0.05, "Young": 0.2},              # Teen softer
    "Weather": {"Sunny": 0.0, "Rainy": 0.1},          # was 0.2
    "Supervision": {"Yes": 0.0, "No": 0.2},           # was 0.4
}

# Softer sigmoid curve (k=1.5 instead of 2.0)
def sigmoid(x, k=1.5):
    return 1 / (1 + math.exp(-k * x))

def compute_danger_probability(evidence):
    risk_score = sum(risk_weights[var][evidence[var]] for var in evidence)
    return sigmoid(risk_score)

# -------------------------------
# Variables
# -------------------------------
activities = ["Calm", "Running", "Jumping"]
proximities = ["Safe", "NearHazard"]
environments = ["Normal", "Slippery"]
ages = ["Teen", "Young"]
weathers = ["Sunny", "Rainy"]
supervisions = ["Yes", "No"]

# Example Scenarios
scenarios = [
    {"Activity": "Jumping", "Proximity": "NearHazard", "Environment": "Slippery",
     "Age": "Young", "Weather": "Rainy", "Supervision": "No"},
    {"Activity": "Running", "Proximity": "Safe", "Environment": "Normal",
     "Age": "Teen", "Weather": "Sunny", "Supervision": "Yes"},
    {"Activity": "Calm", "Proximity": "Safe", "Environment": "Slippery",
     "Age": "Young", "Weather": "Sunny", "Supervision": "Yes"},
]

# -------------------------------
# Input Normalization Helper
# -------------------------------
def normalize_input(prompt, options):
    val = input(prompt).strip().lower()
    mapping = {opt.lower(): opt for opt in options}  # map lowercase → correct form
    while val not in mapping:
        print(f"Invalid input! Choose from: {options}")
        val = input(prompt).strip().lower()
    return mapping[val]

# -------------------------------
# Input Mode Logic
# -------------------------------
def get_evidence(step, input_mode):
    if input_mode == "random":
        return {
            "Activity": random.choice(activities),
            "Proximity": random.choice(proximities),
            "Environment": random.choice(environments),
            "Age": random.choice(ages),
            "Weather": random.choice(weathers),
            "Supervision": random.choice(supervisions),
        }
    elif input_mode == "scenario":
        return scenarios[step % len(scenarios)]
    elif input_mode == "manual":
        return {
            "Activity": normalize_input("Enter Activity (Calm/Running/Jumping): ", activities),
            "Proximity": normalize_input("Enter Proximity (Safe/NearHazard): ", proximities),
            "Environment": normalize_input("Enter Environment (Normal/Slippery): ", environments),
            "Age": normalize_input("Enter Age (Teen/Young): ", ages),
            "Weather": normalize_input("Enter Weather (Sunny/Rainy): ", weathers),
            "Supervision": normalize_input("Enter Supervision (Yes/No): ", supervisions),
        }
    else:
        raise ValueError("Invalid input mode!")

# -------------------------------
# Mode Selection
# -------------------------------
mode = input("\nEnter mode (graph/gui): ").strip().lower()
input_mode = input("Choose input mode (random/scenario/manual): ").strip().lower()

# -------------------------------
# Graph Mode
# -------------------------------
if mode == "graph":
    import matplotlib.pyplot as plt
    probabilities, steps = [], []

    print("\n--- Starting Safety Monitoring (Graph Mode) ---\n")

    for i in range(10):
        evidence = get_evidence(i, input_mode)
        prob_danger = compute_danger_probability(evidence)
        decision, _ = fuzzy_action(prob_danger)
        status = f"Decision: {decision}"

        print(f"Step {i+1}: {evidence} => P(Danger)={prob_danger:.2f} → {status}")
        probabilities.append(prob_danger)
        steps.append(i+1)
        time.sleep(0.5)

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(steps, probabilities, marker='o', color='red', label='Danger Probability')
    plt.axhline(y=0.7, color='blue', linestyle='--', label='Alarm Threshold (0.7)')
    plt.title("Danger Probability Over Time (6 Variables)")
    plt.xlabel("Time Step")
    plt.ylabel("P(Danger = Yes)")
    plt.ylim(0,1)
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# GUI Mode
# -------------------------------
elif mode == "gui":
    import tkinter as tk
    from threading import Thread

    root = tk.Tk()
    root.title("Cognitive Robot Safety Monitor")
    root.geometry("600x320")

    status_label = tk.Label(root, text="Starting...", font=("Arial", 20), pady=20)
    status_label.pack()

    info_label = tk.Label(root, text="", font=("Arial", 12))
    info_label.pack()

    prob_label = tk.Label(root, text="", font=("Arial", 14))
    prob_label.pack()

    def monitoring_loop():
        step = 1
        while True:
            evidence = get_evidence(step, input_mode)
            prob_danger = compute_danger_probability(evidence)

            decision, _ = fuzzy_action(prob_danger)
            if decision == "Safe":
                status, color = "✅ SAFE", "green"
            elif decision == "Caution":
                status, color = "⚠️ CAUTION", "orange"
            elif decision == "High":
                status, color = "🚨 HIGH RISK", "red"
            else:  # Critical
                status, color = "🛑 CRITICAL", "darkred"


            status_label.config(text=status, fg=color)
            info_label.config(text=f"Step {step}: {evidence}")
            prob_label.config(text=f"P(Danger)={prob_danger:.2f}")

            step += 1
            time.sleep(1)

    Thread(target=monitoring_loop, daemon=True).start()
    root.mainloop()

else:
    print("Invalid mode! Please run again and choose either 'graph' or 'gui'.")

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Define universe (danger probability range 0–1)
x = np.linspace(0, 1, 100)

# Membership functions
safe = fuzz.trimf(x, [0.0, 0.0, 0.3])
caution = fuzz.trimf(x, [0.2, 0.4, 0.6])
high = fuzz.trimf(x, [0.5, 0.7, 0.85])
critical = fuzz.trimf(x, [0.7, 1.0, 1.0])

# Plot fuzzy sets
plt.figure(figsize=(10,6))
plt.plot(x, safe, 'g', linewidth=2, label='Safe')
plt.plot(x, caution, 'orange', linewidth=2, label='Caution')
plt.plot(x, high, 'r', linewidth=2, label='High Risk')
plt.plot(x, critical, 'darkred', linewidth=2, label='Critical')

plt.title("Fuzzy Membership Functions for Danger Probability")
plt.xlabel("Danger Probability (0 = Safe, 1 = Certain Danger)")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid(True)
plt.show()
