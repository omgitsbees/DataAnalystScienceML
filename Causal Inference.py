import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

# 1. Generate synthetic data
np.random.seed(42)

# Number of sampels
n = 1000

# Treatment assignment (binary: 0 or 1)
treatment = np.random.binomial(1, 0.5, n)

# Confounder (affects both treatment and outcome)
confounder = np.random.normal(0, 1, n)

# Confounder (affects both treatment and outcome)
outcome = 2*treatment + 3*confounder + np.random.normal(0, 1, n)

# Create DataFrame
data = pd.DataFrame({'Treatment': treatment, 'Confounder': confounder, 'Outcome': outcome})

# 2.  Specify the casual model
model = CausalModel(
    data=data,
    treatment='Treatment',
    outcome='Outcome',
    common_causes=['Confounder'] # List of confounders
)

# Visualize the casual graph
model.view_model()

# 3. Identify the casual effect
identified_estimand = model.identify_effect()

# 4. Estimate the causal effect using the backdoor method
estimate = model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression')

# Print the estimate of the Average Treatment Effect (ATE)
print(f"Estimated ATE: {estimate.value}")

# 5. Refutation: Perform a placebo test to check robustness
refutation = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
print(refutation)