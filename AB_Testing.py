import numpy as np
from scipy import stats

class ABTest:
    def __init__(self):
        self.data = {}

    def add_data(self, group, conversions, total):
        """
        Adds data for a specific group (control or experimental).

        Parameters:
        group (str): The name of the group ('Control' or 'Test').
        conversions (int): The number of conversions (clicks on CTA).
        total (int): The total number of users in the group.
        """
        self.data[group] = {'conversions': conversions, 'total': total}

    def conversion_rate(self, group):
        """
        Calculates the conversion rate for a specific group.

        Parameters:
        group (str): The name of the group ('Control' or 'Test').

        Returns:
        float: The conversion rate for the group.
        """
        group_data = self.data[group]
        return group_data['conversions'] / group_data['total']

    def z_test(self, group1, group2):
        """
        Performs a z-test between two groups to determine if the
        difference in conversion rates is statistically significant.

        Parameters:
        group1 (str): The name of the first group (e.g., 'Control').
        group2 (str): The name of the second group (e.g., 'Test').

        Returns:
        float: The p-value from the z-test, indicating significance.
        """
        # Conversion rates
        p1 = self.conversion_rate(group1)
        p2 = self.conversion_rate(group2)

        # Number of users
        n1 = self.data[group1]['total']
        n2 = self.data[group2]['total']

        # Pooled conversion rate
        p_pool = (self.data[group1]['conversions'] + self.data[group2]['conversions']) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        # Z-score
        z = (p1 - p2) / se

        # P-value (two-tailed test)
        p_value = stats.norm.sf(abs(z)) * 2
        return p_value

# Example Usage: Real-World A/B Test
ab_test = ABTest()

# Group A: Control group (red button)
ab_test.add_data('Control', conversions=240, total=1200)

# Group B: Test group (green button)
ab_test.add_data('Test', conversions=300, total=1250)

# Output results
print(f"Conversion rate for Control group: {ab_test.conversion_rate('Control') * 100:.2f}%")
print(f"Conversion rate for Test group: {ab_test.conversion_rate('Test') * 100:.2f}%")

p_value = ab_test.z_test('Control', 'Test')
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result is statistically significant! The green button leads to a higher conversion rate.")
else:
    print("Result is not statistically significant. No evidence that the green button outperforms the red one.")
