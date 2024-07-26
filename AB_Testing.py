import numpy as np 
from scipy import stats 

class ABTest:
    def __init__(self):
        self.data = {}

    def add_data(self, group, conversions, total):
        """
        Adds data for a specific group.

        Parameters:
        group (str): The name of the group (e.g., 'A' or 'B').
        conversions (int): The number of conversions for this group.
        total (int): The total number of users for this group.
        """
        self.data[group] = {'conversions': conversions, 'total': total}

    def conversion_rate(self, group):
        """
        Calculates the conversion rate for a specific group.

        Parameters:
        group (str): The name of the group (e.g., 'A' or 'B').

        Returns:
        float: The conversion rate for the group.
        """
        group_data = self.data[group]
        return group_data['conversions'] / group_data['total']

    def z_test(self, group1, group2):
        """
        Performs a z-test for two groups.

        Parameters:
        group1 (str): The name of the first group (e.g., 'A').
        group2 (str): The name of the second group (e.g., 'B').

        Returns:
        float: The p-value from the z-test.
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

        # P-value
        p_value = stats.norm.sf(abs(z)) * 2 # two-tailed test
        return p_value

# Example usage 
ab_test = ABTest()
ab_test.add_data('A', conversions=200, total=1000)
ab_test.add_data('B', conversions=250, total=1000)

print("Conversion rate for A:", ab_test.conversion_rate('A'))
print("Conversion rate for B:", ab_test.conversion_rate('B'))
print("P-value:", ab_test.z_test('A', 'B'))