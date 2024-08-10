import numpy as np
import pandas as pd
from faker import Faker
from dateutil.relativedelta import relativedelta
from datetime import datetime

# Initialize Faker and set seed for reproducibility
fake = Faker()
np.random.seed(42)

# Define the number of synthetic records
num_records = 1000

# Generate synthetic names
names = [fake.name() for _ in range(num_records)]

# Generate synthetic ages (random values between 18 and 70)
ages = np.random.randint(18, 71, size=num_records)

# Generate synthetic addresses
addresses = [fake.address().replace("\n", ", ") for _ in range(num_records)]

# Generate synthetic purchase amounts (random float values between 10 and 1000)
purchase_amounts = np.random.uniform(10, 1000, size=num_records).round(2)

# Generate synthetic purchase dates (within the last 2 years)
start_date = datetime.today() - relativedelta(years=2)
end_date = datetime.today()
purchase_dates = [fake.date_between(start_date=start_date, end_date=end_date) for _ in range(num_records)]

# Create a DataFrame
data = {
    'Name': names,
    'Age': ages,
    'Address': addresses,
    'PurchaseAmount': purchase_amounts,
    'PurchaseDate': purchase_dates
}

df = pd.DataFrame(data)

# Save to a CSV file
df.to_csv('synthetic_customer_data.csv', index=False)

print("Synthetic data generated successfully!")
