import pandas as pd
import numpy as np

# Generate perfect RFM data
data = pd.DataFrame({
    'customer_id': [f"CUST_{i:04d}" for i in range(1, 501)],
    'recency': np.random.randint(1, 365, 500),  # Days since last purchase
    'frequency': np.random.poisson(5, 500),     # Number of orders
    'monetary': np.round(np.random.uniform(10, 2000, 500), 2),  # Total spend
    'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], 500),
    'segment': np.random.choice(['Premium', 'Standard', 'New'], 500)
})

# Save to CSV
data.to_csv('rfm_ready_to_use.csv', index=False)
print("Dataset saved as 'rfm_ready_to_use.csv'")