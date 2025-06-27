import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib  # To save the model

def load_data(file_path):
    """Load and preprocess data."""
    data = pd.read_csv(file_path)
    
    # Feature Engineering (if needed)
    data['recency'] = (pd.to_datetime('today') - pd.to_datetime(data['last_purchase_date'])).dt.days
    data['frequency'] = data.groupby('customer_id')['transaction_id'].transform('count')
    data['monetary'] = data.groupby('customer_id')['transaction_amount'].transform('sum')
    
    features = data[['recency', 'frequency', 'monetary', 'avg_basket_size']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return data, scaled_features, scaler

def train_model(scaled_features, n_clusters=4):
    """Train K-Means model."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)
    return kmeans

def save_model(model, scaler, filepath='model/'):
    """Save model and scaler."""
    joblib.dump(model, filepath + 'kmeans_model.pkl')
    joblib.dump(scaler, filepath + 'scaler.pkl')

if __name__ == "__main__":
    # Example usage
    data, scaled_features, scaler = load_data("data/customer_data.csv")
    model = train_model(scaled_features)
    save_model(model, scaler)