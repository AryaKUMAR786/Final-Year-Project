import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

def preprocess_data(df):
    df['Amount in USD'] = df['Amount in USD'].str.replace(',', '', regex=True)
    df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')
    df = df.dropna(subset=['Amount in USD'])
    
    categorical_cols = ['Industry Vertical', 'Sub Vertical', 'City  Location', 'Investment Type']
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df[categorical_cols]).toarray()
    
    scaler = StandardScaler()
    scaled_amount = scaler.fit_transform(df[['Amount in USD']])
    
    return df, encoded_features, encoder, scaler, scaled_amount

def train_model(df, encoded_features, scaled_amount):
    X = np.hstack((encoded_features, scaled_amount))
    
    model = NearestNeighbors(n_neighbors=15, metric='euclidean')
    model.fit(X)
    return model

def recommend_investors(model, encoder, scaler, df, query):
    query_df = pd.DataFrame([query])
    encoded_query = encoder.transform(query_df[['Industry Vertical', 'Sub Vertical', 'City  Location', 'Investment Type']]).toarray()
    scaled_query_amount = scaler.transform([[query['Amount in USD']]])
    query_vector = np.hstack((encoded_query, scaled_query_amount))
    
    distances, indices = model.kneighbors(query_vector, n_neighbors=15)
    
    recommendations = []
    for idx in indices[0]:
        if df.iloc[idx]['Amount in USD'] >= query['Amount in USD']:
            recommendations.append((df.iloc[idx]['Investors Name'], df.iloc[idx]['Amount in USD']))
        if len(recommendations) == 5:
            break
    
    recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by investment amount descending
    return [r[0] for r in recommendations]

# Load dataset
data = pd.read_csv("data.csv")
data, encoded_data, encoder, scaler, scaled_amount = preprocess_data(data)
model = train_model(data, encoded_data, scaled_amount)

# Example query
query_input = {
    "Industry Vertical": "Technology",
    "Sub Vertical": "Online Poker Platform",
    "City  Location": "Bengaluru",
    "Investment Type": "Private Equity",
    "Amount in USD": 1000000
}

recommended_investors = recommend_investors(model, encoder, scaler, data, query_input)
print("Top 5 recommended investors:", recommended_investors)
