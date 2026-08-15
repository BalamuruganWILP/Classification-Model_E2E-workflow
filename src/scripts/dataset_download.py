import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# 1. Fetch raw Cardiotocography dataset directly from UCI repo
print("Fetching raw data from UCI Machine Learning Repository...")
cardiotocography = fetch_ucirepo(id=193)

# 2. Extract features (X) and target targets (y) as Pandas DataFrames
X = cardiotocography.data.features
y = cardiotocography.data.targets

# 3. Combine them back briefly into a single raw master dataframe
df_raw = pd.concat([X, y], axis=1)

# 4. Perform an 80/20 random stratified split
# Stratify=y ensures both sets contain a balanced distribution of Normal/Suspect/Pathologic states
train_df, test_df = train_test_split(
    df_raw, 
    test_size=0.20, 
    random_state=42, 
    stratify=df_raw['NSP']
)

# 5. Export the 20% holdout test partition to the project root directory and save training partition to a CSV file
test_df.to_csv("test_data.csv", index=False)
train_df.to_csv("train_data.csv", index=False)
print(f"Successfully created test partition!")
print(f"Total instances saved to 'test_data.csv': {len(test_df)} records.")
print(f"Total instances saved to 'train_data.csv': {len(train_df)} records.")
