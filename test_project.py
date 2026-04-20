# test_project.py

from data_preprocessing import load_data, preprocess_data, create_user_item_matrix, train_test_split_data
from models import compute_item_similarity, recommend_items, train_svd_model, recommend_svd
from evaluation import precision_at_k, recall_at_k
import numpy as np

# Load data
data = load_data("data/ratings.csv")
data = preprocess_data(data)

print(data.head())
print("Dataset shape:", data.shape)

# Split data
train_data, test_data = train_test_split_data(data)

# Create user-item matrix
train_matrix = create_user_item_matrix(train_data)

# -------- COSINE MODEL --------
similarity_matrix, item_matrix = compute_item_similarity(train_matrix)

user_id = 1
cosine_recs = recommend_items(user_id, train_matrix, similarity_matrix, item_matrix, top_k=5)

print("Cosine Recommendations:", cosine_recs)

# -------- SVD MODEL --------
svd_model, rmse = train_svd_model(data)
print("SVD RMSE:", rmse)

svd_recs = recommend_svd(user_id, data, svd_model, top_k=5)
print("SVD Recommendations:", svd_recs)