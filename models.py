# models.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split
from surprise import accuracy


# ---------------- COSINE MODEL ----------------

def compute_item_similarity(user_item_matrix):
    matrix_filled = user_item_matrix.fillna(0)
    item_matrix = matrix_filled.T
    similarity_matrix = cosine_similarity(item_matrix)
    return similarity_matrix, item_matrix


def recommend_items(user_id, user_item_matrix, similarity_matrix, item_matrix, top_k=5):
    user_ratings = user_item_matrix.loc[user_id].fillna(0)
    scores = similarity_matrix.dot(user_ratings)

    rated_items = user_ratings[user_ratings > 0].index
    ranked_items = np.argsort(scores)[::-1]

    recommendations = []
    for idx in ranked_items:
        item_id = item_matrix.index[idx]
        if item_id not in rated_items:
            recommendations.append(int(item_id))
        if len(recommendations) >= top_k:
            break

    return recommendations


# ---------------- SVD MODEL ----------------

def train_svd_model(data):
    reader = Reader(rating_scale=(0.5, 5.0))

    surprise_data = Dataset.load_from_df(
        data[['user_id', 'item_id', 'rating']],
        reader
    )

    trainset, testset = surprise_split(
        surprise_data,
        test_size=0.2,
        random_state=42
    )

    model = SVD(n_factors=20, n_epochs=10)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)

    return model, rmse


def recommend_svd(user_id, data, model, top_k=5):
    all_items = data['item_id'].unique()
    rated_items = data[data['user_id'] == user_id]['item_id'].tolist()

    predictions = []

    for item in all_items:
        if item not in rated_items:
            pred = model.predict(user_id, item)
            predictions.append((item, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_items = [int(item[0]) for item in predictions[:top_k]]

    return top_items