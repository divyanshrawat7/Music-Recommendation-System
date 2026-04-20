from pyspark.ml.recommendation import ALS

def train_als_model(spark_df):
    als = ALS(
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=10,
        maxIter=10,
        regParam=0.1
    )
    model = als.fit(spark_df)
    return model

def recommend_als(model, user_id, spark_df, top_k=5):
    users_df = spark_df.select("user_id").distinct()
    user_subset = users_df.filter(users_df.user_id == user_id)
    recommendations = model.recommendForUserSubset(user_subset, top_k)
    recs = recommendations.collect()
    if len(recs) == 0:
        return []
    items = recs[0]["recommendations"]
    return [row["item_id"] for row in items]
