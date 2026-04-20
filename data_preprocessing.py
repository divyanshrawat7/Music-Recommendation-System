# data_preprocessing.py
from spark_processing import create_spark_session, load_data_spark, preprocess_data_spark, convert_to_pandas
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path, use_spark=True):
    if use_spark:
        spark = create_spark_session()

        df_spark = load_data_spark(spark, path)
        df_spark = preprocess_data_spark(df_spark)

        data = convert_to_pandas(df_spark)

    else:
        import pandas as pd
        data = pd.read_csv(path)

    return data

def preprocess_data(data):
    data = data.rename(columns={
        'userId': 'user_id',
        'movieId': 'item_id'
    })

    data = data[['user_id', 'item_id', 'rating']]
    return data


def create_user_item_matrix(data):
    user_item_matrix = data.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating'
    )
    return user_item_matrix


def train_test_split_data(data):
    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )
    return train, test
