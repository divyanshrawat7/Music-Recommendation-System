import os
os.environ["JAVA_HOME"] = "C:\\Users\\divya\\Downloads\\OpenJDK17U-jdk_x64_windows_hotspot_17.0.18_8\\jdk-17.0.18+8"

from pyspark.sql import SparkSession


def create_spark_session():
    spark = SparkSession.builder \
        .appName("Recommendation System") \
        .master("local[2]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    return spark


def load_data_spark(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    return df


def preprocess_data_spark(df):
    df = df.withColumnRenamed("userId", "user_id") \
           .withColumnRenamed("movieId", "item_id")

    df = df.select("user_id", "item_id", "rating")

    return df


def convert_to_pandas(df):
    return df.toPandas()

def get_spark_dataframe(path):
    spark = create_spark_session()
    df = load_data_spark(spark, path)
    df = preprocess_data_spark(df)
    return df
