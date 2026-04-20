import streamlit as st
import pandas as pd
import numpy as np
from als_model import train_als_model, recommend_als
from spark_processing import get_spark_dataframe


from data_preprocessing import load_data, create_user_item_matrix
from models import compute_item_similarity, recommend_items, train_svd_model, recommend_svd

#PAGE CONFIGURATION

st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎧",
    layout="wide"
)

#Recommender Theme

st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212;
        color: white;
    }

    h1, h2, h3, h4 {
        color: #1DB954;
    }

    section[data-testid="stSidebar"] {
        background-color: #000000;
    }

    div.stButton > button {
        background-color: #1DB954;
        color: black;
        font-weight: bold;
        border-radius: 20px;
        height: 3em;
        width: 100%;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #17a74a;
        color: black;
    }

    .spotify-card {
        background-color: #181818;
        padding: 18px;
        border-radius: 15px;
        margin-bottom: 12px;
        transition: 0.3s;
    }

    .spotify-card:hover {
        background-color: #282828;
        transform: scale(1.02);
    }

    .metric-box {
        background-color: #181818;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_and_prepare_data():
    print("Loading data started")

    ratings = load_data("ratings.csv")

    #taken small sample for training of 10k only
    ratings = ratings.sample(10000, random_state=42)

    movies = pd.read_csv("movies.csv")
    movies = movies.rename(columns={"movieId": "item_id"})

    spark_df = get_spark_dataframe("ratings.csv")

    return ratings, movies, spark_df

data, movies, spark_df = load_and_prepare_data()

#SIDEBAR

st.sidebar.title("🎛 Control Panel")

user_ids = sorted(data['user_id'].unique())
selected_user = st.sidebar.selectbox("Select User ID", user_ids)

top_k = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Stats")
st.sidebar.write("Total Ratings:", len(data))
st.sidebar.write("Users:", data['user_id'].nunique())
st.sidebar.write("Items:", data['item_id'].nunique())

#HEADER
st.title("🎧 Music Recommendation System")
st.markdown("Discover personalized recommendations using advanced ML algorithms.")

#MODEL PREPARATION

@st.cache_resource
def prepare_cosine_model(data):
    user_item_matrix = create_user_item_matrix(data)
    similarity_matrix, item_matrix = compute_item_similarity(user_item_matrix)
    return user_item_matrix, similarity_matrix, item_matrix


@st.cache_resource
def prepare_svd_model(data):
    model, rmse = train_svd_model(data)
    return model, rmse


#RECOMMENDATIONS

if st.button("🎵 Get Recommendations"):

    col1, col2 = st.columns([3, 1])

    with col1:

        with st.spinner("Generating recommendations..."):
            als_model = train_als_model(spark_df)

            recommendations = recommend_als(
                als_model,
                selected_user,
                spark_df,
                top_k
            )

        recommendations = recommend_als(
            als_model,
            selected_user,
            spark_df,
            top_k
        )

        st.subheader("🎼 Recommended For You")

        for item in recommendations:
            title = movies[movies['item_id'] == item]['title'].values
            if len(title) > 0:
                st.markdown(f"""
                <div class="spotify-card">
                    🎵 <b>{title[0]}</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.write(f"🎵 Item ID: {item}")

    with col2:

        st.subheader("📈 Model Performance")

        st.markdown("""
        <div class="metric-box">
            <h3>ALS (Spark)</h3>
            <p>Distributed Recommendation Model</p>
        </div>
        """, unsafe_allow_html=True)