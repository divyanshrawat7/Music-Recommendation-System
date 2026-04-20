# Distributed Recommendation System using Spark ALS

## About the Project

This project is a recommendation system built using machine learning and big data concepts.  
The goal is to suggest items (movies/music) to users based on their past interactions.

At first, we implemented basic recommendation techniques like **Cosine Similarity** and **SVD**.  
Later, we improved the system by integrating **Apache Spark MLlib’s ALS algorithm**, which allows the system to scale and work efficiently on larger datasets.

The final system uses:
- Spark for distributed processing  
- ALS as the main recommendation model  
- Streamlit for a simple and interactive UI  

---

## Key Features

- Uses **ALS (Spark MLlib)** for distributed recommendations  
- Includes **Cosine Similarity** and **SVD** for comparison  
- Displays recommendations in a clean UI  
- Uses caching to improve performance  
- Handles large datasets efficiently  

---

## Tech Stack

- Python 3.10  
- PySpark (Apache Spark)  
- Scikit-learn  
- Surprise (for SVD)  
- Pandas, NumPy  
- Streamlit  

---

## Project Structure

```
distributed-recommendation-system-spark-als/
│
├── app.py
├── als_model.py
├── spark_processing.py
├── data_preprocessing.py
├── models.py
├── evaluation.py
├── test_project.py
│
└── README.md
```

---

## Setup Instructions

### 1. Install Requirements

- Install **Python 3.10**
- Install **JDK 11 (for Spark)**

Set environment variable:

JAVA_HOME = C:\Program Files\Eclipse Adoptium\jdk-11.x.x

Add to PATH:

%JAVA_HOME%\bin

---

### 2. Clone the Repository

git clone https://github.com/YOUR_USERNAME/MLBD-Recommendation-System.git  
cd MLBD-Recommendation-System

---

### 3. Create Virtual Environment

python -m venv venv  
venv\Scripts\activate

---

### 4. Install Dependencies

pip install --upgrade pip  
pip install numpy==1.26.4 pandas scikit-learn scikit-surprise pyspark streamlit

---

## Running the Project

Run the Streamlit app:

streamlit run app.py

Then open in browser:

http://localhost:8501

---

## Common Issues

**1. Java Gateway Error**  
- Make sure Java is installed  
- Check JAVA_HOME  

**2. Surprise Installation Error**  
pip install numpy==1.26.4  
pip install scikit-surprise  

**3. Slow Performance**  
- Reduce dataset size for testing  

---

## Model Overview

**ALS (Main Model)**  
- Distributed collaborative filtering  
- Works efficiently on large datasets  

**Cosine Similarity**  
- Recommends based on item similarity  

**SVD**  
- Learns hidden patterns and predicts ratings  

---

## Optimization

We used caching (`@st.cache_resource`) to avoid retraining the model again and again, which improves performance.

---

## Final Overview

This project demonstrates how machine learning models can be combined with distributed systems to build scalable recommendation systems.

---

## Contributors

- Divyansh Rawat (M25CSA009)
- Pranav Kumar J (M25CSA021)
- Prapti Halder (M25CSA022)
