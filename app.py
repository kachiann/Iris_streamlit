import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame from the Iris dataset
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                       columns=iris['feature_names'] + ['target'])
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Train a simple Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit app
def main():
    st.title('Iris Flower Prediction')
    
    # Add a menu for different pages
    menu = ["Home", "Dataset", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Iris Flower Classification App")
        st.write("""
        This app predicts the Iris flower species based on sepal and petal measurements.
        Use the menu on the left to navigate between pages:
        - Dataset: View and explore the Iris dataset
        - Prediction: Make predictions using the trained model
        """)
        
    elif choice == "Dataset":
        st.subheader("Iris Dataset")
        st.write(iris_df)
        
        st.subheader("Dataset Description")
        st.write(iris.DESCR)
        
        st.subheader("Data Visualization")
        fig, ax = plt.subplots()
        sns.pairplot(iris_df, hue="species", height=2)
        st.pyplot(fig)
        
    elif choice == "Prediction":
        st.subheader("Prediction")
        
        st.sidebar.header('Input Parameters')
        
        def user_input_features():
            sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
            sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
            petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
            petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
            data = {'sepal length (cm)': sepal_length,
                    'sepal width (cm)': sepal_width,
                    'petal length (cm)': petal_length,
                    'petal width (cm)': petal_width}
            features = pd.DataFrame(data, index=[0])
            return features
        
        df = user_input_features()
        
        st.subheader('User Input parameters')
        st.write(df)
        
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)
        
        st.subheader('Prediction')
        st.write(iris.target_names[prediction][0])
        
        st.subheader('Prediction Probability')
        st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))

if __name__ == '__main__':
    main()