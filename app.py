import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a simple Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit app
def main():
    st.title('Iris Flower Prediction')
    
    st.sidebar.header('Input Parameters')
    
    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_features()
    
    st.subheader('User Input parameters')
    st.write(df)
    
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    st.subheader('Prediction')
    st.write(iris.target_names[prediction])
    
    st.subheader('Prediction Probability')
    st.write(prediction_proba)

if __name__ == '__main__':
    main()