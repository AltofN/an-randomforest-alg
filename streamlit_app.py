import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')
st.info('This app builds a machine learning model!')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    return df

df = load_data()

with st.expander('Data'):
    st.write('**Raw data**')
    st.dataframe(df)

    X_raw = df.drop('species', axis=1)
    y_raw = df['species']

    st.write('**X**')
    st.dataframe(X_raw)

    st.write('**y**')
    st.dataframe(y_raw)

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Sidebar input features
with st.sidebar:
    st.header('Input features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))

# Create input DataFrame
input_df = pd.DataFrame({
    'island': [island],
    'bill_length_mm': [bill_length_mm],
    'bill_depth_mm': [bill_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [gender]
})

# One-hot encoding
encode_cols = ['island', 'sex']
X_encoded = pd.get_dummies(X_raw, prefix=encode_cols)
input_encoded = pd.get_dummies(input_df, prefix=encode_cols)
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Encode target
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y_encoded = y_raw.map(target_mapper)

with st.expander('Input features'):
    st.write('**Input penguin**')
    st.dataframe(input_df)

with st.expander('Data preparation'):
    st.write('**Encoded X (input penguin)**')
    st.dataframe(input_encoded)
    st.write('**Encoded y**')
    st.dataframe(y_encoded)

# Train model (cached)
@st.cache_resource
def train_model(X, y):
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf

clf = train_model(X_encoded, y_encoded)

# Make predictions
prediction = clf.predict(input_encoded)
prediction_proba = clf.predict_proba(input_encoded)

# Format output
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba, column_config={
    'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', min_value=0, max_value=1),
    'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', min_value=0, max_value=1),
    'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', min_value=0, max_value=1),
}, hide_index=True)

species_labels = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f"Predicted species: {species_labels[prediction[0]]}")
