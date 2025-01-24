import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Video Game Analytics Dashboard')

df = pd.read_csv('clustered_data.csv')
feature_importance = pd.read_csv('feature_importance.csv')

st.header('Clustering Insights')
selected_cluster = st.selectbox('Pilih Cluster', df['Cluster'].unique())
cluster_data = df[df['Cluster'] == selected_cluster]

st.subheader(f'Karakteristik Cluster {selected_cluster}')
st.write(cluster_data[['User Rating', 'Price', 'Genre']].describe())

st.header('Rating Prediction Model')
st.image('business_feature_importance.png')
st.write("Top Features Affecting Game Ratings:")
st.dataframe(feature_importance.head(10))

st.header('User Sentiment Analysis')
sentiment_dist = df['Sentiment'].value_counts(normalize=True)
st.bar_chart(sentiment_dist)