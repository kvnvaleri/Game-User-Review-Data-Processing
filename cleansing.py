import pandas as pd
from scipy import stats
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('video_game_reviews.csv')

df = df.drop_duplicates(subset=['Game Title', 'Release Year'])
df['Platform'] = df['Platform'].replace({'PlayStation': 'PS4', 'Nintendo Switch': 'Switch', 'Mobile': 'Android/iOS'})
df['Age Group Targeted'] = df['Age Group Targeted'].replace({'All Ages': 'Everyone', 'Adults': 'Mature', 'Teens': 'Teen', 'Kids': 'Everyone 10+'})
df['Price'] = df['Price'].astype(str).str.replace('$', '').astype(float)
df['User Review Text'] = df['User Review Text'].fillna('No Review')
df['Release Year'] = pd.to_numeric(df['Release Year'], errors='coerce').astype('Int64')
df = df[df['User Rating'] <= 100]
df = df[df['User Rating'] >= 0]
z_scores = stats.zscore(df['Game Length (Hours)'].dropna())
df = df[(z_scores < 3) | (z_scores > -3)]
df['Genre'] = df['Genre'].str.replace('RPG', 'Role-Playing Game').replace('MMORPG', 'Role-Playing Game')

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['Sentiment Score'] = df['User Review Text'].apply(analyze_sentiment)
df['Sentiment'] = df['Sentiment Score'].apply(
    lambda x: 'Positive' if x > 0.2 else 'Negative' if x < -0.2 else 'Neutral'
)

df.to_csv('cleaned_data_video_game_reviews.csv', index=False)