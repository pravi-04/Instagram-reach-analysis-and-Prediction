import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv("Instagram.csv", encoding = 'latin1')
print(data.head())

data.isnull().sum()

data.info()

plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.displot(data['From Home'], kde=True)
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.displot(data['From Hashtags'],kde=True)
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.displot(data['From Explore'],kde=True)
plt.show()

home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()

import nltk
from collections import Counter

def analyze_text_frequency(text_column):
    text = " ".join(text_column)  # Combine all text for frequency analysis
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and lowercase words (optional: remove stopwords)
    word_counts = Counter(tokens)
    return word_counts.most_common(10)  # Get top 10 most frequent words

# Analyze caption and hashtag content
caption_word_counts = analyze_text_frequency(data["Caption"])
hashtag_word_counts = analyze_text_frequency(data["Hashtags"])

# Print top 10 most frequent words for captions and hashtags
print("Top 10 Most Frequent Words in Captions:")
print(caption_word_counts)
print("\nTop 10 Most Frequent Words in Hashtags:")
print(hashtag_word_counts)

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")
figure.show()

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title = "Relationship Between Shares and Total Impressions")
figure.show()

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title = "Relationship Between Post Saves and Total Impressions")
figure.show()

numerical_data = data.select_dtypes(include=[np.number])
correlation = numerical_data.corr()
print(correlation["Impressions"].sort_values(ascending=False))

conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)

figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title = "Relationship Between Profile Visits and Followers Gained")
figure.show()

x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)