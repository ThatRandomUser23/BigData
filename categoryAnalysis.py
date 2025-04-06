import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_excel('ExcelData.xlsx', sheet_name='Combined')

# Preprocess tweets
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuation
    return tweet

df['text'] = df['text'].apply(preprocess_tweet)

# Sentiment analysis
df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Keyword dictionaries
keyword_dicts = {
    "efficiency": ["on-time", "punctual", "efficient", "delayed", "late"],
    "hospitality": ["friendly", "helpful", "courteous", "rude", "unfriendly"],
    "comfort": ["comfortable", "seats", "legroom", "clean", "dirty"],
    "food_beverage": ["food", "meal", "snack", "drink", "quality"],
    "safety": ["safe", "safety", "secure", "concerns"]
}

# Function to classify tweets
def classify_tweet(tweet):
    classifications = {}
    for category, keywords in keyword_dicts.items():
        found = any(keyword in tweet for keyword in keywords)
        classifications[category] = found
    return classifications

# Apply classification to tweets
df['classifications'] = df['text'].apply(classify_tweet)

# Calculate mean polarity for each airline and category
categories = list(keyword_dicts.keys())
airlines = list(set(df['airline']))

quality_scores = {}
for category in categories:
    filtered_df = df[df['classifications'].apply(lambda x: x[category])]
    if not filtered_df.empty:
        quality_scores[category] = filtered_df.groupby('airline')['polarity'].mean().to_dict()
    else:
        quality_scores[category] = {}

# Calculate overall quality score
overall_scores = df.groupby('airline')['polarity'].mean().to_dict()
quality_scores['overall'] = overall_scores

# Prepare data for plotting
data = {
    'Category': [],
    'Airline': [],
    'Score': []
}
for category in categories:
    for airline in airlines:
        score = quality_scores[category].get(airline, 0)
        data['Category'].append(category)
        data['Airline'].append(airline)
        data['Score'].append(score)

df_grouped = pd.DataFrame(data)

# Pivot the data for easier plotting
df_pivot = df_grouped.pivot(index='Category', columns='Airline', values='Score')
print(df_pivot)

# Plot grouped bar chart
categories = df_pivot.index
x = np.arange(len(categories))  # Label locations
width = 0.15  # Width of bars

fig, ax = plt.subplots(figsize=(12, 6))

for i, airline in enumerate(df_pivot.columns):
    ax.bar(x + i * width, df_pivot[airline], width, label=airline)

# Add labels and title
ax.set_xlabel('Category')
ax.set_ylabel('Mean Polarity Score')
ax.set_title('Airline Quality Scores Across Categories')
ax.set_xticks(x + width * (len(df_pivot.columns) / 2 - 0.5))
ax.set_xticklabels(categories, rotation=45)
ax.legend(title='Airline')

plt.tight_layout()
plt.show()

from math import pi

# Prepare data for radar chart
categories = list(quality_scores.keys())
airlines = df['airline'].unique()
data = {airline: [quality_scores[category].get(airline, 0) for category in categories] for airline in airlines}

# Plot radar chart
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
for airline, values in data.items():
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    values += values[:1]  # Close the loop
    angles += angles[:1]
    ax.plot(angles, values, label=airline)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks([n / float(len(categories)) * 2 * pi for n in range(len(categories))])
ax.set_xticklabels(categories)
ax.set_title('Airline Quality Comparison')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

plt.tight_layout()
plt.show()
