import pandas as pd
import matplotlib.pyplot as plt
# import textblob
from textblob import TextBlob

df = pd.read_excel('ExcelData.xlsx', sheet_name='Combined')
df.head()

df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Calculate correlation
correlation = df['polarity'].corr(df['subjectivity'])

print(f"Correlation between polarity and subjectivity: {correlation}")
# Conclusion very weak; so we will not use subjectivity in our analysis

# Section 1: Data Visualization
# Create a table of each airline with the number of reviews, mean polarity, mean subjectivity, and sd polarity and subjectivity
airline_stats = df.groupby('airline').agg(
    reviews=('text', 'count'),
    mean_polarity=('polarity', 'mean'),
    mean_subjectivity=('subjectivity', 'mean'),
    sd_polarity=('polarity', 'std'),
    sd_subjectivity=('subjectivity', 'std')
).reset_index()
print(airline_stats)

# Perform Anova test
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
# Perform ANOVA
model = ols('polarity ~ C(airline)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
# Perform Tukey's test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df['polarity'], df['airline'])
print(tukey)
# Print the mean and sd of polarity and subjectivity for each airline   
print("Mean and SD of polarity and subjectivity for each airline:")

# Sort by category
# Function to classify tweets
def classify_tweet(tweet):
    classifications = {}
    for category, keywords in keyword_dicts.items():
        found = False
        for keyword in keywords:
            if keyword in tweet:
                found = True
                break
        classifications[category] = found
    return classifications

# Apply classification to tweets
import pandas as pd
import re

# Load the CSV file
# df = pd.read_csv('2025-Project-data-set-Tweets-CSV-file.csv', encoding='ISO-8859-1')

# Preprocess tweets
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuation
    return tweet

df['text'] = df['text'].apply(preprocess_tweet)


# Keyword dictionaries
efficiency_keywords = ["on-time", "punctual", "efficient", "delayed", "late"]
hospitality_keywords = ["friendly", "helpful", "courteous", "rude", "unfriendly"]
comfort_keywords = ["comfortable", "seats", "legroom", "clean", "dirty"]
food_beverage_keywords = ["food", "meal", "snack", "drink", "quality"]
safety_keywords = ["safe", "safety", "secure", "concerns"]

keyword_dicts = {
    "efficiency": efficiency_keywords,
    "hospitality": hospitality_keywords,
    "comfort": comfort_keywords,
    "food_beverage": food_beverage_keywords,
    "safety": safety_keywords
}

df['classifications'] = df['text'].apply(classify_tweet)
print(df.head())


# Lists to hold counts and mean polarities
counts = []
mean_polarities = []
categories = list(keyword_dicts.keys())

# Analyze classifications
for category in keyword_dicts.keys():
    count = df['classifications'].apply(lambda x: x[category]).sum()
    counts.append(count)
    
    # Ensure there are tweets in this category to avoid NaN
    filtered_df = df[df['classifications'].apply(lambda x: x[category])]
    if not filtered_df.empty:
        mean_polarity = filtered_df['polarity'].mean()
    else:
        mean_polarity = 0  # Default if no tweets in category
    mean_polarities.append(mean_polarity)
    
    print(f"{category}: {count} tweets")
    print(f"Mean polarity for {category}: {mean_polarity}")

# Create bar charts
plt.figure(figsize=(10,6))
plt.subplot(1, 2, 1)
plt.bar(categories, counts)
plt.title('Count of Tweets by Category')
plt.xlabel('Category')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.bar(categories, mean_polarities)
plt.title('Mean Polarity by Category')
plt.xlabel('Category')
plt.ylabel('Mean Polarity')

plt.tight_layout()
plt.show()


# Prepare data for grouped bar chart
# Calculate mean polarity for each airline and category
categories = list(keyword_dicts.keys())
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
plot_data = {}
for category, scores in quality_scores.items():
    plot_data[category] = scores

# Convert data into DataFrame for easier plotting
df_plot = pd.DataFrame(plot_data).T

airlines = list(set(df['airline']))
categories = list(quality_scores.keys())

data = {
    'Category': [],
    'Airline': [],
    'Score': []
}

for category in categories:
    for airline in airlines:
        score = quality_scores[category].get(airline, 0)  # Default to 0 if no score
        data['Category'].append(category)
        data['Airline'].append(airline)
        data['Score'].append(score)

df_grouped = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10,6))
for airline in airlines:
    airline_df = df_grouped[df_grouped['Airline'] == airline]
    plt.bar(airline_df['Category'], airline_df['Score'], label=airline)

plt.title('Airline Quality Scores Across Categories')
plt.xlabel('Category')
plt.ylabel('Mean Polarity Score')
plt.legend(title='Airline')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
