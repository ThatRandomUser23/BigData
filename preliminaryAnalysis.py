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