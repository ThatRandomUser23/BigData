import pandas as pd
from textblob import TextBlob

# Load the dataset
df = pd.read_excel('ExcelData.xlsx', sheet_name='Combined')

# Analyze user tweet frequency
user_frequencies = df['name'].value_counts()
user_frequencies_df = user_frequencies.reset_index()
user_frequencies_df.columns = ['User', 'Tweet_Count']

# Save the analysis to a new CSV file
user_frequencies_df.to_csv('User_Tweet_Frequency_Analysis.csv', index=False)

# Display the top 5 users for verification
print(user_frequencies_df.head())


# Generate sentiment polarity for JetBlueNews tweets
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Show the response by JetBlueNews
df_jetblue = df[df['name'] == 'JetBlueNews']

df_jetblue['polarity'] = df_jetblue['text'].apply(get_polarity)
# Calculate the mean polarity from JetBlueNews towards each airline
mean_polarity_jetblue = df_jetblue.groupby('airline')['polarity'].mean().reset_index()
mean_polarity_jetblue.columns = ['Airline', 'Mean_Polarity']
print(mean_polarity_jetblue)

# Show the response NOT by JetBlueNews
df_others = df[df['name'] != 'JetBlueNews']
# Generate sentiment polarity for other airlines tweets
df_others['polarity'] = df_others['text'].apply(get_polarity)
# Calculate the mean polarity for each airline
mean_polarity = df_others.groupby('airline')['polarity'].mean().reset_index()
mean_polarity.columns = ['Airline', 'Mean_Polarity']
print(mean_polarity)

# Show the response by JetBlueNews
print(df_jetblue['text'].head(20))

# Compare the polarity of JetBlueNews against non-JetBlueNews on Delta and Virgin America Airlines
df_jetblue_delta = df_jetblue[df_jetblue['airline'] == 'Delta']
df_jetblue_delta['polarity'] = df_jetblue_delta['text'].apply(get_polarity)
df_others_delta = df_others[df_others['airline'] == 'Delta']
df_others_delta['polarity'] = df_others_delta['text'].apply(get_polarity)
# Do t-test to compare the means
from scipy import stats
t_stat, p_value = stats.ttest_ind(df_jetblue_delta['polarity'], df_others_delta['polarity'])
print(f"JetBlueNews vs Others (Delta) - t-statistic: {t_stat}, p-value: {p_value}")
# Compare the polarity of JetBlueNews against non-JetBlueNews on Virgin America Airlines
df_jetblue_virgin = df_jetblue[df_jetblue['airline'] == 'Virgin America']
df_jetblue_virgin['polarity'] = df_jetblue_virgin['text'].apply(get_polarity)
df_others_virgin = df_others[df_others['airline'] == 'Virgin America']
df_others_virgin['polarity'] = df_others_virgin['text'].apply(get_polarity)
# Do t-test to compare the means
t_stat_virgin, p_value_virgin = stats.ttest_ind(df_jetblue_virgin['polarity'], df_others_virgin['polarity'])
print(f"JetBlueNews vs Others (Virgin America) - t-statistic: {t_stat_virgin}, p-value: {p_value_virgin}")

# Not significant but could be conflict of interest