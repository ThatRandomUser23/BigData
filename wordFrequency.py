import spacy
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_excel('ExcelData.xlsx', sheet_name='Combined')

# Preprocess tweets using spaCy
def preprocess_tweet(text):
    doc = nlp(text.lower())  # Convert to lowercase and process with spaCy
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]  # Lemmatize and remove stopwords/punctuation
    return words

df['processed_text'] = df['text'].apply(preprocess_tweet)

# Flatten all words into a single list for overall word frequency analysis
all_words = [word for words in df['processed_text'] for word in words]
word_frequencies = Counter(all_words).most_common(20)

# Visualize overall word frequencies
words, counts = zip(*word_frequencies)
plt.figure(figsize=(10, 6))
plt.bar(words, counts, color='skyblue')
plt.title('Top Words Across All Tweets')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save to CSV for further analysis
pd.DataFrame(word_frequencies, columns=['Word', 'Frequency']).to_csv('Overall_Word_Frequency.csv', index=False)
