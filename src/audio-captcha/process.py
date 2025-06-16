import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already present
nltk.download('vader_lexicon')

augmentation = "noise"
thresh = 0.3

# Load the CSV
csv_path = f"results/audio-captcha/{augmentation}/results_raw.csv"
df = pd.read_csv(csv_path)

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Compute sentiment scores
def get_sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']

df['comparison_score'] = df['comparison'].apply(get_sentiment_score)

# Grade thresholds
def classify_score(score):
    if score >= thresh:
        return 'Pass'
    elif score <= -thresh:
        return 'Fail'
    else:
        return 'Borderline'

df['grade'] = df['comparison_score'].apply(classify_score)

# Save cleaned results
df.to_csv(f"results/audio-captcha/{augmentation}/results_clean.csv", index=False)

# Summary as DataFrame
grade_counts = df['grade'].value_counts().rename_axis('grade').reset_index(name='count')
score_stats = df[['comparison_score']].describe().transpose().reset_index().rename(columns={'index': 'metric'})

# Combine both into one summary DataFrame
summary_df = pd.concat([grade_counts, score_stats], axis=0, ignore_index=True)

# Save summary to CSV
summary_df.to_csv(f"results/audio-captcha/{augmentation}/results_summary.csv", index=False)

# Also print for reference
print(grade_counts)
print("\nScore summary:")
print(score_stats)
