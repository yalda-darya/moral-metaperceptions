import pandas as pd
import os

if __name__ == "__main__":
    # Load the dataset
    with open(os.path.join("data", "raw", "isi_dataset_complete.pkl"), "rb") as pickle_file:
        data = pd.read_pickle(pickle_file)

    print(f"Loaded {len(data)} tweets")

    # Attempt to convert 'created_at' to datetime, skipping errors
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce")

    # Optional: Drop rows where 'created_at' could not be converted
    # data = data.dropna(subset=['created_at'])

    # Proceed with the rest of your data processing and analysis
    data["year_month"] = data["created_at"].dt.to_period("M")

    # Calculate the total sample size (10% of all tweets)
    total_tweets = len(data)
    total_sample_size = int(total_tweets * 0.1)

    # Calculate the number of months
    num_months = data["year_month"].nunique()

    # Determine the number of tweets to sample from each month
    tweets_per_month = total_sample_size // num_months

    # Define a sampling function
    def sample_tweets(group):
        sample_size = min(len(group), tweets_per_month)
        return group.sample(n=sample_size, random_state=1)

    # Apply the sampling function
    sampled_df = data.groupby("year_month").apply(sample_tweets).reset_index(drop=True)

    # Save to a CSV file
    sampled_df.to_csv("sampled_tweets_equal_per_month.csv", index=False)

    print(f"Saved {len(sampled_df)} sampled tweets to sampled_tweets_equal_per_month.csv")
