import os

import pandas as pd

# create an empty dataframe to store the combined data
combined_data = pd.DataFrame()

# print current working directory
print(os.getcwd())

# loop through all csv files in the csv_files folder
for file_name in os.listdir("./raw/abortion_sentiment_prediction/abortion2_csv_files"):
    if file_name.endswith(".csv"):
        # read the csv file into a dataframe
        file_path = os.path.join("./raw/abortion_sentiment_prediction/abortion2_csv_files", file_name)
        df = pd.read_csv(file_path, dtype={"text": str, "id": str, "annotation": str})

        # Drop the rows with empty annotation
        df = df[df["annotation"].isna()]

        # rename the 'id' to 'id_str' and 'text' to 'full_text'
        df.rename(columns={"id": "id_str", "text": "full_text"}, inplace=True)

        # add a column with the file name to the dataframe
        # df["data_source"] = file_name

        # concatenate the dataframe to the combined_data dataframe
        combined_data = pd.concat([combined_data, df], ignore_index=True)

# save the combined data to a csv file
combined_data.to_csv("./raw/abortion_sentiment_prediction/wave2_not_annotated.csv", index=False)
