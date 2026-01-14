import json

import pandas as pd
import pymongo

# Load the data
with open("data/firstpass.json", "r") as f:
    data = json.load(f)

# Connect to your MongoDB instance
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["PSYC626_Abortion_Project_Database"]  # Replace with your database name
tweets_collection = db["reproduction_tweets"]


years = [2015, 2016, 2017, 2018, 2019, 2020]
hashtag_types = ["both", "none", "prochoice", "prolife"]
for year in years:
    for hashtag_type in hashtag_types:
        ids_for_query = []
        for year_hashtag_item in data:
            if (
                year_hashtag_item["_id"]["year"] == year
                and year_hashtag_item["_id"]["hashtagType"] == hashtag_type
            ):
                ids_for_query.extend([item["$oid"] for item in year_hashtag_item["tweetIdsSubset"]])

        tweets = tweets_collection.find({"_id": {"$in": ids_for_query}})

        df = pd.DataFrame(list(tweets))

        df.to_csv("data/{year}_{hashtag_type}", index=False)
