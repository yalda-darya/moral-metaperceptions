import os

import pandas as pd

# text for abortion2 and full_text for abortion1
TEXT_COLUMN_NAME = "text"

# Set the path to the directory containing the .xlsx files
xlsx_dir = os.path.join(os.getcwd(), "raw", "abortion_sentiment_prediction", "abortion2_xlsx_files")
print(f"{xlsx_dir=}")

csv_dir = os.path.join(os.getcwd(), "raw", "abortion_sentiment_prediction", "abortion2_csv_files")
print(f"{csv_dir=}")

# Create a list of all the .xlsx files in the directory
xlsx_files = [f for f in os.listdir(xlsx_dir) if f.endswith(".xlsx")]

print(f"Found {len(xlsx_files)} .xlsx files")

# Loop through the list of .xlsx files
for xlsx_file in xlsx_files:
    # Read each .xlsx file into a pandas dataframe
    print(xlsx_file)
    # ["conversation_id","conversation_id_str","created_at","full_text","id","id_str","lang","user.created_at","user.id","user.id_str","user.lang","user.name","user.screen_name","created_at_date","annotation"]
    df = pd.read_excel(
        os.path.join(xlsx_dir, xlsx_file),
        usecols=["id", TEXT_COLUMN_NAME, "annotation"],
        dtype={"id": str, "annotation": str, "text": str},
    )

    # Write the dataframe to a .csv file in the same directory as the original .xlsx file
    csv_file = os.path.splitext(xlsx_file)[0] + ".csv"
    df.to_csv(os.path.join(csv_dir, csv_file), index=False)
