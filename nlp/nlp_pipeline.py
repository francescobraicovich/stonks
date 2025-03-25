import pandas as pd
from nlp_utils import score_text_gemini, system_text
import os
from sklearn.model_selection import train_test_split
import json

stock_mentions = "../data/stock_mentions.parquet"
stock_mentions = pd.read_parquet(stock_mentions)

with open("api_key.txt", "r") as f:
    api_key = f.read().strip()

def score_row_gemini(row):
    score, tokens = score_text_gemini(row['text'], row['ticker'], api_key)
    print(f'scored {row['origin']} at index {row['original_index']}')
    return score

def score_dataframe(df, start_index=0, end_index=99):
    return df.iloc[start_index:end_index].apply(score_row_gemini, axis=1)

def apply_scores(df, start_index=0, end_index=99):
    reduced_df = df.iloc[start_index:end_index]
    reduced_df['score'] = score_dataframe(df, start_index, end_index)
    return reduced_df

def save_scores(df, start_index=0, end_index=99):
    reduced_df = apply_scores(df, start_index, end_index)
    reduced_df.to_parquet(f"fine_tuning/data/partial/reduced_stock_mentions_{first_index}_{last_index}.parquet")
    return None

def score_full_dataset(df, batch_size=100):
    for i in range(0, len(df), batch_size):
        try:
            save_scores(df, i, i + batch_size)
        except:
            print(f"failed to save batch {i} to {i + batch_size}")
    return None

def retrieve_scores(df, start_index=0, end_index=99):
    return pd.read_parquet(f"fine_tuning/data/partial/reduced_stock_mentions_{start_index}_{end_index}.parquet")

def retrieve_full_scores(df):
    directory = os.fsencode('fine_tuning/data/partial/')

    filenames = []
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".parquet"): 
            filenames.append(filename)
            continue
        else:
            continue

    filenames = sorted(filenames)
    dataframes = []

    for filename in filenames:
        dataframes.append(pd.read_parquet(f"fine_tuning/data/partial/{filename}"))

    return pd.concat(dataframes)



def system_row(row):
    prompt = system_text(row['ticker'])
    return prompt

def system_dataframe(df):
    df['system_text'] = df.apply(system_row, axis=1)
    return df

def to_json_df(df):
    return pd.DataFrame([{"messages": [{"role": "system", "content": row["system_text"]}, {"role": "user", "content": row["text"]},{"role": "assistant", "content": row["score"]}]} for _, row in df.iterrows()])

def split_json(json_df):
    # Split into 80% train, 20% temp (which will be split further)
    train_data, temp_data = train_test_split(json_df, test_size=0.2, random_state=42)

    # Split temp into 50% validation, 50% test (so each is 10% of total)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Convert DataFrames back to list format for JSON saving
    train_json = train_data.to_dict(orient="records")
    val_json = val_data.to_dict(orient="records")
    test_json = test_data.to_dict(orient="records")

    return train_json, val_json, test_json

def save_jsons(train_json, val_json, test_json):
    # Save train set
    with open("fine_tuning/data/json/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_json:
            f.write(json.dumps(item) + '\n')

    # Save validation set
    with open("fine_tuning/data/json/valid.jsonl", "w", encoding="utf-8") as f:
        for item in val_json:
            f.write(json.dumps(item) + '\n')

    # Save test set
    with open("fine_tuning/data/json/test.jsonl", "w", encoding="utf-8") as f:
        for item in test_json:
            f.write(json.dumps(item) + '\n')