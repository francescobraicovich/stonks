import pandas as pd
from nlp_utils import score_text_gemini, system_text
import os
from sklearn.model_selection import train_test_split
import json


def score_row_gemini(row):
    """
    Scores a single row of data using the Gemini scoring function.

    This function takes a row of data, extracts the 'text' and 'ticker' fields,
    and uses the `score_text_gemini` function to compute a score and tokenize
    the text. The score is then returned.

    Args:
        row (dict): A dictionary representing a single row of data. It must
                    contain the keys:
                    - 'text' (str): The text to be scored.
                    - 'ticker' (str): The ticker symbol associated with the text.

    Returns:
        int: The computed score for the given row.
    """
    score, tokens = score_text_gemini(row['text'], row['ticker'], api_key)
    # print(f'scored {row['origin']} at index {row['original_index']}')
    return score


def save_scores(df, path, start_index=0, end_index=99):
    """
    Saves a subset of a DataFrame with calculated scores to a Parquet file.

    This function takes a DataFrame, applies a scoring function to a subset of its rows,
    and saves the resulting DataFrame to a Parquet file. The subset is determined by
    the `start_index` and `end_index` parameters.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data to process.
        path (str): The directory path where the output Parquet file will be saved.
        start_index (int, optional): The starting index of the subset. Defaults to 0.
        end_index (int, optional): The ending index (exclusive) of the subset. Defaults to 99.

    Returns:
        None
    """
    reduced_df = df.iloc[start_index:end_index]
    reduced_df['score'] = reduced_df.apply(score_row_gemini, axis=1)
    reduced_df.to_parquet(f"{path}/reduced_stock_mentions_{start_index}_{end_index}.parquet")
    return None

def score_full_dataset(df, path, batch_size=100, start_index=0):
    """
    Scores the entire dataset in batches and saves the results.

    This function processes a DataFrame in chunks of a specified batch size,
    scoring each batch and saving the results to the specified path. If an
    error occurs while saving a batch, it logs the error and continues with
    the next batch.

    Args:
        df (pandas.DataFrame): The dataset to be scored.
        path (str): The file path where the scores will be saved.
        batch_size (int, optional): The number of rows to process in each batch. Defaults to 100.

    Returns:
        None
    """
    for i in range(start_index, len(df), batch_size):
        try:
            save_scores(df, path, i, i + batch_size)
        except Exception as e:
            print(f"failed to save batch {i} to {i + batch_size}: {e}")
    return None

def retrieve_full_scores(folder_path):
    """
    Retrieve and concatenate all Parquet files from a specified folder.
    This function reads all files with a `.parquet` extension in the given folder,
    sorts them by filename, and concatenates their contents into a single pandas DataFrame.
    Args:
        folder_path (str): The path to the folder containing the Parquet files.
    Returns:
        pandas.DataFrame: A DataFrame containing the concatenated data from all Parquet files.
    """
    directory = os.fsencode(folder_path)

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
        dataframes.append(pd.read_parquet(os.path.join(folder_path, filename)))

    return pd.concat(dataframes)

def system_row(row):
    """
    Generates a system prompt based on the provided row data.

    Args:
        row (dict): A dictionary containing data for a single row. 
                    It must include a 'ticker' key.

    Returns:
        str: A prompt generated using the 'ticker' value from the row.
    """
    prompt = system_text(row['ticker'])
    return prompt

def system_dataframe(df):
    """
    Applies the `system_row` function to each row of the given DataFrame and 
    creates a new column named 'system_text' with the results.

    Args:
        df (pandas.DataFrame): The input DataFrame to process.

    Returns:
        pandas.DataFrame: The modified DataFrame with an additional 'system_text' column.
    """
    df['system_text'] = df.apply(system_row, axis=1)
    return df

def to_json_df(df):
    """
    Converts a DataFrame into a JSON-like DataFrame suitable for NLP processing.

    Each row of the input DataFrame is transformed into a dictionary with a 
    "messages" key, which contains a list of dictionaries representing a 
    conversation. The conversation includes three roles: "system", "user", 
    and "assistant", with their respective content extracted from the input 
    DataFrame columns.

    Args:
        df (pd.DataFrame): Input DataFrame with the following required columns:
            - "system_text": Text content for the "system" role.
            - "text": Text content for the "user" role.
            - "score": Text content for the "assistant" role.

    Returns:
        pd.DataFrame: A DataFrame where each row contains a single dictionary 
        with a "messages" key, formatted for NLP tasks.
    """
    return pd.DataFrame([{"messages": [{"role": "system", "content": row["system_text"]}, {"role": "user", "content": row["text"]},{"role": "assistant", "content": row["score"]}]} for _, row in df.iterrows()])

def split_json(json_df):
    """
    Splits a JSON-like DataFrame into training, validation, and test datasets.

    This function divides the input DataFrame into three subsets:
    - 80% for training
    - 10% for validation
    - 10% for testing

    The splitting is performed using a random seed for reproducibility.

    Args:
        json_df (pd.DataFrame): A pandas DataFrame containing the JSON-like data to be split.

    Returns:
        tuple: A tuple containing three lists of dictionaries:
            - train_json (list): Training data in JSON-like format.
            - val_json (list): Validation data in JSON-like format.
            - test_json (list): Test data in JSON-like format.
    """
    # Split into 80% train, 20% temp (which will be split further)
    train_data, temp_data = train_test_split(json_df, test_size=0.2, random_state=42)

    # Split temp into 50% validation, 50% test (so each is 10% of total)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Convert DataFrames back to list format for JSON saving
    train_json = train_data.to_dict(orient="records")
    val_json = val_data.to_dict(orient="records")
    test_json = test_data.to_dict(orient="records")

    return train_json, val_json, test_json

def save_jsons(train_json, val_json, test_json, folder_path):
    """
    Saves training, validation, and test datasets as JSON Lines files in the specified folder.

    Args:
        train_json (list): A list of dictionaries representing the training dataset.
        val_json (list): A list of dictionaries representing the validation dataset.
        test_json (list): A list of dictionaries representing the test dataset.
        folder_path (str): The path to the folder where the JSON Lines files will be saved.

    Files Created:
        - train.jsonl: Contains the training dataset, with each dictionary serialized as a JSON object on a new line.
        - val.jsonl: Contains the validation dataset, with each dictionary serialized as a JSON object on a new line.
        - test.jsonl: Contains the test dataset, with each dictionary serialized as a JSON object on a new line.
    """
    # Save train set
    with open(os.path.join(folder_path, 'train.jsonl'), "w", encoding="utf-8") as f:
        for item in train_json:
            f.write(json.dumps(item) + '\n')

    # Save validation set
    with open(os.path.join(folder_path, 'valid.jsonl'), "w", encoding="utf-8") as f:
        for item in val_json:
            f.write(json.dumps(item) + '\n')

    # Save test set
    with open(os.path.join(folder_path, 'test.jsonl'), "w", encoding="utf-8") as f:
        for item in test_json:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    rel_stock_mentions_path = os.path.join('data', 'stock_mentions.parquet')
    rel_api_key_path = "api_key.txt"
    rel_scored_partial_path = os.path.join('fine_tuning', 'data', 'partial')
    rel_json_path = os.path.join('fine_tuning', 'data', 'json')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    rel_stock_mentions_path = os.path.join(parent_dir, rel_stock_mentions_path)
    rel_api_key_path = os.path.join(current_dir, rel_api_key_path)
    rel_scored_partial_path = os.path.join(current_dir, rel_scored_partial_path)
    rel_json_path = os.path.join(current_dir, rel_json_path)


    stock_mentions = pd.read_parquet(rel_stock_mentions_path)
    stock_mentions = stock_mentions.drop_duplicates(subset=['ticker', 'text'], keep='first')



    print("loaded file")

    with open(rel_api_key_path, "r") as f:
        api_key = f.read().strip()
    
    score_full_dataset(stock_mentions, rel_scored_partial_path, batch_size=50)

    print("scored file")

    scored_df = retrieve_full_scores(rel_scored_partial_path)

    scored_df = system_dataframe(scored_df)

    json_df = to_json_df(scored_df)

    train_json, val_json, test_json = split_json(json_df)

    save_jsons(train_json, val_json, test_json, rel_json_path)

    print("saved json files")