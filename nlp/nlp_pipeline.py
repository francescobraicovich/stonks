# DEPRECATED: This file is deprecated and will be removed in the future.
# DEPRECATED: Please use the nlp/nlp_utils.py file instead.

import pandas as pd
from mlx_lm import load, generate

# suppose we have a list of comments with the following attributes: text, stocks

def score_text(text, stock, model_path):
    """
    Classifies the sentiment of a given text comment about a specific stock.

    Args:
        text (str): The text comment to be analyzed.
        stock (str): The stock symbol or name that the comment is about.
        model_path (str): The file path to the pre-trained model and tokenizer.

    Returns:
        int: The sentiment classification of the comment:
             1 if the comment is positive (bullish) about the stock,
            -1 if the comment is negative (bearish) about the stock,
             0 if the comment is neutral about the stock.
    """
    model, tokenizer = load(model_path)

    messages = [{
        'role': 'system',
        'content': f'''
                    You need to act as an expert linguist, trader and researcher, read the reddit comment about {stock} that is given to you, and classify wether the comment is bullish or bearish about {stock}.
                    Your answer should be 1 if the comment is positive about the stock, -1 if it is negative, and 0 if it is neutral. Do not reply to the comment, just classify it.
                    ''',
    },
    {
        'role': 'user',
        'content': text,
    }]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    text = generate(model, tokenizer, prompt=prompt, verbose=False)

    try:
        text = int(text)
    except ValueError:
        text = 0
    
    return text

def score_comments(comments, model_path):
    """
    Classifies the sentiment of a list of text comments about specific stocks.

    Args:
        comments (list): A list of dictionaries where each dictionary has the keys 'text' and 'stocks'.
        model_path (str): The file path to the pre-trained model and tokenizer.

    Returns:
        pd.DataFrame: A DataFrame with the columns 'text', 'stocks', and 'sentiment'.
                      The 'sentiment' column contains the sentiment classification of each comment.
    """
    data = []
    for comment in comments:
        if len(comment['stocks']) != 1:
            continue
        sentiment = score_text(comment['text'], comment['stocks'], model_path)
        data.append({
            'text': comment['text'],
            'stocks': comment['stocks'],
            'sentiment': sentiment
        })

    return pd.DataFrame(data)

