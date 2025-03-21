import pandas as pd
from mlx_lm import load, generate
from google import genai
from google.genai import types
from ratelimit import limits, sleep_and_retry

# 30 calls per minute
CALLS = 15
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def check_limit():
    ''' Empty function just to check for calls to API '''
    return None

def system_text(stock):
    prompt = f'''You need to act as an expert linguist, trader and researcher, read the reddit comment, and classify wether the comment is bullish or bearish about the ticker: {stock}.
                Your answer should be 1 if the comment is positive about {stock}, -1 if it is negative, and 0 if it is neutral. Do not reply to the comment, just classify it.
                '''
    return prompt

def score_text_gemini(text, stock, api_key):
    client = genai.Client(api_key=api_key)
    check_limit()
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_text(stock)),
    contents=text
    )

    output = response.text
    tokens = response.usage_metadata.total_token_count

    try:
        output = int(output)
    except ValueError:
        output = 0
    
    return output, tokens

def score_text_local(text, stock, model_path):
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
    output = generate(model, tokenizer, prompt=prompt, verbose=False)

    try:
        output = int(output)
    except ValueError:
        output = 0
    
    return output