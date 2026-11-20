import json
import os

import ollama
import pandas as pd

MODEL = 'martain7r/finance-llama-8b:q4_k_m'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_FILE = os.path.join(SCRIPT_DIR, 'apple_news_test_v3.csv')   # change this to use another news file

ollama.pull(MODEL)

# The AI must answer in exactly this JSON shape.
# "reason" comes first so the AI explains itself before it picks the sentiment.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]},
    },
    "required": ["reason", "sentiment"],
}


def load_news(path=NEWS_FILE):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='cp1252')


def predict_with_llm(headline, summary):
    # SYSTEM PROMPT - who the AI is
    system_prompt = """You are a financial analyst AI that provides sentiment analysis by analysing news about Apple."""

    # USER PROMPT - the news to analyse and the answer form
    user_prompt = f"""News headline: {headline}
News summary: {summary}

Is this news POSITIVE, NEGATIVE or NEUTRAL for Apple's share price?
Answer ONLY with a JSON object in exactly this format:
{{"reason": "one short sentence in your own words", "sentiment": "POSITIVE or NEGATIVE or NEUTRAL"}}"""

    response = ollama.chat(
        model=MODEL,
        format=ANSWER_SCHEMA,     # forces the answer into the JSON shape above
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={
            'temperature': 0.0,     # always pick the most likely answer
            'seed': 42,
            'num_predict': 500,   # stop overly long answers
        }
    )
    # Turn the JSON text into a Python dictionary
    return json.loads(response['message']['content'])


if __name__ == '__main__':
    news = load_news()
    print(f"Read {len(news)} news articles from {os.path.basename(NEWS_FILE)}\n")

    results = []
    for i, row in news.iterrows():
        answer = predict_with_llm(row['Headline'], row['Summary'])
        results.append(answer['sentiment'])

        print(f"News {i + 1} of {len(news)} ({row['Date']}): {row['Headline']}")
        print(json.dumps(answer, indent=2))
        print()

    news['AI_Sentiment'] = results
    print("Summary:")
    print(news['AI_Sentiment'].value_counts().to_string())
