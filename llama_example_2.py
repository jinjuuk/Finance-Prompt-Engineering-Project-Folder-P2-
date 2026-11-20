import ollama

ollama.pull('martain7r/finance-llama-8b:q4_k_m')

news_data = """ {
    {
    'headline': 'Apple Posts Fiscal Q4 Beat On Strong iPhone, Services Sales'
    'contents': 'Apple beat expectations for fiscal Q4, earning $2.10 a share (versus $1.99 expected) on $114.48 billion in revenue (versus $113.4 billion). 
    Earnings rose 13.5% and sales rose 11.7% year over year. iPhone (+17.5%), Services (+9.8%) and Mac (+23.7%) grew, while iPad (-10.4%) and Wearables (-7.9%) fell. 
    Apple guided December-quarter revenue growth of 7–10%, above analysts’ 5.8%. In his first report as CEO, John Ternus highlighted the iPhone Duo, 
    iPhone 18 Pro lineup and new Siri AI. Despite this, the stock fell over 4% after hours to 324.11.'
    }
}"""


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


def predict_with_llm(news_data= news_data):  
    # IMPROVED SYSTEM PROMPT - With portfolio context and trading logic
    system_prompt = """You are a financial analyst AI that provides sentiment analysis by analysing the news data"""

    # IMPROVED USER PROMPT - With portfolio state information
    user_prompt = f"""Read News article: {news_data}

Is this news POSITIVE, NEGATIVE or NEUTRAL for Apple's share price?
Answer ONLY with a JSON object in exactly this format:
{{"reason": "one short sentence in your own words", "sentiment": "POSITIVE or NEGATIVE or NEUTRAL"}}"""
    
    response = ollama.chat(
        model='martain7r/finance-llama-8b:q4_k_m',
        format=ANSWER_SCHEMA,     # forces the answer into the JSON shape above
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], 
        options={
            'temperature': 0.1,  # Lower temperature for more consistent outputs
            'num_predict': 500,  # Reduced to prevent overly long responses
            'top_p': 0.9,
            'top_k': 40
        }
    )
    # Parse response
    return response['message']['content']

if __name__ == '__main__':
    result = predict_with_llm()
    print(result)