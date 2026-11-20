"""
AI Prompt Trading Simulator
===========================
Students write a SYSTEM PROMPT (the AI's trading strategy).
For every Apple news article the program:
  1. builds a USER PROMPT with the news and today's numbers (fixed, written by the program),
  2. asks the local Ollama model for a decision,
  3. forces the answer into a fixed JSON form {"action", "sentiment", "reason"} (Ollama JSON schema),
  4. carries out the action: BUY = spend all cash, SELL = sell all shares, HOLD = do nothing.
Then it shows the trades on the Apple price chart and how much profit was made.

Run:   python prompt_dashboard.py
Open:  http://127.0.0.1:8050
Needs: Ollama running with the model below, and apple_stock_test.csv + apple_news_test.csv in this folder.
"""
import json
import os
import threading
import traceback

import dash
import ollama
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, ctx, no_update

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_FILE = os.path.join(SCRIPT_DIR, 'apple_stock_test_v2.csv')
NEWS_FILE = os.path.join(SCRIPT_DIR, 'apple_news_test_v3.csv')

MODEL = 'gemma4:e2b' #'martain7r/finance-llama-8b:q4_k_m'
INITIAL_CASH = 10000
SHORT_WINDOW = 5
LONG_WINDOW = 20
PRICE_COLUMN = 'Open'

# ---------------------------------------------------------------------------
# The three parts of every question to the AI
# ---------------------------------------------------------------------------
# 1. SYSTEM PROMPT - written by the student (this is only the starting example)
DEFAULT_SYSTEM_PROMPT = """You are a stock trader who trades Apple shares.
Your goal is to make a profit.

Your strategy:
- BUY when the news is positive for Apple.
- SELL when the news is negative for Apple.
- Otherwise HOLD."""

# 2. USER PROMPT - fixed, filled in by the program for every news article
USER_PROMPT_TEMPLATE = """Today's date: {date}
Apple share price today: ${price:,.2f}
Average price of the last 5 days: ${sma_short:,.2f}
Average price of the last 20 days: ${sma_long:,.2f}
Price trend: {trend}
Do we own Apple shares now? {position}
Cash available: ${cash:,.2f}

News headline: {headline}
News summary: {summary}

Follow your strategy and decide what to do today.
If the news is not really about Apple's business, or is unclear, the sentiment is NEUTRAL.
Check that your action follows your strategy.
Keep your reason to one or two short sentences, and do not copy the headline into it.
Answer ONLY with a JSON object in exactly this format:
{{"reason": "explain in your own words what this news means for Apple and why you choose this action", "sentiment": "POSITIVE or NEGATIVE or NEUTRAL", "action": "BUY or SELL or HOLD"}}"""

# 3. ANSWER FORM - Ollama only lets the model answer in exactly this JSON shape.
# The reason comes first so the model explains before it decides
# (tested: with "action" first, this model answered SELL to every news article).
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]},
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
    },
    "required": ["reason", "sentiment", "action"],
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    stock = pd.read_csv(STOCK_FILE, encoding='utf-8')
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock = stock.sort_values('Date').reset_index(drop=True)
    stock['SMA_Short'] = stock[PRICE_COLUMN].rolling(window=SHORT_WINDOW).mean()
    stock['SMA_Long'] = stock[PRICE_COLUMN].rolling(window=LONG_WINDOW).mean()

    # New news files are UTF-8; the original 2025 file is in Windows encoding (curly quotes)
    try:
        news = pd.read_csv(NEWS_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        news = pd.read_csv(NEWS_FILE, encoding='cp1252')
    news['News_Date'] = pd.to_datetime(news['Date'], format='%m/%d/%Y')
    news = news.drop(columns='Date').sort_values('News_Date').reset_index(drop=True)

    # News published on a weekend or holiday is traded on the next trading day
    trading_days = stock[['Date']].rename(columns={'Date': 'Trade_Date'})
    news = pd.merge_asof(news, trading_days, left_on='News_Date', right_on='Trade_Date', direction='forward')

    # The stock data must cover the news period (and start ~1 month earlier for the 20-day average)
    first_ready = stock.loc[stock['SMA_Long'].notna(), 'Date'].min()
    usable = news['Trade_Date'].notna() & (news['News_Date'] >= first_ready)
    if not usable.any():
        raise SystemExit(
            f"\nThe news and the stock data do not cover the same dates.\n"
            f"  Stock data: {stock['Date'].min():%Y-%m-%d} to {stock['Date'].max():%Y-%m-%d}\n"
            f"  News:       {news['News_Date'].min():%Y-%m-%d} to {news['News_Date'].max():%Y-%m-%d}\n"
            f"Update apple_stock_test.csv so it starts about 1 month before the first news and ends after the last news.")
    if not usable.all():
        print(f"Note: {(~usable).sum()} news article(s) skipped because they fall outside the stock data "
              f"(or in its first 20 trading days, before the 20-day average exists).")
    news = news[usable].reset_index(drop=True)
    return stock, news


def build_user_prompt(day, article, cash, shares, buy_price):
    if shares > 0:
        position = f"Yes, {shares} shares (we paid ${buy_price:,.2f} per share)"
    else:
        position = "No"
    return USER_PROMPT_TEMPLATE.format(
        date=day['Date'].strftime('%Y-%m-%d (%A)'),
        price=day[PRICE_COLUMN],
        sma_short=day['SMA_Short'],
        sma_long=day['SMA_Long'],
        trend="UP (5-day average is above the 20-day average)" if day['SMA_Short'] > day['SMA_Long']
              else "DOWN (5-day average is below the 20-day average)",
        position=position,
        cash=cash,
        headline=article['Headline'],
        summary=article['Summary'],
    )


def ask_ai(system_prompt, user_prompt, temperature=0.0):
    # Temperature 0: always the most likely words, so the same prompt gives the same answers.
    # Above 0 there is no fixed seed, so each run can turn out differently (that is the point).
    options = {'temperature': temperature}
    if temperature == 0:
        options['seed'] = 42
    # Very rarely the model sends back an empty or cut-off answer: ask once more,
    # and if it still cannot be read, HOLD instead of stopping the whole run.
    raw = ''
    for attempt in range(2):
        response = ollama.chat(
            model=MODEL,
            messages=[{'role': 'system', 'content': system_prompt},
                      {'role': 'user', 'content': user_prompt}],
            format=ANSWER_SCHEMA,
            options=options,
        )
        raw = response['message']['content']
        try:
            answer = json.loads(raw)
            if isinstance(answer, dict):
                return answer, raw
        except json.JSONDecodeError:
            pass
    unreadable = {'action': 'HOLD', 'sentiment': 'NEUTRAL',
                  'reason': "(The AI's answer could not be read, so we HOLD.)"}
    return unreadable, raw or '(empty answer)'


def friendly_error(error):
    text = str(error)
    if 'not found' in text and 'model' in text:
        return f"The AI model is not installed. Open a terminal and run:\n    ollama pull {MODEL}"
    if 'allocate' in text or 'memory' in text.lower():
        return ("The computer does not have enough free memory to load the AI model.\n"
                "Close other programs (browser tabs, other dashboards) and click Run again.")
    if 'connect' in text.lower() or 'refused' in text.lower():
        return "Cannot reach Ollama. Start the Ollama app (or run 'ollama serve' in a terminal) and click Run again."
    return f"{type(error).__name__}: {text}"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simulation(system_prompt, report_progress, temperature=0.0):
    stock, news = load_data()
    cash, shares, buy_price, buy_date = float(INITIAL_CASH), 0, None, None
    decisions, trades = [], []

    for i, article in news.iterrows():
        day = stock.loc[stock['Date'] == article['Trade_Date']].iloc[0]
        price = float(day[PRICE_COLUMN])
        date_str = day['Date'].strftime('%Y-%m-%d')
        report_progress(i, len(news), article['Headline'])

        user_prompt = build_user_prompt(day, article, cash, shares, buy_price)
        answer, raw_answer = ask_ai(system_prompt, user_prompt, temperature)
        action = str(answer.get('action', 'HOLD')).upper()
        if action not in ('BUY', 'SELL', 'HOLD'):
            action = 'HOLD'
        sentiment = str(answer.get('sentiment', 'NEUTRAL')).upper()
        reason = str(answer.get('reason', ''))

        # Carry out the AI's action (only if it is possible)
        if action == 'BUY':
            if shares > 0:
                result = "No trade: we already own shares"
            elif int(cash // price) == 0:
                result = "No trade: not enough cash"
            else:
                shares = int(cash // price)
                cash -= shares * price
                buy_price, buy_date = price, date_str
                result = f"BOUGHT {shares} shares at ${price:,.2f}"
                trades.append({'date': day['Date'], 'type': 'BUY', 'price': price, 'shares': shares,
                               'reason': reason, 'headline': article['Headline']})
        elif action == 'SELL':
            if shares == 0:
                result = "No trade: we own no shares to sell"
            else:
                profit = (price - buy_price) * shares
                result = f"SOLD {shares} shares at ${price:,.2f}  ->  {money(profit, signed=True)} on this trade"
                trades.append({'date': day['Date'], 'type': 'SELL', 'price': price, 'shares': shares,
                               'reason': reason, 'headline': article['Headline'], 'profit': profit,
                               'bought': f"{buy_date} at ${buy_price:,.2f}"})
                cash += shares * price
                shares, buy_price = 0, None
        else:
            result = "No trade: HOLD"

        decisions.append({'date': day['Date'], 'news_date': article['News_Date'], 'price': price,
                          'headline': article['Headline'], 'action': action, 'sentiment': sentiment,
                          'reason': reason, 'result': result,
                          'user_prompt': user_prompt, 'raw_answer': raw_answer})
        print(f"[{i + 1}/{len(news)}] {date_str}  AI: {action:<4} {sentiment:<8}  {result}")

    # Shares still owned on the last day are sold, so every run ends in cash
    last = stock.iloc[-1]
    last_price = float(last[PRICE_COLUMN])
    if shares > 0:
        profit = (last_price - buy_price) * shares
        trades.append({'date': last['Date'], 'type': 'SELL', 'price': last_price, 'shares': shares,
                       'reason': 'End of simulation: remaining shares are sold', 'headline': '',
                       'profit': profit, 'bought': f"{buy_date} at ${buy_price:,.2f}"})
        cash += shares * last_price

    # Comparison: buy on the first news day and simply keep the shares
    first_price = decisions[0]['price']
    hold_shares = int(INITIAL_CASH // first_price)
    hold_final = INITIAL_CASH - hold_shares * first_price + hold_shares * last_price

    return {
        'stock': stock,
        'decisions': decisions,
        'trades': trades,
        'final': cash,
        'profit': cash - INITIAL_CASH,
        'profit_pct': (cash - INITIAL_CASH) / INITIAL_CASH * 100,
        'round_trips': sum(1 for t in trades if t['type'] == 'SELL'),
        'hold_profit_pct': (hold_final - INITIAL_CASH) / INITIAL_CASH * 100,
        'temperature': temperature,
    }


def money(value, signed=False):
    if signed:
        return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"
    return f"${value:,.2f}"


# ---------------------------------------------------------------------------
# A run takes a few minutes, so it runs in the background and the page
# checks its progress every second.
# ---------------------------------------------------------------------------
state = {'running': False, 'step': 0, 'total': 0, 'headline': '', 'result': None, 'error': None, 'version': 0}
run_history = []
state_lock = threading.Lock()


def background_run(system_prompt, temperature):
    def report_progress(step, total, headline):
        with state_lock:
            state.update(step=step, total=total, headline=headline)

    try:
        result = run_simulation(system_prompt, report_progress, temperature)
        run_history.append({'prompt': system_prompt, 'profit_pct': result['profit_pct'],
                            'round_trips': result['round_trips'], 'temperature': temperature})
        error = None
    except Exception as e:
        traceback.print_exc()
        result, error = None, friendly_error(e)
    with state_lock:
        state.update(running=False, result=result, error=error, version=state['version'] + 1)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
BOX = {'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}
TEXT_BOX = {'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #ddd',
            'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap', 'fontSize': '13px'}


def example_user_prompt():
    stock, news = load_data()
    article = news.iloc[0]
    day = stock.loc[stock['Date'] == article['Trade_Date']].iloc[0]
    return build_user_prompt(day, article, INITIAL_CASH, 0, None)


app = dash.Dash(__name__)
app.title = "AI Prompt Trading Simulator"

app.layout = html.Div([
    html.H1("AI Prompt Engineering with Stock Trading Simulator", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '5px'}),
    html.P(f"Write a strategy for the AI. It reads Apple news from 2025 and trades with ${INITIAL_CASH:,}. "
           f"How much profit can your prompt make?",
           style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0'}),

    # 1. System prompt (students edit this)
    html.Div([
        html.H3("✏️ 1. System prompt: your trading strategy", style={'color': '#2c3e50', 'marginTop': '0'}),
        html.P("Tell the AI who it is, what its goal is and when to BUY, SELL or HOLD. "
               "This text is the same for every news article.",
               style={'color': '#555', 'fontSize': '14px'}),
        dcc.Textarea(id='system-prompt', value=DEFAULT_SYSTEM_PROMPT,
                     style={'width': '100%', 'height': '200px', 'fontFamily': 'monospace', 'fontSize': '14px',
                            'padding': '10px', 'borderRadius': '5px', 'border': '2px solid #8e44ad',
                            'boxSizing': 'border-box'}),
        html.Div([
            html.Button('▶ Run simulation', id='run-btn', n_clicks=0,
                        style={'padding': '10px 30px', 'fontSize': '16px', 'backgroundColor': '#27ae60', 'color': 'white',
                               'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'marginRight': '10px'}),
            html.Button('↺ Reset to the example strategy', id='reset-btn', n_clicks=0,
                        style={'padding': '10px 20px', 'fontSize': '14px', 'backgroundColor': '#95a5a6', 'color': 'white',
                               'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'marginRight': '10px'}),
            html.Button(id='temp-toggle', n_clicks=0,
                        style={'padding': '10px 20px', 'fontSize': '14px', 'color': 'white',
                               'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'marginRight': '15px'}),
            html.Div([
                dcc.Slider(id='temperature', min=0, max=1, step=0.1, value=1.0,
                           marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'}),
            ], id='temp-slider-box', style={'width': '320px'}),
        ], style={'marginTop': '10px', 'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '5px'}),
        html.P(id='temp-help', style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '8px 0 0 0'}),
        html.Div(id='progress', style={'marginTop': '15px'}),
    ], style=BOX),

    # 2. User prompt and answer form (fixed)
    html.Div([
        html.H3("🔒 2. User prompt: what the AI gets for each news article", style={'color': '#2c3e50', 'marginTop': '0'}),
        html.P("The program writes this for you. The numbers and the news change for every article. "
               "Example for the first news article:", style={'color': '#555', 'fontSize': '14px'}),
        html.Div(example_user_prompt(), style=dict(TEXT_BOX, color='#555')),
        html.H4("🔒 The AI must answer on this form", style={'color': '#2c3e50', 'marginBottom': '5px'}),
        html.Div('{"reason": "the AI explains its decision in its own words",\n "sentiment": "POSITIVE" or "NEGATIVE" or "NEUTRAL",\n'
                 ' "action": "BUY" or "SELL" or "HOLD"}', style=dict(TEXT_BOX, color='#555')),
        html.P("Ollama makes sure the answer always has exactly this shape, so the program can read the action. "
               "BUY spends all our cash on shares. SELL sells all our shares. HOLD does nothing. "
               "Shares still owned on the last day are sold automatically.",
               style={'color': '#7f8c8d', 'fontSize': '12px', 'marginBottom': '0'}),
    ], style=BOX),

    # 3. Results (filled in after a run)
    html.Div(id='results'),

    # Earlier runs, to compare prompts
    html.Div(id='history'),

    dcc.Interval(id='poll', interval=1000),
    dcc.Store(id='shown-version'),
    dcc.Store(id='run-request'),
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px', 'fontFamily': 'Arial, sans-serif',
          'backgroundColor': 'white'})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(Output('system-prompt', 'value'), Input('reset-btn', 'n_clicks'), prevent_initial_call=True)
def reset_prompt(n_clicks):
    return DEFAULT_SYSTEM_PROMPT


def chosen_temperature(toggle_clicks, slider_value):
    """Temperature OFF (even clicks) = 0, the AI gives the same answer every time"""
    if toggle_clicks % 2 == 0 or slider_value is None:
        return 0.0
    return min(max(float(slider_value), 0.0), 1.0)


@app.callback(
    [Output('temp-toggle', 'children'), Output('temp-toggle', 'style'),
     Output('temp-slider-box', 'style'), Output('temp-help', 'children')],
    Input('temp-toggle', 'n_clicks'), Input('temperature', 'value'),
    State('temp-toggle', 'style'),
)
def show_temperature(toggle_clicks, slider_value, button_style):
    temperature = chosen_temperature(toggle_clicks, slider_value)
    if toggle_clicks % 2 == 1:
        help_text = (f"🌡️ Temperature {temperature:.1f}: the AI picks its words with some randomness, "
                     f"so running the same prompt twice can give different trades. "
                     f"Higher = more random. Click the button again to switch it off.")
        return ("🌡️ Temperature: ON", dict(button_style, backgroundColor='#e67e22'),
                {'width': '320px'}, help_text)
    help_text = ("🌡️ Temperature is OFF (0): the AI always gives the same answer to the same prompt, "
                 "so only your prompt changes the result. Click the button to try a higher temperature.")
    return "🌡️ Temperature: OFF (0)", dict(button_style, backgroundColor='#34495e'), {'display': 'none'}, help_text


@app.callback(Output('run-request', 'data'),
              Input('run-btn', 'n_clicks'),
              State('system-prompt', 'value'), State('temp-toggle', 'n_clicks'), State('temperature', 'value'),
              prevent_initial_call=True)
def start_run(n_clicks, system_prompt, toggle_clicks, slider_value):
    temperature = chosen_temperature(toggle_clicks, slider_value)
    with state_lock:
        if state['running']:
            return no_update
        state.update(running=True, step=0, total=0, headline='', error=None)
    threading.Thread(target=background_run, args=(system_prompt or '', temperature), daemon=True).start()
    return no_update


@app.callback(
    [Output('progress', 'children'), Output('run-btn', 'disabled'),
     Output('results', 'children'), Output('history', 'children'), Output('shown-version', 'data')],
    Input('poll', 'n_intervals'),
    State('shown-version', 'data'),
)
def update_page(n_intervals, shown_version):
    with state_lock:
        snapshot = dict(state)

    if snapshot['running']:
        total = snapshot['total'] or 1
        done = snapshot['step']
        progress = html.Div([
            html.P(f"🤖 Asking the AI about news {done + 1} of {snapshot['total'] or '...'}: "
                   f"\"{snapshot['headline']}\"", style={'margin': '0 0 6px 0', 'color': '#2c3e50'}),
            html.Div(html.Div(style={'width': f"{done / total * 100:.0f}%", 'height': '100%',
                                     'backgroundColor': '#27ae60', 'borderRadius': '5px'}),
                     style={'height': '12px', 'backgroundColor': '#d5dbdb', 'borderRadius': '5px'}),
            html.P("Each article takes a few seconds. You can scroll down to look at the last results while you wait.",
                   style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '6px 0 0 0'}),
        ])
        return progress, True, no_update, no_update, no_update

    if snapshot['version'] == shown_version:
        return no_update, False, no_update, no_update, no_update

    if snapshot['error']:
        progress = html.Div([html.Strong("❌ The simulation stopped: "), html.Pre(snapshot['error'], style={'whiteSpace': 'pre-wrap', 'margin': '6px 0 0 0'})],
                            style={'color': '#721c24', 'backgroundColor': '#f8d7da', 'padding': '12px',
                                   'borderRadius': '5px', 'border': '1px solid #f5c6cb'})
        return progress, False, no_update, history_table(), snapshot['version']

    result = snapshot['result']
    progress = html.P("✅ Done! Scroll down to see the results." if result else "",
                      style={'color': '#27ae60', 'fontWeight': 'bold', 'margin': '0'})
    return progress, False, results_section(result) if result else "", history_table(), snapshot['version']


# ---------------------------------------------------------------------------
# Result sections
# ---------------------------------------------------------------------------
def results_section(r):
    return html.Div([
        html.Div([html.H3("💰 3. Result", style={'color': '#2c3e50', 'marginTop': '0'}), summary_cards(r)], style=BOX),
        html.Div([
            html.H3("📈 4. Apple stock graph with the AI's trades", style={'color': '#2c3e50', 'marginTop': '0'}),
            html.P("▲ BUY  |  ▼ SELL  |  ◆ News (green = AI says positive, red = negative, grey = neutral)  |  "
                   "Hover a marker to see the AI's reason", style={'color': '#7f8c8d', 'fontSize': '12px'}),
            dcc.Graph(figure=price_figure(r), style={'height': '550px'}),
        ], style=BOX),
        html.Div([
            html.H3("📰 5. What the AI decided for each news article", style={'color': '#2c3e50', 'marginTop': '0'}),
            decisions_table(r['decisions']),
            html.Details([
                html.Summary("🔍 Show the exact prompts sent and the AI's raw answers",
                             style={'cursor': 'pointer', 'fontWeight': 'bold', 'marginTop': '15px'}),
                html.Div(prompts_log(r['decisions']), style=dict(TEXT_BOX, fontSize='12px', marginTop='10px')),
            ]),
        ], style=BOX),
    ])


def summary_cards(r):
    def card(title, value, color):
        return html.Div([html.Div(title, style={'fontSize': '13px', 'color': '#7f8c8d'}),
                         html.Div(value, style={'fontSize': '26px', 'fontWeight': 'bold', 'color': color})],
                        style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px',
                               'flex': '1', 'minWidth': '180px', 'textAlign': 'center'})

    profit_color = '#27ae60' if r['profit'] >= 0 else '#e74c3c'
    if r['profit_pct'] > r['hold_profit_pct']:
        verdict = "🎉 Your AI strategy did better than simply buying on the first news day and keeping the shares."
    else:
        verdict = "Simply buying on the first news day and keeping the shares did better. Can your prompt beat it?"
    return html.Div([
        html.Div([
            card("Start money", money(INITIAL_CASH), '#2c3e50'),
            card("Final money", money(r['final']), profit_color),
            card("Profit", f"{money(r['profit'], signed=True)} ({r['profit_pct']:+.2f}%)", profit_color),
            card("Trades (buy → sell)", str(r['round_trips']), '#2c3e50'),
        ], style={'display': 'flex', 'gap': '15px', 'flexWrap': 'wrap'}),
        html.P(f"{verdict}  (Buy and keep: {r['hold_profit_pct']:+.2f}%)   🌡️ Temperature used: {r['temperature']:.1f}",
               style={'marginTop': '15px', 'fontWeight': 'bold', 'color': '#8e44ad'}),
    ])


def wrap(text, width=70):
    words, lines, line = str(text).split(), [], ""
    for word in words:
        if line and len(line) + len(word) > width:
            lines.append(line)
            line = ""
        line += word + " "
    lines.append(line)
    return "<br>".join(lines)


def price_figure(r):
    stock, decisions, trades = r['stock'], r['decisions'], r['trades']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock[PRICE_COLUMN], mode='lines', name='Apple price',
                             line=dict(color='blue', width=2), opacity=0.7))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['SMA_Short'], mode='lines', name='5-day average',
                             line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['SMA_Long'], mode='lines', name='20-day average',
                             line=dict(color='red', width=1.5)))

    # News markers, a little above the price so they do not hide the trades
    colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
    fig.add_trace(go.Scatter(
        x=[d['date'] for d in decisions], y=[d['price'] * 1.04 for d in decisions], mode='markers', name='News',
        marker=dict(symbol='diamond', size=11, color=[colors.get(d['sentiment'], 'gray') for d in decisions],
                    line=dict(color='white', width=1)),
        text=[f"<b>News:</b> {wrap(d['headline'])}<br><b>AI:</b> {d['action']} ({d['sentiment']})<br>"
              f"<b>Reason:</b> {wrap(d['reason'])}<br><b>Result:</b> {d['result']}" for d in decisions],
        hovertemplate='%{x|%Y-%m-%d}<br>%{text}<extra></extra>'))

    for trade_type, symbol, color in [('BUY', 'triangle-up', 'green'), ('SELL', 'triangle-down', 'red')]:
        group = [t for t in trades if t['type'] == trade_type]
        if not group:
            continue
        fig.add_trace(go.Scatter(
            x=[t['date'] for t in group], y=[t['price'] for t in group], mode='markers', name=trade_type,
            marker=dict(symbol=symbol, size=16, color=color, line=dict(color='black', width=1)),
            text=[f"{t['shares']} shares<br><b>Reason:</b> {wrap(t['reason'])}" +
                  (f"<br><b>Bought:</b> {t['bought']}<br><b>Trade profit:</b> {money(t['profit'], signed=True)}"
                   if trade_type == 'SELL' else "") for t in group],
            hovertemplate=f'<b>{trade_type}</b> on %{{x|%Y-%m-%d}} at $%{{y:.2f}}<br>%{{text}}<extra></extra>'))

    fig.update_layout(xaxis_title='Date', yaxis_title='Price ($)', hovermode='closest', template='plotly_white',
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'), margin=dict(t=20))
    return fig


def decisions_table(decisions):
    cell = {'padding': '8px', 'borderBottom': '1px solid #ddd', 'verticalAlign': 'top', 'fontSize': '13px'}
    action_colors = {'BUY': '#27ae60', 'SELL': '#e74c3c', 'HOLD': '#7f8c8d'}
    sentiment_colors = {'POSITIVE': '#27ae60', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#7f8c8d'}   # same as the news diamonds
    # Columns follow the order the AI fills in its answer: reason -> sentiment -> action
    header = html.Tr([html.Th(h, style=dict(cell, textAlign='left', borderBottom='2px solid #bbb'))
                      for h in ['#', 'Trade day', 'News headline', "AI's reason", 'Sentiment', 'Action', 'What happened']])
    rows = [html.Tr([
        html.Td(str(n), style=cell),
        html.Td(d['date'].strftime('%Y-%m-%d'), style=dict(cell, whiteSpace='nowrap')),
        html.Td(d['headline'], style=cell),
        html.Td(d['reason'], style=cell),
        html.Td(d['sentiment'], style=dict(cell, color=sentiment_colors.get(d['sentiment'], '#7f8c8d'))),
        html.Td(html.Strong(d['action']), style=dict(cell, color=action_colors.get(d['action']))),
        html.Td(d['result'], style=dict(cell, fontWeight='bold' if not d['result'].startswith('No trade') else 'normal')),
    ]) for n, d in enumerate(decisions, start=1)]
    return html.Table([header] + rows, style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': 'white'})


def prompts_log(decisions):
    parts = []
    for n, d in enumerate(decisions, start=1):
        parts.append(f"===== NEWS {n} of {len(decisions)} =====\n--- USER PROMPT ---\n{d['user_prompt']}\n"
                     f"--- AI ANSWER ---\n{d['raw_answer']}\n--- WHAT HAPPENED ---\n{d['result']}\n")
    return "\n".join(parts)


def history_table():
    if not run_history:
        return ""
    cell = {'padding': '8px', 'borderBottom': '1px solid #ddd', 'verticalAlign': 'top', 'fontSize': '13px'}
    best = max(h['profit_pct'] for h in run_history)
    header = html.Tr([html.Th(h, style=dict(cell, textAlign='left', borderBottom='2px solid #bbb'))
                      for h in ['Run', 'System prompt', 'Temperature', 'Profit', 'Trades']])
    rows = []
    for n, h in reversed(list(enumerate(run_history, start=1))):
        short = h['prompt'].replace('\n', ' ')
        short = short if len(short) <= 120 else short[:117] + '...'
        rows.append(html.Tr([
            html.Td(f"{n}{' ⭐' if h['profit_pct'] == best else ''}", style=cell),
            html.Td(short, title=h['prompt'], style=dict(cell, fontFamily='monospace')),
            html.Td(f"{h['temperature']:.1f}", style=cell),
            html.Td(f"{h['profit_pct']:+.2f}%", style=dict(cell, color='#27ae60' if h['profit_pct'] >= 0 else '#e74c3c',
                                                         fontWeight='bold')),
            html.Td(str(h['round_trips']), style=cell),
        ]))
    return html.Div([
        html.H3("🏆 Your runs (newest first)", style={'color': '#2c3e50', 'marginTop': '0'}),
        html.P("Change one thing in your prompt at a time, then compare. Hover a prompt to read all of it.",
               style={'color': '#7f8c8d', 'fontSize': '12px'}),
        html.Table([header] + rows, style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': 'white'}),
    ], style=BOX)


if __name__ == '__main__':
    print("Open http://127.0.0.1:8050 in your web browser.")
    app.run(debug=False)
