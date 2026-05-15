import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import sqlite3

positive_words = ["growth", "gain", "beat", "strong", "record", "surge", "profit", "upgrade", "positive", "bullish"]
negative_words = ["loss", "drop", "fall", "weak", "miss", "risk", "lawsuit", "cut", "negative", "bearish"]


def init_database():
    conn = sqlite3.connect("stock_agent.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            symbol TEXT,
            quantity REAL,
            buy_price REAL
        )
        """
    )

    conn.commit()
    conn.close()


def add_watchlist_symbol(symbol):
    conn = sqlite3.connect("stock_agent.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)", (symbol.upper(),))
    conn.commit()
    conn.close()


def get_watchlist_symbols():
    conn = sqlite3.connect("stock_agent.db")
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM watchlist")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def clear_watchlist():
    conn = sqlite3.connect("stock_agent.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM watchlist")
    conn.commit()
    conn.close()


def add_portfolio_position(symbol, quantity, buy_price):
    conn = sqlite3.connect("stock_agent.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO portfolio (symbol, quantity, buy_price) VALUES (?, ?, ?)",
        (symbol.upper(), quantity, buy_price)
    )
    conn.commit()
    conn.close()


def get_portfolio_positions():
    conn = sqlite3.connect("stock_agent.db")
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, quantity, buy_price FROM portfolio")
    rows = cursor.fetchall()
    conn.close()
    return rows


def clear_portfolio():
    conn = sqlite3.connect("stock_agent.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM portfolio")
    conn.commit()
    conn.close()


def analyze_sentiment(title):
    title_lower = title.lower()
    positive_score = sum(1 for word in positive_words if word in title_lower)
    negative_score = sum(1 for word in negative_words if word in title_lower)

    if positive_score > negative_score:
        return "Positive 🙂"
    elif negative_score > positive_score:
        return "Negative 🙁"
    else:
        return "Neutral 😐"


def get_news_title_and_link(item):
    content = item.get("content", {})

    title = item.get("title") or content.get("title") or "No title available"
    link = item.get("link") or content.get("canonicalUrl", {}).get("url", "")

    return title, link


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_ai_summary(stock, company_name, current_price, price_change_percent, volatility, latest_rsi, risk, recommendation, news_summary):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")

        if not api_key:
            return "OpenAI API key not found. Add it inside .streamlit/secrets.toml to enable the AI summary."

        client = OpenAI(api_key=api_key)

        prompt = f"""
        You are a modern AI stock research assistant for beginner investors.
        Your job is to create a clean, useful, and easy-to-read stock summary.

        Stock Symbol: {stock}
        Company: {company_name}
        Current Price: {current_price}
        6 Month Price Change: {price_change_percent:.2f}%
        Volatility: {volatility:.2f}%
        RSI: {latest_rsi:.2f}
        Risk Level: {risk}
        System Recommendation: {recommendation}
        Recent News Headlines: {news_summary}

        Write the response in this exact format using markdown:

        ### Quick Verdict
        Give a clear 2-3 sentence summary of what is happening with the stock right now.

        ### Price Trend
        Explain the 6-month trend in simple language. Mention whether momentum looks strong, weak, or mixed.

        ### Risk Check
        Explain the risk level using volatility and RSI. Keep it beginner-friendly.

        ### News Impact
        Explain what the recent news headlines suggest about investor sentiment. Do not overclaim.

        ### What To Watch Next
        Give 3 short bullet points about what a beginner should watch before making any decision.

        ### Final Note
        End with one educational reminder that this is not financial advice and users should research before investing.

        Style rules:
        - Sound professional but simple.
        - Avoid robotic language.
        - Do not promise profits.
        - Do not say BUY is guaranteed.
        - Keep it under 220 words.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI finance education assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=550,
            temperature=0.4
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI summary could not be generated: {e}"


def ask_finance_chatbot(question, stock_context):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")

        if not api_key:
            return "OpenAI API key not found. Add it inside .streamlit/secrets.toml to enable the chatbot."

        client = OpenAI(api_key=api_key)

        prompt = f"""
        You are an AI finance education chatbot. Use the stock context below to answer the user's question.

        Stock Context:
        {stock_context}

        User Question:
        {question}

        Answer in simple beginner-friendly language.
        Do not promise profits.
        Do not give guaranteed financial advice.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI finance education assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.4
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Chatbot response could not be generated: {e}"



st.set_page_config(page_title="AI Stock Analysis Agent", page_icon="📈", layout="wide")
init_database()
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #111827;
    }

    h1, h2, h3, h4 {
        color: #111827;
    }

    .ai-summary-box, .chatbot-box {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #E5E7EB;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    }

    .summary-title {
        font-size: 28px;
        font-weight: 700;
        color: #047857;
    }

    .robot-title {
        font-size: 28px;
        font-weight: 700;
        color: #2563EB;
    }

    .summary-helper-text {
        color: #4B5563;
        font-size: 15px;
        margin-top: 8px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("📊 AI Stock Agent")
st.sidebar.markdown("Built by **Gopichand Katta**")
st.sidebar.divider()
st.sidebar.markdown("### Features")
st.sidebar.markdown(
    """
    - Live stock data
    - Moving averages
    - RSI analysis
    - News sentiment
    - AI stock summary
    - Multi-stock comparison
    - AI finance chatbot
    """
)
st.sidebar.divider()
st.sidebar.caption("Educational project — not financial advice.")

st.title("📈 AI Stock Analysis Agent")
st.markdown("### Built by **Gopichand Katta**")
st.info(
    "Enter a stock symbol to analyze live market data, technical indicators, news sentiment, and AI-generated insights."
)

stock = st.text_input(
    "Enter Stock Symbol",
    "AAPL",
    help="Example: AAPL, TSLA, NVDA, MSFT"
)

if stock:
    stock = stock.upper()

    st.header(f"Stock Analysis for {stock}")

    ticker = yf.Ticker(stock)

    try:
        with st.spinner("Fetching latest stock data..."):
            data = ticker.history(period="6mo")

        with st.spinner("Loading company information..."):
            info = ticker.info

        if data.empty:
            st.error("No stock data found. Please check the stock symbol.")

        else:
            company_name = info.get("longName", stock)
            current_price = info.get("currentPrice", "Not available")
            market_cap = info.get("marketCap", "Not available")
            volume = info.get("volume", "Not available")

            st.success(f"Company: {company_name}")

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Current Price", current_price)
            metric_col2.metric("Market Cap", market_cap)
            metric_col3.metric("Volume", volume)

            st.divider()

            data["MA20"] = data["Close"].rolling(window=20).mean()
            data["MA50"] = data["Close"].rolling(window=50).mean()
            data["RSI"] = calculate_rsi(data["Close"])

            st.subheader("6 Month Stock Price Chart with Moving Averages")

            chart_data = data.reset_index()

            fig = px.line(
                chart_data,
                x="Date",
                y=["Close", "MA20", "MA50"],
                title=f"{stock} Closing Price with 20-Day and 50-Day Moving Averages"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Candlestick Price Chart")

            candlestick_fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=chart_data["Date"],
                        open=chart_data["Open"],
                        high=chart_data["High"],
                        low=chart_data["Low"],
                        close=chart_data["Close"],
                        name="Price"
                    )
                ]
            )

            candlestick_fig.update_layout(
                title=f"{stock} Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=500
            )

            st.plotly_chart(candlestick_fig, use_container_width=True)

            st.subheader("Technical Indicator: RSI")

            latest_rsi = data["RSI"].dropna().iloc[-1]

            if latest_rsi > 70:
                rsi_signal = "Overbought ⚠️"
                rsi_explanation = "The stock may be overbought, meaning the price has risen quickly and could pull back."
            elif latest_rsi < 30:
                rsi_signal = "Oversold 🟢"
                rsi_explanation = "The stock may be oversold, meaning the price has fallen quickly and could recover."
            else:
                rsi_signal = "Neutral 😐"
                rsi_explanation = "The stock is not showing extreme buying or selling pressure."

            col_rsi1, col_rsi2 = st.columns(2)
            col_rsi1.metric("RSI", f"{latest_rsi:.2f}")
            col_rsi2.metric("RSI Signal", rsi_signal)

            st.info(rsi_explanation)

            with st.expander("View Recent Stock Data"):
                st.dataframe(data.tail(10))

            st.subheader("AI Stock Insight")

            latest_price = data["Close"].iloc[-1]
            old_price = data["Close"].iloc[0]

            price_change_percent = ((latest_price - old_price) / old_price) * 100
            volatility = data["Close"].pct_change().std() * 100

            if price_change_percent > 10 and volatility < 3 and latest_rsi < 70:
                trend = "strong upward 📈"
                risk = "Medium"
                recommendation = "BUY ✅"
                explanation = "The stock has strong positive momentum, controlled volatility, and RSI is not overbought."

            elif price_change_percent > 0:
                trend = "upward 📈"
                risk = "Medium"
                recommendation = "HOLD ⚠️"
                explanation = "The stock is moving upward, but the growth or technical signals suggest waiting before buying aggressively."

            else:
                trend = "downward 📉"
                risk = "High"
                recommendation = "AVOID ❌"
                explanation = "The stock has negative momentum over the selected time period."

            st.write(f"The stock shows a {trend} trend over the last 6 months.")

            col4, col5, col6 = st.columns(3)
            col4.metric("6 Month Change", f"{price_change_percent:.2f}%")
            col5.metric("Volatility", f"{volatility:.2f}%")
            col6.metric("Recommendation", recommendation)

            st.write(f"Estimated Risk Level: {risk}")
            st.info(explanation)

            st.subheader("Latest News Sentiment")

            news_items = ticker.news
            news_titles = []

            if not news_items:
                st.warning("No recent news found for this stock.")
            else:
                for item in news_items[:5]:
                    title, link = get_news_title_and_link(item)
                    sentiment = analyze_sentiment(title)
                    news_titles.append(title)

                    st.write(f"**{title}**")
                    st.write(f"Sentiment: {sentiment}")

                    if link:
                        st.write(f"[Read more]({link})")

                    st.divider()

            news_summary = " | ".join(news_titles[:5]) if news_titles else "No recent news available."

            st.markdown('<div class="ai-summary-box">', unsafe_allow_html=True)
            st.markdown('<div class="summary-title">🤖 AI Stock Research Summary</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="summary-helper-text">Get a clean beginner-friendly breakdown of trend, risk, news impact, and what to watch next.</div>',
                unsafe_allow_html=True
            )

            if st.button("Generate Better AI Summary", type="primary"):
                with st.spinner("Creating a smarter AI stock summary..."):
                    ai_summary = generate_ai_summary(
                        stock,
                        company_name,
                        current_price,
                        price_change_percent,
                        volatility,
                        latest_rsi,
                        risk,
                        recommendation,
                        news_summary
                    )
                    st.markdown(ai_summary)

            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("Multi-Stock Comparison")
            st.write("Compare multiple stocks based on normalized 6-month performance.")

            comparison_input = st.text_input(
                "Enter stock symbols separated by commas",
                "AAPL, TSLA, NVDA, MSFT"
            )

            if comparison_input:
                symbols = [symbol.strip().upper() for symbol in comparison_input.split(",")]
                comparison_data = pd.DataFrame()

                for symbol in symbols:
                    comparison_ticker = yf.Ticker(symbol)
                    comparison_history = comparison_ticker.history(period="6mo")

                    if not comparison_history.empty:
                        normalized_price = (
                            comparison_history["Close"] / comparison_history["Close"].iloc[0]
                        ) * 100
                        comparison_data[symbol] = normalized_price

                if comparison_data.empty:
                    st.warning("No comparison data found. Please check the symbols.")
                else:
                    comparison_data = comparison_data.reset_index()

                    comparison_fig = px.line(
                        comparison_data,
                        x="Date",
                        y=symbols,
                        title="Normalized Stock Performance Comparison"
                    )

                    st.plotly_chart(comparison_fig, use_container_width=True)

            st.markdown('<div class="chatbot-box">', unsafe_allow_html=True)
            st.markdown('<div class="robot-title">🤖 AI Finance Assistant</div>', unsafe_allow_html=True)
            st.write("Ask questions about the selected stock, risk, trend, RSI, or recent news.")

            stock_context = f"""
            Stock Symbol: {stock}
            Company: {company_name}
            Current Price: {current_price}
            6 Month Change: {price_change_percent:.2f}%
            Volatility: {volatility:.2f}%
            RSI: {latest_rsi:.2f}
            RSI Signal: {rsi_signal}
            Risk Level: {risk}
            Recommendation: {recommendation}
            Recent News: {news_summary}
            """

            user_question = st.text_input(
                "Ask the AI Finance Chatbot",
                "Why is this stock risky or safe?"
            )

            if st.button("Ask Chatbot", type="primary"):
                with st.spinner("Thinking..."):
                    chatbot_answer = ask_finance_chatbot(user_question, stock_context)
                    st.info(chatbot_answer)

            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("⭐ Stock Watchlist")
            st.write("Save favorite stocks and quickly review current price movement.")

            new_watch_symbol = st.text_input("Add stock to watchlist", "AAPL")

            watch_col1, watch_col2 = st.columns(2)

            with watch_col1:
                if st.button("Add to Watchlist"):
                    add_watchlist_symbol(new_watch_symbol)
                    st.success(f"{new_watch_symbol.upper()} added to watchlist.")

            with watch_col2:
                if st.button("Clear Watchlist"):
                    clear_watchlist()
                    st.warning("Watchlist cleared.")

            watchlist_symbols = get_watchlist_symbols()

            if watchlist_symbols:
                watchlist_data = []

                for symbol in watchlist_symbols:
                    try:
                        watchlist_ticker = yf.Ticker(symbol)
                        watchlist_info = watchlist_ticker.info
                        watchlist_history = watchlist_ticker.history(period="5d")

                        current_watch_price = watchlist_info.get("currentPrice", 0)
                        company_watch_name = watchlist_info.get("shortName", symbol)

                        if not watchlist_history.empty and len(watchlist_history) >= 2:
                            previous_close = watchlist_history["Close"].iloc[-2]
                            price_change = current_watch_price - previous_close
                            price_change_percent = (price_change / previous_close) * 100
                        else:
                            price_change = 0
                            price_change_percent = 0

                        watchlist_data.append({
                            "Symbol": symbol,
                            "Company": company_watch_name,
                            "Current Price": round(current_watch_price, 2),
                            "Daily Change": round(price_change, 2),
                            "Daily Change %": round(price_change_percent, 2)
                        })

                    except Exception:
                        pass

                if watchlist_data:
                    watchlist_df = pd.DataFrame(watchlist_data)
                    st.dataframe(watchlist_df, use_container_width=True)
                else:
                    st.warning("No watchlist data found. Please check the saved symbols.")
            else:
                st.info("Your watchlist is empty. Add a stock symbol above.")

            st.divider()

            st.subheader("📁 Portfolio Tracker")
            st.write("Track your stock positions, profit/loss, and portfolio allocation.")

            portfolio_input = st.text_area(
                "Enter portfolio data (Symbol, Quantity, Buy Price)",
                "AAPL,10,180\nTSLA,5,220\nNVDA,3,950"
            )

            portfolio_rows = [row.strip() for row in portfolio_input.splitlines() if row.strip()]
            portfolio_data = []

            for row in portfolio_rows:
                try:
                    symbol, quantity, buy_price = row.split(",")

                    symbol = symbol.strip().upper()
                    quantity = float(quantity.strip())
                    buy_price = float(buy_price.strip())

                    portfolio_ticker = yf.Ticker(symbol)
                    portfolio_info = portfolio_ticker.info
                    current_stock_price = portfolio_info.get("currentPrice", 0)

                    invested_value = quantity * buy_price
                    current_value = quantity * current_stock_price
                    profit_loss = current_value - invested_value
                    profit_loss_percent = ((current_value - invested_value) / invested_value) * 100

                    portfolio_data.append({
                        "Symbol": symbol,
                        "Quantity": quantity,
                        "Buy Price": round(buy_price, 2),
                        "Current Price": round(current_stock_price, 2),
                        "Invested": round(invested_value, 2),
                        "Current Value": round(current_value, 2),
                        "P/L": round(profit_loss, 2),
                        "P/L %": round(profit_loss_percent, 2)
                    })

                except Exception:
                    pass

            if portfolio_data:
                portfolio_df = pd.DataFrame(portfolio_data)

                total_invested = portfolio_df["Invested"].sum()
                total_current = portfolio_df["Current Value"].sum()
                total_pl = portfolio_df["P/L"].sum()

                portfolio_metric1, portfolio_metric2, portfolio_metric3 = st.columns(3)
                portfolio_metric1.metric("Total Invested", f"${total_invested:,.2f}")
                portfolio_metric2.metric("Current Portfolio Value", f"${total_current:,.2f}")
                portfolio_metric3.metric("Total Profit/Loss", f"${total_pl:,.2f}")

                st.dataframe(portfolio_df, use_container_width=True)

                allocation_fig = px.pie(
                    portfolio_df,
                    names="Symbol",
                    values="Current Value",
                    title="Portfolio Allocation"
                )

                st.plotly_chart(allocation_fig, use_container_width=True)

            st.caption(
                "Built by Gopichand Katta | This is an educational project, not financial advice."
            )

    except Exception as e:
        st.error("Something went wrong while fetching stock data.")
        st.write(e)