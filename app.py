import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from openai import OpenAI

positive_words = ["growth", "gain", "beat", "strong", "record", "surge", "profit", "upgrade", "positive", "bullish"]
negative_words = ["loss", "drop", "fall", "weak", "miss", "risk", "lawsuit", "cut", "negative", "bearish"]


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
        You are an AI stock analysis assistant. Explain this stock in simple beginner-friendly language.

        Stock Symbol: {stock}
        Company: {company_name}
        Current Price: {current_price}
        6 Month Price Change: {price_change_percent:.2f}%
        Volatility: {volatility:.2f}%
        RSI: {latest_rsi:.2f}
        Risk Level: {risk}
        Recommendation: {recommendation}
        Recent News Summary: {news_summary}

        Give a short analysis with:
        1. Simple stock overview
        2. Trend explanation
        3. Risk explanation
        4. News sentiment explanation
        5. Final educational note

        Do not provide guaranteed financial advice.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI finance education assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.4
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI summary could not be generated: {e}"

st.set_page_config(page_title="AI Stock Analysis Agent", layout="wide")

st.title("📈 AI Stock Analysis Agent")
st.write("Enter a stock symbol to analyze stock market data.")

stock = st.text_input("Enter Stock Symbol", "AAPL")

if stock:
    stock = stock.upper()

    st.subheader(f"Stock Analysis for {stock}")

    ticker = yf.Ticker(stock)

    try:
        data = ticker.history(period="6mo")
        info = ticker.info

        if data.empty:
            st.error("No stock data found. Please check the stock symbol.")

        else:
            company_name = info.get("longName", stock)
            current_price = info.get("currentPrice", "Not available")
            market_cap = info.get("marketCap", "Not available")
            volume = info.get("volume", "Not available")

            st.success(f"Company: {company_name}")

            col1, col2, col3 = st.columns(3)

            col1.metric("Current Price", current_price)
            col2.metric("Market Cap", market_cap)
            col3.metric("Volume", volume)

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

            st.subheader("Recent Stock Data")
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

            st.write(
                f"The stock shows a {trend} trend over the last 6 months."
            )

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

            st.subheader("AI Generated Stock Summary")

            news_summary = " | ".join(news_titles[:5]) if news_titles else "No recent news available."

            if st.button("Generate AI Summary"):
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
                st.write(ai_summary)

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

            st.caption(
                "Note: This is an educational project, not financial advice."
            )


    except Exception as e:
        st.error("Something went wrong while fetching stock data.")
        st.write(e)