import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
import random
import time
from fake_useragent import UserAgent
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from textblob import TextBlob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_news_from_source(url, ua):
    try:
        headers = {
            'User-Agent': ua.random,
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        if "cointelegraph.com" in url:
            articles = soup.find_all('article', class_='post-card__article')[:5]
            return [article.find('span', class_='post-card__title').text.strip() for article in articles if article.find('span', class_='post-card__title')]
        elif "coindesk.com" in url:
            articles = soup.find_all('div', class_='article-cardstyles__AcTitle-sc-q1x8lc-4')[:5]
            return [article.find('h6').text.strip() for article in articles if article.find('h6')]
        elif "cryptonews.com" in url:
            articles = soup.find_all('div', class_='cn-tile article')[:5]
            return [article.find('h4').text.strip() for article in articles if article.find('h4')]
    except Exception as e:
        logging.error(f"Error fetching news from {url}: {str(e)}")
        return []


def fetch_crypto_news():
    news_sources = [
        "https://cointelegraph.com/",
        "https://www.coindesk.com/",
        "https://cryptonews.com/"
    ]
    ua = UserAgent()
    headlines = []
    with ThreadPoolExecutor(max_workers=len(news_sources)) as executor:
        future_to_url = {executor.submit(fetch_news_from_source, url, ua): url for url in news_sources}
        for future in as_completed(future_to_url):
            headlines.extend(future.result())
    time.sleep(random.uniform(1, 3))
    return headlines


def fetch_data(symbol, timeframe, exchange_id='mexc'):
    try:
        exchange = getattr(ccxt, exchange_id)()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        return None


def calculate_indicators(df):
    df['ema_short'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_short'] - df['ema_long']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['bollinger_mid'] = df['close'].rolling(window=20).mean()
    df['bollinger_std'] = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = df['bollinger_mid'] + 2 * df['bollinger_std']
    df['bollinger_lower'] = df['bollinger_mid'] - 2 * df['bollinger_std']
    df['historical_volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    df['chikou_span'] = df['close'].shift(-26)
    return df


def calculate_risk_metrics(df, window=252):
    returns = df['close'].pct_change()
    df['var_95'] = returns.rolling(window=window).quantile(0.05)
    df['es_95'] = returns[returns <= df['var_95']].rolling(window=window).mean()
    df['max_drawdown'] = (df['close'] / df['close'].cummax() - 1).rolling(window=window).min()
    return df


def calculate_performance_metrics(df):
    returns = df['close'].pct_change()
    risk_free_rate = 0.02
    excess_returns = returns - risk_free_rate / 252
    df['sharpe_ratio'] = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    df['sortino_ratio'] = (excess_returns.mean() / excess_returns[excess_returns < 0].std()) * np.sqrt(252)
    df['calmar_ratio'] = (returns.mean() * 252) / abs(df['max_drawdown'].min())
    return df


def sentiment_analysis(headlines):
    sentiments = [TextBlob(headline).sentiment.polarity for headline in headlines]
    return np.mean(sentiments)


def generate_signals(df):
    buy_signals = (df['macd'] > df['signal']) & (df['rsi'] < 30) & (df['close'] > df['sma_50']) & (df['sma_50'] > df['sma_200']) & (df['close'] < df['bollinger_lower'])
    sell_signals = (df['macd'] < df['signal']) & (df['rsi'] > 70) & (df['close'] < df['sma_50']) & (df['sma_50'] < df['sma_200']) & (df['close'] > df['bollinger_upper'])
    return buy_signals, sell_signals


def plot_data(df, symbol):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24), sharex=True)
    ax1.plot(df['timestamp'], df['close'], label='Close Price')
    ax1.plot(df['timestamp'], df['sma_50'], label='50-day SMA')
    ax1.plot(df['timestamp'], df['sma_200'], label='200-day SMA')
    ax1.plot(df['timestamp'], df['bollinger_upper'], label='Bollinger Upper', linestyle='--')
    ax1.plot(df['timestamp'], df['bollinger_lower'], label='Bollinger Lower', linestyle='--')
    ax1.fill_between(df['timestamp'], df['senkou_span_a'], df['senkou_span_b'], where=df['senkou_span_a'] >= df['senkou_span_b'], facecolor='green', alpha=0.1)
    ax1.fill_between(df['timestamp'], df['senkou_span_a'], df['senkou_span_b'], where=df['senkou_span_a'] < df['senkou_span_b'], facecolor='red', alpha=0.1)
    ax1.plot(df['timestamp'], df['tenkan_sen'], label='Tenkan-sen')
    ax1.plot(df['timestamp'], df['kijun_sen'], label='Kijun-sen')
    ax1.set_title(f'{symbol} Price, Moving Averages, Bollinger Bands, and Ichimoku Cloud')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(df['timestamp'], df['macd'], label='MACD')
    ax2.plot(df['timestamp'], df['signal'], label='Signal')
    ax2.set_title('MACD')
    ax2.set_ylabel('MACD')
    ax2.legend()

    ax3.plot(df['timestamp'], df['rsi'], label='RSI')
    ax3.axhline(y=30, color='r', linestyle='--')
    ax3.axhline(y=70, color='r', linestyle='--')
    ax3.set_title('RSI')
    ax3.set_ylabel('RSI')
    ax3.legend()

    ax4.plot(df['timestamp'], df['historical_volatility'], label='Historical Volatility')
    ax4.set_title('Historical Volatility')
    ax4.set_ylabel('Volatility')
    ax4.set_xlabel('Date')
    ax4.legend()

    plt.tight_layout()
    plt.show()


def print_current_data(df, symbol):
    latest_data = df.iloc[-1]
    print(f"\nLatest data for {symbol} ({latest_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}):")
    print(f"Close: {latest_data['close']:.2f}")
    print(f"MACD: {latest_data['macd']:.2f}")
    print(f"Signal: {latest_data['signal']:.2f}")
    print(f"RSI: {latest_data['rsi']:.2f}")
    print(f"50-day SMA: {latest_data['sma_50']:.2f}")
    print(f"200-day SMA: {latest_data['sma_200']:.2f}")
    print(f"Bollinger Upper: {latest_data['bollinger_upper']:.2f}")
    print(f"Bollinger Lower: {latest_data['bollinger_lower']:.2f}")
    print(f"Historical Volatility: {latest_data['historical_volatility']:.2f}")
    print(f"Tenkan-sen: {latest_data['tenkan_sen']:.2f}")
    print(f"Kijun-sen: {latest_data['kijun_sen']:.2f}")
    print(f"Senkou Span A: {latest_data['senkou_span_a']:.2f}")
    print(f"Senkou Span B: {latest_data['senkou_span_b']:.2f}")
    print(f"Chikou Span: {latest_data['chikou_span']:.2f}")
    print(f"VaR (95%): {latest_data['var_95']:.2f}")
    print(f"Expected Shortfall (95%): {latest_data['es_95']:.2f}")
    print(f"Max Drawdown: {latest_data['max_drawdown']:.2f}")
    print(f"Sharpe Ratio: {latest_data['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {latest_data['sortino_ratio']:.2f}")
    print(f"Calmar Ratio: {latest_data['calmar_ratio']:.2f}")


def refresh_and_check_signals(symbol, timeframe, exchange_id='mexc'):
    previous_buy_signal = False
    previous_sell_signal = False
    refresh_interval = 60  # seconds

    # Fetch initial data and print current information
    logging.info('Fetching initial market data...')
    df = fetch_data(symbol, timeframe, exchange_id)
    if df is not None:
        df = calculate_indicators(df)
        df = calculate_risk_metrics(df)
        df = calculate_performance_metrics(df)
        print_current_data(df, symbol)
        plot_data(df, symbol)

    while True:
        logging.info('Fetching latest market data...')
        df = fetch_data(symbol, timeframe, exchange_id)
        if df is not None:
            df = calculate_indicators(df)
            df = calculate_risk_metrics(df)
            df = calculate_performance_metrics(df)
            buy_signals, sell_signals = generate_signals(df)
            latest_buy_signal = buy_signals.iloc[-1]
            latest_sell_signal = sell_signals.iloc[-1]

            if latest_buy_signal != previous_buy_signal or latest_sell_signal != previous_sell_signal:
                logging.info('Signal update detected.')
                print_current_data(df, symbol)

                if latest_buy_signal:
                    print(f"BUY signal detected at price {df.iloc[-1]['close']:.2f}")
                elif latest_sell_signal:
                    print(f"SELL signal detected at price {df.iloc[-1]['close']:.2f}")
                else:
                    print("No clear buy or sell signal. Consider holding or conducting further analysis.")

                previous_buy_signal = latest_buy_signal
                previous_sell_signal = latest_sell_signal
                plot_data(df, symbol)
            else:
                logging.info('No signal updates.')
        else:
            logging.error('Unable to fetch data.')

        time.sleep(refresh_interval)


def main():
    symbol = input("Enter the futures contract you want to analyze (e.g., BTC/USDT): ")
    timeframe = input("Enter the timeframe (e.g., 1d, 4h, 1h): ")
    exchange_id = input("Enter the exchange ID (default: mexc): ") or 'mexc'

    logging.info("Fetching latest crypto news...")
    headlines = fetch_crypto_news()
    print("\nLatest Headlines:")
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")

    sentiment_score = sentiment_analysis(headlines)
    print(f"\nOverall sentiment score: {sentiment_score:.2f} (-1 to 1, negative to positive)")

    logging.info("Starting continuous signal monitoring...")
    refresh_and_check_signals(symbol, timeframe, exchange_id)


if __name__ == "__main__":
    main()
