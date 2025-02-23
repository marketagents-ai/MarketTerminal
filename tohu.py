import feedparser
import yfinance as yf
import pandas as pd
from datetime import datetime
import heapq
from typing import List, Dict, Tuple, Union, Literal, Optional
import markdown
import yaml
from pathlib import Path
import argparse
import json
import numpy as np
import re
from html import unescape
import ccxt
from dateutil import parser as date_parser
from datetime import timezone
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import time

PeriodType = Literal['3d', '5d', '1mo', '3mo', '1y']
CryptoPeriodType = Literal['3d', '5d', '1mo', '3mo', '1y']

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy/pandas numeric types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class CryptoTicker:
    """Mirror of yfinance.Ticker for cryptocurrency data"""
    
    def __init__(self, symbol: str, exchange: str = 'binance'):
        self.symbol = symbol
        self.exchange_id = exchange
        self.exchange = getattr(ccxt, self.exchange_id)()
        self._info_cache = None
    
    def info(self) -> Dict:
        """Get information about this cryptocurrency/trading pair."""
        if not self._info_cache:
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                markets = self.exchange.fetch_markets()
                market_info = next((m for m in markets if m['symbol'] == self.symbol), {})
                self._info_cache = {
                    'symbol': self.symbol,
                    'longName': self.symbol.replace('/', '-'),
                    'shortName': self.symbol.split('/')[0],
                    'exchange': self.exchange_id,
                    'marketCap': ticker.get('quoteVolume'),
                    'volume24h': ticker.get('volume'),
                    'previousClose': ticker.get('close'),
                    'open': ticker.get('open'),
                    'dayHigh': ticker.get('high'),
                    'dayLow': ticker.get('low'),
                    'bid': ticker.get('bid'),
                    'ask': ticker.get('ask'),
                    'baseAsset': market_info.get('base', self.symbol.split('/')[0]),
                    'quoteAsset': market_info.get('quote', self.symbol.split('/')[1]),
                    'pricePrecision': market_info.get('precision', {}).get('price'),
                    'percentChange24h': ticker.get('percentage'),
                }
            except Exception as e:
                print(f"Error fetching info for {self.symbol}: {str(e)}")
                self._info_cache = {'symbol': self.symbol, 'error': str(e)}
        return self._info_cache

    def _map_timeframe(self, period: CryptoPeriodType) -> str:
        """Map yfinance period to CCXT timeframe"""
        mapping = {
            '3d': '1h',    # 3 days with hourly data
            '5d': '1h',    # 5 days with hourly data
            '1mo': '4h',   # 1 month with 4-hour data
            '3mo': '1d',   # 3 months with daily data
            '1y': '1d',    # 1 year with daily data
        }
        return mapping.get(period, '1d')
    
    def _calculate_limit(self, period: CryptoPeriodType) -> int:
        """Calculate number of candles needed for the period"""
        mapping = {
            '3d': 72,     # 3 days * 24 hours
            '5d': 120,    # 5 days * 24 hours
            '1mo': 180,   # 30 days / 4 hours
            '3mo': 90,    # ~90 days
            '1y': 365,    # ~365 days
        }
        return mapping.get(period, 100)

    def history(self, period: CryptoPeriodType = '5d') -> pd.DataFrame:
        """Get historical market data."""
        try:
            timeframe = self._map_timeframe(period)
            limit = self._calculate_limit(period)
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            if not ohlcv:
                return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            }, inplace=True)
            
            return df
        except Exception as e:
            print(f"Error fetching history for {self.symbol}: {str(e)}")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

class CryptoFeed:
    """Main interface for cryptocurrency data"""
    
    def __init__(self):
        self.available_exchanges = ccxt.exchanges
        self._default_exchange = 'binance'
        
    def Ticker(self, symbol: str, exchange: str = None) -> CryptoTicker:
        """Get a CryptoTicker instance for the specified symbol."""
        if not exchange:
            exchange = self._default_exchange
            
        if '/' not in symbol:
            quote_assets = ['USDT', 'USD', 'BTC', 'ETH', 'BNB', 'BUSD']
            for quote in quote_assets:
                if symbol.endswith(quote):
                    base = symbol[:-len(quote)]
                    symbol = f"{base}/{quote}"
                    break
                    
        return CryptoTicker(symbol, exchange)
    
    def set_default_exchange(self, exchange: str) -> None:
        """Set the default exchange"""
        if exchange in self.available_exchanges:
            self._default_exchange = exchange
        else:
            print(f"Warning: Exchange '{exchange}' not found. Using {self._default_exchange}")

class CryptoSigns:
    """Cryptocurrency market data component."""
    
    def __init__(self, config: Dict):
        self.crypto_pairs = config.get('crypto_pairs', [])
        self.exchange = config.get('crypto_exchange', 'binance')
        self.feed = CryptoFeed()
        if self.exchange:
            self.feed._default_exchange = self.exchange

    def fetch_crypto_data(self) -> Dict[str, Dict]:
        """Fetch current cryptocurrency market data."""
        crypto_data = {}
        for pair in self.crypto_pairs:
            try:
                ticker = self.feed.Ticker(pair)
                info = ticker.info()
                history = ticker.history(period="1d")
                
                if not history.empty:
                    crypto_data[pair] = {
                        'price': history['Close'].iloc[-1],
                        'change': history['Close'].iloc[-1] - history['Open'].iloc[0],
                        'volume': history['Volume'].iloc[-1],
                        'name': info.get('longName', pair),
                        'market_cap': info.get('marketCap', 'N/A')
                    }
            except Exception as e:
                print(f"Error fetching crypto data for {pair}: {str(e)}")
        return crypto_data

class SECFiling:
    """Represents a parsed SEC filing with relevant transaction data"""
    
    def __init__(self, filing_data: Dict):
        self.accession_number = filing_data.get('accession_number')
        self.owner_name = filing_data.get('owner_name')
        self.owner_title = filing_data.get('owner_title', '')
        self.issuer_name = filing_data.get('issuer_name')
        self.filing_date = filing_data.get('filing_date')
        self.transactions = filing_data.get('transactions', [])
        self.footnotes = filing_data.get('footnotes', {})

class FinancialSigns:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.rss_feeds = self.config['rss_feeds']
        self.tickers = self.config['tickers']
        self.currency_pairs = self.config.get('currency_pairs', [])
        self.crypto = CryptoSigns(self.config) if 'crypto_pairs' in self.config else None
        self.news_cache = {}
        self.stock_cache = {}

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        required_keys = ['rss_feeds', 'tickers', 'output_settings']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in config file")
                
        return config

    def _clean_html(self, text: str) -> str:
        """Clean HTML tags, decode HTML entities, and remove copyright notices from text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        text = unescape(text)
        # Remove copyright notices (from © to end of sentence)
        text = re.sub(r'©.*?[.!?](?:\s|$)', '', text, flags=re.IGNORECASE)
        # Remove rights reserved phrases
        text = re.sub(r'All rights reserved\.?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'For personal use only\.?\s*', '', text, flags=re.IGNORECASE)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fetch_rss_news(self) -> List[Dict]:
        """
        Fetch news from RSS feeds and return top stories with standardized timestamps.
        Returns list of dicts with title, link, published (ISO format), source, and summary.
        """
        news_items = []
        
        def standardize_timestamp(date_str: str) -> str:
            """Convert various date formats to ISO format with UTC timezone"""
            try:
                # Parse the date string and ensure UTC timezone
                dt = date_parser.parse(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except Exception:
                return datetime.now(timezone.utc).isoformat()
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    # Get published date from various possible fields
                    date_str = entry.get('published', '')
                    if not date_str:
                        date_str = entry.get('updated', '')
                    if not date_str:
                        date_str = entry.get('created', '')
                    
                    news_items.append({
                        'title': self._clean_html(entry.title),
                        'link': entry.link,
                        'published': standardize_timestamp(date_str),
                        'source': self._clean_html(feed.feed.title),
                        'summary': self._clean_html(entry.get('summary', ''))
                    })
            except Exception as e:
                print(f"Error fetching RSS feed {feed_url}: {str(e)}")
                
        return news_items

    def fetch_stock_data(self) -> Dict[str, Dict]:
        """Fetch current stock data for tracked tickers."""
        stock_data = {}
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                history = stock.history(period="1d")
                
                stock_data[ticker] = {
                    'price': history['Close'].iloc[-1],
                    'change': history['Close'].iloc[-1] - history['Open'].iloc[0],
                    'volume': history['Volume'].iloc[-1],
                    'name': info.get('longName', ticker),
                    'market_cap': info.get('marketCap', 'N/A')
                }
            except Exception as e:
                print(f"Error fetching stock data for {ticker}: {str(e)}")
                
        return stock_data

    def fetch_currency_rates(self) -> Dict[str, Dict]:
        """Fetch current currency exchange rates."""
        currency_data = {}
        
        for pair in self.currency_pairs:
            try:
                ticker = yf.Ticker(f"{pair}=X")
                history = ticker.history(period="1d")
                
                if not history.empty:
                    currency_data[pair] = {
                        'rate': history['Close'].iloc[-1],
                        'change': history['Close'].iloc[-1] - history['Open'].iloc[0],
                        'change_pct': ((history['Close'].iloc[-1] - history['Open'].iloc[0]) / history['Open'].iloc[0]) * 100
                    }
            except Exception as e:
                print(f"Error fetching currency data for {pair}: {str(e)}")
                
        return currency_data

    def generate_markdown_report(self, top_k: int = None) -> str:
        """Generate a markdown report with top news and stock updates."""
        if top_k is None:
            top_k = self.config['output_settings'].get('top_k', 10)
            
        news_items = self.fetch_rss_news()
        stock_data = self.fetch_stock_data()
        
        # Sort news by date (assuming consistent date format)
        news_items.sort(key=lambda x: x['published'], reverse=True)
        
        # Generate markdown content
        md_content = f"# {self.config['output_settings'].get('report_title', 'Financial Terminal Report')}\n\n"
        md_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add stock summary
        md_content += "## Stock Updates\n\n"
        for ticker, data in stock_data.items():
            md_content += f"### {data['name']} ({ticker})\n"
            md_content += f"- Price: ${data['price']:.2f}\n"
            md_content += f"- Change: ${data['change']:.2f}\n"
            md_content += f"- Volume: {data['volume']:,}\n"
            md_content += f"- Market Cap: {data['market_cap']:,}\n\n"
        
        # Add currency rates section
        if self.currency_pairs:
            currency_data = self.fetch_currency_rates()
            md_content += "## Currency Exchange Rates\n\n"
            for pair, data in currency_data.items():
                md_content += f"### {pair}\n"
                md_content += f"- Rate: {data['rate']:.4f}\n"
                md_content += f"- Change: {data['change']:.4f} ({data['change_pct']:.2f}%)\n\n"
        
        # Add top news
        md_content += f"## Top {top_k} Latest News\n\n"
        for i, item in enumerate(news_items[:top_k], 1):
            md_content += f"### {i}. {item['title']}\n"
            md_content += f"Source: {item['source']} | Published: {item['published']}\n\n"
            md_content += f"{item['summary']}\n\n"
            md_content += f"({item['link']})\n\n"
        
        # Add cryptocurrency section if configured
        if self.crypto:
            crypto_data = self.crypto.fetch_crypto_data()
            md_content += "\n## Cryptocurrency Markets\n\n"
            for pair, data in crypto_data.items():
                md_content += f"### {data['name']} ({pair})\n"
                md_content += f"- Price: ${data['price']:.2f}\n"
                md_content += f"- Change: ${data['change']:.2f}\n"
                md_content += f"- Volume: {data['volume']:,.0f}\n"
                md_content += f"- Market Cap: {data['market_cap']:,}\n\n"
        
        return md_content

    def get_stock_history(self, ticker: str, period: PeriodType = '5d', width: int = 80, height: int = 20) -> Dict:
        """
        Fetch historical stock data and format as ASCII chart with external ticks.
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                raise ValueError("No data available for this period")
                
            prices = hist['Close'].values
            dates = hist.index.strftime('%Y-%m-%d').values
            
            # Safe price range calculation with fallback
            min_price = float(np.nanmin(prices))
            max_price = float(np.nanmax(prices))
            price_range = max_price - min_price
            
            # Handle edge cases
            if not price_range or np.isnan(price_range):
                base_price = float(np.nanmean(prices)) or 100.0
                min_price = base_price * 0.99
                max_price = base_price * 1.01
                price_range = max_price - min_price
            
            # Adjust dimensions for borders and ticks
            chart_width = width - 12  # Extra space for external price labels
            chart_height = height - 4  # Extra space for date labels
            
            # Safe normalization with bounds checking
            try:
                normalized = np.round(
                    ((prices - min_price) / price_range * (chart_height-1))
                ).clip(0, chart_height-1)
                normalized = np.nan_to_num(normalized, nan=chart_height//2).astype(np.int32)
            except (ValueError, TypeError, ZeroDivisionError):
                normalized = np.full_like(prices, chart_height//2, dtype=np.int32)
            
            # Create chart with dynamic size (including space for external ticks)
            chart = [[' ' for _ in range(width)] for _ in range(height)]
            
            # Draw borders
            for i in range(1, height-2):  # Adjusted for external ticks
                chart[i][8] = '│'  # Moved right to accommodate price labels
                chart[i][width-1] = '│'
            for i in range(8, width):  # Start after price labels
                chart[1][i] = '─'
                chart[height-3][i] = '─'  # Moved up for date labels
            chart[1][8] = '┌'
            chart[1][width-1] = '┐'
            chart[height-3][8] = '└'
            chart[height-3][width-1] = '┘'
            
            # Add price ticks with external labels
            for i in range(5):
                y_pos = 2 + int((height-5) * (4-i) / 4)  # Adjusted for borders
                price = min_price + (price_range * (i / 4))
                price_str = f"${price:,.0f}"
                chart[y_pos][7] = '─'  # External tick mark
                for j, char in enumerate(price_str[:7]):
                    chart[y_pos][j] = char
            
            # Safe plotting with bounds checking
            x_scale = (width-12) / max(len(normalized)-1, 1)
            for i in range(len(normalized) - 1):
                y1, y2 = normalized[i], normalized[i+1]
                x1 = min(9 + int(i * x_scale), width-2)
                x2 = min(9 + int((i + 1) * x_scale), width-2)
                
                if x2 > x1:
                    dx = x2 - x1
                    dy = y2 - y1
                    
                    if dx >= abs(dy):
                        for x in range(x1, x2):
                            y = int(y1 + dy * (x - x1) / dx)
                            y = min(max(2, y), height-4)  # Adjusted bounds
                            chart[height-3-y][x] = '\\' if dy < 0 else ('/' if dy > 0 else '─')
                    else:
                        step = 1 if y2 > y1 else -1
                        for y in range(min(y1, y2), max(y1, y2), step):
                            x = min(x1 + int(dx * abs(y - y1) / abs(dy or 1)), width-2)
                            if 2 <= y <= height-4:  # Adjusted bounds
                                chart[height-3-y][x] = '│'
                
                if 2 <= y1 <= height-4:  # Adjusted bounds
                    chart[height-3-y1][x1] = '●'
            
            # Date ticks below the chart
            valid_dates = min(5, len(dates))
            if valid_dates > 1:
                indices = np.linspace(0, len(dates)-1, valid_dates).astype(int)
                for idx in indices:
                    x_pos = min(9 + int(idx * x_scale), width-2)
                    chart[height-2][x_pos] = '│'  # External tick mark
                    date_str = dates[idx][-5:]
                    for j, char in enumerate(date_str):
                        pos = x_pos-2+j
                        if 8 <= pos < width-1:
                            chart[height-1][pos] = char
            
            return {
                'ticker': ticker,
                'period': period,
                'chart': '\n'.join(''.join(row) for row in chart),
                'current_price': float(np.nan_to_num(prices[-1])),
                'change': float(np.nan_to_num(prices[-1] - prices[0])),
                'change_pct': float(np.nan_to_num(((prices[-1] - prices[0]) / prices[0]) * 100))
            }
            
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {str(e)}")
            return None

    def get_volume_profile(self, ticker: str, period: PeriodType = '5d', width: int = 80, height: int = 20) -> Dict:
        """Generate volume profile showing price levels with highest trading activity"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                raise ValueError("No data available")
            
            # Calculate price range and bins
            price_min = hist['Low'].min()
            price_max = hist['High'].max()
            bins = height - 4  # Available space for bars
            
            # Create price bins and accumulate volume
            price_bins = np.linspace(price_min, price_max, bins)
            volumes = np.zeros(bins-1)  # One less than bins for histogram
            
            # Accumulate volume in each price bin
            for _, row in hist.iterrows():
                bin_indices = np.digitize([row['Close']], price_bins)[0] - 1
                if 0 <= bin_indices < len(volumes):  # Safety check
                    volumes[bin_indices] += row['Volume']
            
            # Normalize volumes for display
            max_vol = volumes.max() if volumes.any() else 1
            bar_lengths = np.floor((volumes * (width-12) / max_vol)).astype(int)
            
            # Create chart
            chart = [[' ' for _ in range(width)] for _ in range(height)]
            
          
            # Draw vertical border and price labels
            for i in range(2, height-3):
                chart[i][8] = '│'
                if i-2 < len(price_bins)-1:  # Check if we have a price for this level
                    price = price_bins[i-2]
                    price_str = f"${price:>7,.2f}"  # Right-align price with fixed width
                    for j, char in enumerate(price_str):
                        chart[i][j] = char
            
            # Plot volume bars (top to bottom)
            for i, length in enumerate(reversed(bar_lengths)):
                if i < height-4:  # Stay within bounds
                    y = i + 2  # Start below top border
                    for x in range(length):
                        if 9 <= x+9 < width-1:  # Leave space for right border
                            chart[y][x+9] = '█'
            
            # Add volume scale at top
            max_vol_str = f"Vol: {int(max_vol):,}"
            for i, char in enumerate(max_vol_str):
                if i < width-10:  # Leave space from right border
                    chart[0][i+9] = char
            
            return {
                'chart': '\n'.join(''.join(row) for row in chart),
                'max_volume': int(max_vol),
                'price_range': f"${price_min:.2f}-${price_max:.2f}"
            }
        except Exception as e:
            print(f"Error generating volume profile for {ticker}: {str(e)}")
            return None

    def get_rsi_chart(self, ticker: str, period: PeriodType = '5d', width: int = 80, height: int = 20) -> Dict:
        """Generate Relative Strength Index (RSI) chart"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                raise ValueError("No data available")
            
            # Calculate RSI with better NaN handling
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Use exponential moving average for smoother calculation
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            
            # Handle division by zero
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # Use newer pandas methods for filling NaN values
            rsi = rsi.ffill().fillna(50)  # Default to 50 if no valid values
            
            # Rest of the charting code remains the same
            chart = [[' ' for _ in range(width)] for _ in range(height)]
            
            # Draw borders and RSI levels
            for i in range(1, height-2):
                chart[i][8] = '│'
                rsi_level = 100 - (i-1) * (100/(height-4))
                level_str = f"{rsi_level:3.0f}"
                for j, char in enumerate(level_str[:7]):
                    chart[i][j] = char
                
            # Plot RSI line
            x_scale = (width-12) / max(len(rsi)-1, 1)
            y_scale = (height-4) / 100
            
            for i in range(len(rsi)-1):
                y1 = int((100-rsi.iloc[i]) * y_scale)
                y2 = int((100-rsi.iloc[i+1]) * y_scale)
                x1 = min(9 + int(i * x_scale), width-2)
                x2 = min(9 + int((i+1) * x_scale), width-2)
                
                if 1 <= y1 < height-3 and 1 <= y2 < height-3:
                    chart[y1+1][x1] = '●'
                    
                    # Draw connecting lines
                    if x2 > x1:
                        for x in range(x1+1, x2):
                            y = int(y1 + (y2-y1) * (x-x1)/(x2-x1))
                            if 1 <= y < height-3:
                                chart[y+1][x] = '─' if y2 == y1 else ('/' if y2 < y1 else '\\')
                                
            # Add overbought/oversold lines
            for i in range(9, width):
                chart[int(20 * y_scale)+1][i] = '·'  # Oversold (20)
                chart[int(80 * y_scale)+1][i] = '·'  # Overbought (80)
                
            return {
                'chart': '\n'.join(''.join(row) for row in chart),
                'current_rsi': float(rsi.iloc[-1])
            }
        except Exception as e:
            print(f"Error generating RSI chart for {ticker}: {str(e)}")
            return None

    def get_macd_chart(self, ticker: str, period: PeriodType = '5d', width: int = 80, height: int = 20) -> Dict:
        """Generate MACD (Moving Average Convergence Divergence) chart"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            # Calculate MACD
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            chart = [[' ' for _ in range(width)] for _ in range(height)]
            
            # Find value ranges for scaling
            max_val = max(macd.max(), signal.max(), histogram.max())
            min_val = min(macd.min(), signal.min(), histogram.min())
            value_range = max_val - min_val
            
            # Draw borders and value labels
            for i in range(1, height-2):
                chart[i][8] = '│'
                value = max_val - (i-1) * (value_range/(height-4))
                value_str = f"{value:.2f}"
                for j, char in enumerate(value_str[:7]):
                    chart[i][j] = char
                
            # Plot MACD components
            x_scale = (width-12) / max(len(macd)-1, 1)
            y_scale = (height-4) / value_range
            
            # Plot histogram bars
            for i in range(len(histogram)):
                x = min(9 + int(i * x_scale), width-2)
                y_zero = int((max_val / value_range) * (height-4))
                y_val = int(histogram.iloc[i] * y_scale)
                
                if y_val > 0:
                    for y in range(y_zero-y_val, y_zero):
                        if 1 <= y < height-3:
                            chart[y+1][x] = '█'
                else:
                    for y in range(y_zero, y_zero-y_val):
                        if 1 <= y < height-3:
                            chart[y+1][x] = '░'
                            
            # Plot MACD and signal lines
            for series, char in [(macd, '─'), (signal, '·')]:
                for i in range(len(series)-1):
                    y1 = int((max_val - series.iloc[i]) * y_scale)
                    y2 = int((max_val - series.iloc[i+1]) * y_scale)
                    x1 = min(9 + int(i * x_scale), width-2)
                    x2 = min(9 + int((i+1) * x_scale), width-2)
                    
                    if 1 <= y1 < height-3 and 1 <= y2 < height-3:
                        chart[y1+1][x1] = '●'
                        if x2 > x1:
                            for x in range(x1+1, x2):
                                y = int(y1 + (y2-y1) * (x-x1)/(x2-x1))
                                if 1 <= y < height-3:
                                    chart[y+1][x] = char
                                    
            return {
                'chart': '\n'.join(''.join(row) for row in chart),
                'current_macd': float(macd.iloc[-1]),
                'current_signal': float(signal.iloc[-1]),
                'current_histogram': float(histogram.iloc[-1])
            }
        except Exception as e:
            print(f"Error generating MACD chart for {ticker}: {str(e)}")
            return None

    def get_insider_trades(self, ticker: str, limit: int = 10) -> List[SECFiling]:
        """Fetch recent Form 4 filings for ticker."""
        try:
            headers = {
                'User-Agent': 'FinancialSigns/1.0 (your@email.com)',
                'Accept-Encoding': 'gzip, deflate'
            }
            
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=4&owner=include&count={limit}&output=atom"
            print(f"Fetching from: {url}")
            time.sleep(0.1)
            response = requests.get(url, headers=headers)
            
            feed = feedparser.parse(response.text)
            print(f"Found {len(feed.entries)} entries")
            filings = []
            
            # Get company CIK from feed title (e.g., "MICROSOFT CORP  (0000789019)")
            company_cik = None
            if feed.feed.title:
                cik_match = re.search(r'\((\d+)\)', feed.feed.title)
                if cik_match:
                    company_cik = cik_match.group(1).lstrip('0')
            
            for entry in feed.entries[:limit]:
                print(f"Processing entry: {entry.link}")
                match = re.search(r'/(\d{10}-\d{2}-\d{6})-index', entry.link)
                if match and company_cik:
                    accession_number = match.group(1)
                    # Pass both the company CIK and accession number
                    filing = self.fetch_sec_filing(accession_number, company_cik)
                    if filing:
                        filings.append(filing)
                        
            return filings
            
        except Exception as e:
            print(f"Error fetching insider trades for {ticker}: {str(e)}")
            return []

    def fetch_sec_filing(self, accession_number: str, company_cik: str = None) -> SECFiling:
        """Fetch basic Form 4 filing info from SEC."""
        try:
            headers = {
                'User-Agent': 'FinancialSigns/1.0 (your@email.com)',
            }
            
            parts = accession_number.split('-')
            # Use provided company CIK if available, otherwise use accession number CIK
            cik = company_cik or parts[0].lstrip('0')
            acc = ''.join(parts)
            
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{accession_number}.txt"
            print(f"Fetching SEC filing from: {url}")
            
            time.sleep(0.1)
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            content = response.text
            
            # Basic header info
            filing_data = {
                'accession_number': accession_number,
                'owner_name': re.search(r'COMPANY CONFORMED NAME:\s*([^\n]+)', content).group(1).strip(),
                'issuer_name': re.search(r'ISSUER:.*?COMPANY CONFORMED NAME:\s*([^\n]+)', content, re.DOTALL).group(1).strip(),
                'filing_date': re.search(r'FILED AS OF DATE:\s*(\d+)', content).group(1),
                'transactions': [],
                'footnotes': {}
            }
            
            xml_start = content.find('<XML>')
            if xml_start != -1:
                xml_content = content[xml_start:]
                
                # Get title
                title_match = re.search(r'<officerTitle>(.*?)</officerTitle>', xml_content)
                if title_match:
                    filing_data['owner_title'] = title_match.group(1).strip()
                
                # Get footnotes
                for match in re.finditer(r'<footnote id="(F\d+)">(.*?)</footnote>', xml_content, re.DOTALL):
                    filing_data['footnotes'][match.group(1)] = match.group(2).strip()
                
                # Find transactions with footnotes
                trans_pattern = (
                    r'<nonDerivativeTransaction>.*?'
                    r'<transactionAmounts>.*?'
                    r'<transactionShares>.*?<value>(\d+\.?\d*)</value>'
                    r'(?:.*?<footnoteId id="(F\d+)"/>)?.*?'
                    r'<transactionPricePerShare>.*?<value>(\d+\.?\d*)</value>'
                )
                
                for match in re.finditer(trans_pattern, xml_content, re.DOTALL):
                    transaction = {
                        'shares': float(match.group(1)),
                        'price_per_share': float(match.group(3))
                    }
                    if match.group(2):  # If footnote reference exists
                        footnote_id = match.group(2)
                        if footnote_id in filing_data['footnotes']:
                            transaction['footnote'] = filing_data['footnotes'][footnote_id]
                    
                    filing_data['transactions'].append(transaction)
            
            return SECFiling(filing_data)
            
        except Exception as e:
            print(f"Error fetching SEC filing {accession_number}: {str(e)}")
            return None

    def search_company(self, ticker: str) -> Dict[str, str]:
        """Search SEC EDGAR for company by ticker."""
        try:
            headers = {
                'User-Agent': 'FinancialSigns/1.0 (your@email.com)',
                'Accept-Encoding': 'gzip, deflate'
            }
            
            # Add owner=include to match get_insider_trades
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=4&owner=include&output=atom"
            time.sleep(0.1)
            response = requests.get(url, headers=headers)
            
            feed = feedparser.parse(response.text)
            if feed.feed.title:
                return {'name': feed.feed.title}
            return None
            
        except Exception as e:
            print(f"Error searching for ticker {ticker}: {str(e)}")
            return None

    def get_report(self, format: str = 'markdown', top_k: int = None, period: PeriodType = '5d', save_path: str = None) -> Union[str, Dict]:
        """Generate financial report in specified format and optionally save to file.
        
        Args:
            format: Output format ('markdown' or 'json')
            top_k: Number of news items to include
            period: Historical data period
            save_path: Optional path to save the report
        """
        if top_k is None:
            top_k = self.config['output_settings'].get('top_k', 10)
            
        news_items = self.fetch_rss_news()
        stock_data = self.fetch_stock_data()
        news_items.sort(key=lambda x: x['published'], reverse=True)
        
        if format == 'json':
            currency_data = self.fetch_currency_rates() if self.currency_pairs else {}
            crypto_data = self.crypto.fetch_crypto_data() if self.crypto else {}
            result = {
                'generated_at': datetime.now().isoformat(),
                'stocks': {k: {
                    'price': float(v['price']),
                    'change': float(v['change']),
                    'volume': int(v['volume']),
                    'name': v['name'],
                    'market_cap': str(v['market_cap']) if v['market_cap'] != 'N/A' else 'N/A',
                    'history': {
                        'current_price': float(v['price']),
                        'change': float(v['change']),
                        'change_pct': float(v['change'] / (v['price'] - v['change']) * 100)
                    }
                } for k, v in stock_data.items()},
                'currencies': currency_data,
                'crypto': crypto_data,
                'news': news_items[:top_k]
            }
        else:
            md_content = self.generate_markdown_report(top_k)
            md_content += "\n## Historical Charts\n\n"
            for ticker in self.tickers:
                history = self.get_stock_history(ticker, period)
                if history:
                    md_content += f"\n### {ticker} ({period})\n"
                    md_content += "```\n"
                    md_content += history['chart']
                    md_content += "\n```\n"
                    md_content += f"Change: ${history['change']:.2f} ({history['change_pct']:.1f}%)\n\n"
            
            for ticker in self.tickers:
                md_content += f"\n### Technical Analysis for {ticker}\n\n"
                
                # Volume Profile
                volume_profile = self.get_volume_profile(ticker, period)
                if volume_profile:
                    md_content += "#### Volume Profile\n```\n"
                    md_content += volume_profile['chart']
                    md_content += "\n```\n"
                    
                # RSI
                rsi_chart = self.get_rsi_chart(ticker, period)
                if rsi_chart:
                    md_content += f"#### RSI (Current: {rsi_chart['current_rsi']:.1f})\n```\n"
                    md_content += rsi_chart['chart']
                    md_content += "\n```\n"
                    
                # MACD
                macd_chart = self.get_macd_chart(ticker, period)
                if macd_chart:
                    md_content += "#### MACD\n```\n"
                    md_content += macd_chart['chart']
                    md_content += "\n```\n"
                    md_content += f"MACD: {macd_chart['current_macd']:.3f} "
                    md_content += f"Signal: {macd_chart['current_signal']:.3f} "
                    md_content += f"Histogram: {macd_chart['current_histogram']:.3f}\n\n"

            # Add crypto if configured
            if self.crypto:
                for pair in self.crypto.crypto_pairs:
                    ticker = self.crypto.feed.Ticker(pair)
                    history = ticker.history(period)
                    if not history.empty:
                        last_price = history['Close'].iloc[-1]
                        first_price = history['Open'].iloc[0]
                        change = last_price - first_price
                        change_pct = (change / first_price) * 100
                        md_content += "#### Crypto Exchange\n"
                        md_content += f"\n### {pair} ({period})\n"
                        md_content += f"Price: ${last_price:.2f}\n"
                        md_content += f"Change: ${change:.2f} ({change_pct:.1f}%)\n\n"
            
            result = md_content
        
        # Save report if path provided
        if save_path:
            save_path = Path(save_path)
            if format == 'json':
                with open(save_path.with_suffix('.json'), 'w') as f:
                    json.dump(result, f, indent=2, cls=CustomJSONEncoder)
            else:
                with open(save_path.with_suffix('.md'), 'w') as f:
                    f.write(result)
        
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Financial Terminal Report Generator')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--format', choices=['markdown', 'json'], default='markdown',
                       help='Output format')
    parser.add_argument('--top-k', type=int, help='Number of news items')
    parser.add_argument('--period', choices=['3d', '5d', '1mo', '3mo', '1y'], 
                       default='5d', help='Historical data period')
    parser.add_argument('--save', help='Path to save the report (without extension)')
    args = parser.parse_args()
    
    terminal = FinancialSigns(args.config)
    
    result = terminal.get_report(format=args.format, top_k=args.top_k, period=args.period, save_path=args.save)
    
    if args.format == 'json':
        print(json.dumps(result, indent=2, cls=CustomJSONEncoder))
    else:
        print(result)
    
