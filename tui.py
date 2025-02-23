#!/usr/bin/env python3
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console, Group
from rich import box

from tohu import FinancialSigns

from datetime import datetime
import time
import msvcrt  # Windows-specific keyboard inputq

from ollama import chat

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum

import yfinance as yf

import yaml
import json

from concurrent.futures import ThreadPoolExecutor
import threading
import queue

import random
from fuzzywuzzy import fuzz
import math
from pathlib import Path


class MarketSentiment(str, Enum):
    """Market sentiment classifications"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    HAWKISH = "hawkish"
    CAUTIOUS = "cautious"
    NEUTRAL = "neutral"

class NewsContextMode(str, Enum):
    """News filtering modes for AI context building"""
    RECENT = "recent"     # Most recent news items
    RANDOM = "random"     # Random sampling of news
    RELEVANT = "relevant" # News relevant to current ticker

class LoadingState(str, Enum):
    """Loading states for panels"""
    READY = "ready"
    LOADING = "loading"
    ERROR = "error"

class ASCIIAnimation:
    """Handles loading and display of ASCII art animations with proper centering"""
    def __init__(self, asset_path: str):
        self.frames = []
        self.current_frame = 0
        self.last_update = time.time()
        self.frame_duration = 0.2
        self.dimensions = (0, 0)  # (width, height) of the largest frame
        
        try:
            # Load frames from markdown file
            with open(asset_path, 'r') as f:
                content = f.read()
                blocks = content.split('```')
                # Preserve whitespace when extracting frames
                raw_frames = []
                for block in blocks[1::2]:
                    # Remove any trailing/leading newlines but preserve internal whitespace
                    frame = block.strip('\n')
                    if frame:  # Only add non-empty frames
                        raw_frames.append(frame)
                
                # Calculate maximum dimensions including whitespace
                max_width = 0
                max_height = 0
                for frame in raw_frames:
                    lines = frame.splitlines()
                    max_width = max(max_width, max(len(line) for line in lines))
                    max_height = max(max_height, len(lines))
                
                self.dimensions = (max_width, max_height)
                self.frames = raw_frames
                
        except Exception:
            # Fallback spinner with fixed dimensions
            self.frames = [
                "⠋", "⠙", "⠹",
                "⠸", "⠼", "⠴",
                "⠦", "⠧", "⠇",
                "⠏"
            ]
            self.dimensions = (9, 1)  # Fixed size for fallback spinner

    def get_frame(self) -> str:
        """Get current animation frame"""
        current_time = time.time()
        if current_time - self.last_update > self.frame_duration:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.last_update = current_time
        return self.frames[self.current_frame]

    def position_in_panel(self, panel_width: int, panel_height: int) -> Text:
        """Left-align and top-align the current frame in the given panel dimensions"""
        frame = self.get_frame()
        frame_lines = frame.splitlines()
        
        # Left alignment - minimal padding
        h_padding = 0  # Minimal left padding
        
        # No vertical padding for top alignment
        content = Text()
        
        # Add each line with minimal horizontal padding
        for line in frame_lines:
            leading_spaces = len(line) - len(line.lstrip())
            padding = ' ' * (h_padding + leading_spaces)
            content.append(padding + line.lstrip() + '\n', style="bold orange3")
        
        # Fill remaining height with empty lines
        remaining_height = panel_height - len(frame_lines)
        if remaining_height > 0:
            content.append('\n' * remaining_height)
            
        return content

class TrendArrows:
    """Unicode arrows for showing trends and their strength"""
    STRONG_UP = "↑↑"    # or "↑↑"
    UP = "↑"
    SLIGHT_UP = "↗"   # or "↗"
    NEUTRAL = "→"
    SLIGHT_DOWN = "↘"  # or "↘"
    DOWN = "↓"
    STRONG_DOWN = "↓↓"  # or "↓↓"
    
    @classmethod
    def get_sentiment_arrow(cls, value: float) -> str:
        """Get arrow based on sentiment value (-2 to +2 scale)"""
        if value > 1.5: return cls.STRONG_UP
        if value > 0.5: return cls.UP
        if value > 0.1: return cls.SLIGHT_UP
        if value < -1.5: return cls.STRONG_DOWN
        if value < -0.5: return cls.DOWN
        if value < -0.1: return cls.SLIGHT_DOWN
        return cls.NEUTRAL
    
    @classmethod
    def get_price_arrow(cls, change_pct: float) -> str:
        """Get arrow based on price change percentage"""
        if change_pct > 5: return cls.STRONG_UP
        if change_pct > 2: return cls.UP
        if change_pct > 0: return cls.SLIGHT_UP
        if change_pct < -5: return cls.STRONG_DOWN
        if change_pct < -2: return cls.DOWN
        if change_pct < 0: return cls.SLIGHT_DOWN
        return cls.NEUTRAL

class SentimentTracker:
    """Tracks weighted rolling sentiment averages with persistence"""
    def __init__(self, window_size: int = 10, file_path: str = 'sentiment_history.jsonl', retention_days: int = 7):
        self.window_size = window_size
        self.file_path = file_path
        self.retention_days = retention_days
        self.sentiments: Dict[str, List[tuple[MarketSentiment, int, float]]] = {}
        self.load_sentiments()
        
    def load_sentiments(self):
        """Load historical sentiments from jsonl, filtering by recency"""
        try:
            current_time = time.time()
            retention_seconds = self.retention_days * 24 * 60 * 60
            with open(self.file_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Skip entries older than retention period
                    if current_time - entry['timestamp'] > retention_seconds:
                        continue
                    ticker = entry['ticker']
                    if ticker not in self.sentiments:
                        self.sentiments[ticker] = []
                    self.sentiments[ticker].append((
                        MarketSentiment(entry['sentiment']),
                        entry['confidence'],
                        entry['timestamp']
                    ))
            # Trim to window size for each ticker
            for ticker in self.sentiments:
                self.sentiments[ticker] = sorted(
                    self.sentiments[ticker], 
                    key=lambda x: x[2],
                    reverse=True
                )[:self.window_size]
        except FileNotFoundError:
            pass
    
    def save_sentiment(self, ticker: str, sentiment: MarketSentiment, confidence: int):
        """Save new sentiment to jsonl file"""
        entry = {
            'ticker': ticker,
            'sentiment': sentiment.value,
            'confidence': confidence,
            'timestamp': time.time()
        }
        with open(self.file_path, 'a') as f:
            json.dump(entry, f)
            f.write('\n')
            
    def add_sentiment(self, ticker: str, sentiment: MarketSentiment, confidence: int):
        """Add sentiment and persist to file"""
        if ticker not in self.sentiments:
            self.sentiments[ticker] = []
        
        timestamp = time.time()
        self.sentiments[ticker].append((sentiment, confidence, timestamp))
        if len(self.sentiments[ticker]) > self.window_size:
            self.sentiments[ticker].pop(0)
            
        self.save_sentiment(ticker, sentiment, confidence)
    
    def get_average_sentiment(self, ticker: str) -> Optional[tuple[MarketSentiment, float, float]]:
        """Returns (sentiment, confidence, raw_value) with time-weighted values"""
        if ticker not in self.sentiments or not self.sentiments[ticker]:
            return None
            
        current_time = time.time()
        
        # Convert sentiments to numerical values with time decay
        sentiment_values = {
            MarketSentiment.BULLISH: 2,
            MarketSentiment.HAWKISH: 1,
            MarketSentiment.NEUTRAL: 0,
            MarketSentiment.CAUTIOUS: -1,
            MarketSentiment.BEARISH: -2
        }
        
        # Calculate time-weighted average (newer sentiments count more)
        total_weight = 0
        weighted_sum = 0
        
        for sent, conf, timestamp in self.sentiments[ticker]:
            # Exponential decay over 24 hours
            age_hours = (current_time - timestamp) / 3600
            time_weight = math.exp(-age_hours / 24)
            weight = conf * time_weight
            
            total_weight += weight
            weighted_sum += sentiment_values[sent] * weight
        
        avg_value = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Convert back to sentiment
        if avg_value > 1.5:
            sentiment = MarketSentiment.BULLISH
        elif avg_value > 0.5:
            sentiment = MarketSentiment.HAWKISH
        elif avg_value > -0.5:
            sentiment = MarketSentiment.NEUTRAL
        elif avg_value > -1.5:
            sentiment = MarketSentiment.CAUTIOUS
        else:
            sentiment = MarketSentiment.BEARISH
            
        # Calculate confidence as percentage of max possible confidence
        # Factor in age of data
        max_conf = self.window_size * 5  # max confidence per entry
        confidence = min(100, int((total_weight / max_conf) * 100))
        
        return sentiment, confidence, avg_value

def load_config():
    """Load config from config.yaml"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return (
        config['system_prompt'],  
        config['market_context_prompt'],  
        config['model'],
        config['field_descriptions']
    )

# Load config at module level
SYSTEM_PROMPT, MARKET_CONTEXT_PROMPT, MODEL_NAME, FIELD_DESCRIPTIONS = load_config()

class MarketAnalysis(BaseModel):
    __doc__ = f"""Analysis powered by {MODEL_NAME}"""
    summary: str = Field(..., description=FIELD_DESCRIPTIONS['summary'])
    thinking: str = Field(..., description=FIELD_DESCRIPTIONS['thinking'])
    sentiment: MarketSentiment = Field(..., description=FIELD_DESCRIPTIONS['sentiment'])
    key_factors: List[str] = Field(..., min_items=1, description=FIELD_DESCRIPTIONS['key_factors'])
    technical_indicators: List[str] = Field(..., min_items=1, description=FIELD_DESCRIPTIONS['technical_indicators'])
    risk_factors: List[str] = Field(..., min_items=1, description=FIELD_DESCRIPTIONS['risk_factors'])
    recommendation: str = Field(..., description=FIELD_DESCRIPTIONS['recommendation'])
    confidence_level: int = Field(..., ge=1, le=5, description=FIELD_DESCRIPTIONS['confidence_level'])
    #time_horizon: str = Field(..., description=FIELD_DESCRIPTIONS['time_horizon'])

class AnalysisManager:
    """Manages asynchronous market analysis requests"""
    def __init__(self, throttle_seconds: int = 60):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_future = None
        self.analysis_queue = queue.Queue()
        self.last_request_time = 0
        self.throttle_seconds = throttle_seconds

    def request_analysis(self, analysis_func, *args, **kwargs):
        """Submit a new analysis request if not already running"""
        current_time = time.time()
        if (current_time - self.last_request_time < self.throttle_seconds or 
            (self.current_future and not self.current_future.done())):
            return

        self.last_request_time = current_time
        self.current_future = self.executor.submit(analysis_func, *args, **kwargs)
        self.current_future.add_done_callback(self._handle_completion)


    def _handle_completion(self, future):
        """Callback when analysis completes - puts result in queue"""
        try:
            result = future.result()
            self.analysis_queue.put(("success", result))
        except Exception as e:
            self.analysis_queue.put(("error", str(e)))

    def check_results(self) -> Optional[tuple[str, Any]]:
        """Non-blocking check for new results"""
        try:
            return self.analysis_queue.get_nowait()
        except queue.Empty:
            return None

class AwarenessConfig(BaseModel):
    """Configuration for news awareness filtering"""
    limit: int = Field(default=8, description="Number of news items to include")
    relevance_threshold: float = Field(default=60.0, description="Minimum relevance score")
    weights: Dict[str, float] = Field(
        default={
            "ticker": 1.0,
            "name": 0.9,
            "industry": 0.8,
            "sector": 0.7
        }
    )

class RefreshConfig(BaseModel):
    """Configuration for all refresh rates and timing intervals"""
    screen_fps: int = Field(default=4, ge=1, le=60, description="Screen refreshes per second")
    data_update: int = Field(default=60, ge=10, description="Seconds between market data updates")
    analysis_throttle: int = Field(default=60, ge=10, description="Minimum seconds between AI analysis")
    scroll_interval: float = Field(default=1.0, ge=0.1, description="Seconds between auto-scrolls")
    sentiment_retention: int = Field(default=7, ge=1, description="Days to keep sentiment history")

class FinancialDashboard:
    DEFAULT_PERIOD = '1mo'
    SCROLL_INDICATORS = {
        'static': '•',    # Static mode
        'infinite': '∞',  # Infinite scroll
        'pingpong': '↕'   # Pingpong mode
    }
    CHART_HEIGHT_RATIO = 0.33  # 33% of available height
    MIN_CHART_HEIGHT = 10
    MIN_CHART_WIDTH = 40
    PADDING = {
        'horizontal': 6,  # left+right padding
        'vertical': 2     # top+bottom padding
    }
    AVAILABLE_PERIODS = ['3d', '5d', '1mo', '3mo', '6mo', '1y']
    SCROLL_MODES = ["static", "infinite", "pingpong"]
    CHART_TYPES = ["price", "volume", "rsi", "macd"]  # Add chart types

    def __init__(self, config_path: str = 'config.yaml'):
        # Load refresh rates from config first
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.refresh_config = RefreshConfig(**config.get('refresh_rates', {}))
        
        self.terminal = FinancialSigns(config_path)
        self.news_items: List[Dict] = []
        self.stock_data: Dict = {}
        self.currency_data: Dict = {}
        self.crypto_data: Dict = {}
        self.charts = {
            "price": {},
            "volume": {},
            "rsi": {},
            "macd": {}
        }
        self.update_queue = queue.Queue()
        self.last_update = datetime.now()
        self.current_ticker_idx = 0
        self.current_period = '1mo'
        self.console = Console()

        # Scroll state
        self.news_scroll = 0
        self.scroll_mode = "infinite"
        self.scroll_direction = 1
        self.last_scroll_time = time.time()
        self.scroll_interval = 1.0

        # Search state
        self.search_mode = False
        self.search_query = ""
        self.search_results: List[Dict] = []

        # Command input state (integrated in status panel)
        self.input_mode = False
        self.command_str = ""

        # AI Chat state
        self.ai_messages = []
        self.ai_response = None
        self.ai_error = None

        # Add sentiment tracking with config
        self.stock_sentiments: Dict[str, MarketSentiment] = {}
        self.sentiment_tracker = SentimentTracker(
            retention_days=self.refresh_config.sentiment_retention
        )

        self.analysis_manager = AnalysisManager(
            throttle_seconds=self.refresh_config.analysis_throttle
        )

        self.current_chart_type = "price"  # Add current chart type

        self.news_context_mode = NewsContextMode.RECENT

        self.scroll_interval = self.refresh_config.scroll_interval

        # Load awareness config once
        self.awareness_config = AwarenessConfig()
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f).get('awareness', {})
                self.awareness_config = AwarenessConfig(**yaml_config)
        except Exception:
            pass

        # Add loading animation
        assets_dir = Path(__file__).parent / 'assets'
        self.loading_animation = ASCIIAnimation(
            assets_dir / 'tui_loading.md'
        )

        # Add loading states
        self.panel_states = {
            "ai_analysis": LoadingState.LOADING,
            "chart": LoadingState.LOADING,
            "market_data": LoadingState.LOADING,
            "news": LoadingState.LOADING
        }

    def update_data(self):
        """Background thread to update news, stocks, charts and currency rates."""
        last_size = None
        while True:
            try:
                self.news_items = self.terminal.fetch_rss_news()
                self.stock_data = self.terminal.fetch_stock_data()
                self.currency_data = self.terminal.fetch_currency_rates()
                if self.terminal.crypto:
                    self.crypto_data = self.terminal.crypto.fetch_crypto_data()

                # Check for terminal size change (force chart update if changed)
                current_size = self.console.size
                force_chart_update = (last_size != current_size)
                last_size = current_size

                width, height = current_size
                # Calculate chart dimensions based on available size and padding
                chart_height = max(int(height * self.CHART_HEIGHT_RATIO) - self.PADDING['vertical'], self.MIN_CHART_HEIGHT)
                chart_width = max(width - self.PADDING['horizontal'], self.MIN_CHART_WIDTH)

                # Update all chart types for each ticker if needed
                if force_chart_update or not any(self.charts.values()):
                    for ticker in self.terminal.tickers:
                        # Price history chart
                        history = self.terminal.get_stock_history(
                            ticker,
                            self.current_period,
                            width=chart_width,
                            height=chart_height
                        )
                        if history:
                            self.charts["price"][ticker] = history['chart']

                        # Volume profile
                        volume = self.terminal.get_volume_profile(
                            ticker,
                            self.current_period,
                            width=chart_width,
                            height=chart_height
                        )
                        if volume:
                            self.charts["volume"][ticker] = volume['chart']

                        # RSI chart
                        rsi = self.terminal.get_rsi_chart(
                            ticker,
                            self.current_period,
                            width=chart_width,
                            height=chart_height
                        )
                        if rsi:
                            self.charts["rsi"][ticker] = rsi['chart']

                        # MACD chart
                        macd = self.terminal.get_macd_chart(
                            ticker,
                            self.current_period,
                            width=chart_width,
                            height=chart_height
                        )
                        if macd:
                            self.charts["macd"][ticker] = macd['chart']

                self.last_update = datetime.now()
                self.update_queue.put("Data updated")
                time.sleep(self.refresh_config.data_update)
            except Exception as e:
                self.update_queue.put(f"Error: {str(e)}")
                time.sleep(5)  # Error backoff

    def search(self, query: str) -> List[Dict]:
        """Search through data and return items in standard news format
        Searches:
        - News: title, summary, description, content
        - Stocks: ticker, name, sector, industry, price ranges, volume thresholds
        Returns standardized news-format items for display
        """
        results = []
        query = query.lower()
        
        # 1. Search through news items
        for news in self.news_items:
            content = (news.get('summary') or 
                      news.get('description') or 
                      news.get('content:encoded', '')[:500])
                      
            if query in news['title'].lower() or query in content.lower():
                news_item = {
                    'title': news['title'],
                    'summary': content,
                    'source': news['source'],
                    'published': news['published'],
                    'link': news['link']
                }
                results.append(news_item)
        
        # 2. Enhanced stock search
        for ticker, data in self.stock_data.items():
            stock_info = {}
            try:
                stock = yf.Ticker(ticker)
                stock_info = stock.info
            except:
                pass

            # Search criteria
            search_matches = [
                query in ticker.lower(),
                query in data['name'].lower(),
                query in stock_info.get('sector', '').lower(),
                query in stock_info.get('industry', '').lower(),
            ]

            # Price range search (e.g. "price>100" or "price<50")
            if 'price' in query:
                try:
                    if '>' in query:
                        price_threshold = float(query.split('>')[1])
                        search_matches.append(data['price'] > price_threshold)
                    elif '<' in query:
                        price_threshold = float(query.split('<')[1])
                        search_matches.append(data['price'] < price_threshold)
                except:
                    pass

            # Volume threshold search (e.g. "vol>1000000")
            if 'vol' in query:
                try:
                    if '>' in query:
                        vol_threshold = float(query.split('>')[1])
                        search_matches.append(data['volume'] > vol_threshold)
                    elif '<' in query:
                        vol_threshold = float(query.split('<')[1])
                        search_matches.append(data['volume'] < vol_threshold)
                except:
                    pass

            # Market cap search (e.g. "cap>1B" or "cap<500M")
            if 'cap' in query:
                try:
                    multiplier = {'T': 1e12, 'B': 1e9, 'M': 1e6}.get(query[-1].upper(), 1)
                    value = float(query[4:-1]) * multiplier
                    if '>' in query:
                        search_matches.append(data['market_cap'] > value)
                    elif '<' in query:
                        search_matches.append(data['market_cap'] < value)
                except:
                    pass

            # Add stock to results if any criteria match
            if any(search_matches):
                extra_info = f"Sector: {stock_info.get('sector', 'N/A')} | Industry: {stock_info.get('industry', 'N/A')}"
                results.append({
                    'title': f"{ticker} - {data['name']}",
                    'summary': (f"Price: ${data['price']:.2f} ({data['change']:+.2f}%) | "
                              f"Volume: {data['volume']:,} | "
                              f"Cap: {self.format_market_cap(data['market_cap'])} | "
                              f"{extra_info}"),
                    'source': 'Stocks',
                    'published': datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z'),
                    'link': f"https://finance.yahoo.com/quote/{ticker}"
                })

        return results

    def format_market_cap(self, value: float) -> str:
        if not isinstance(value, (int, float)):
            return str(value)
        if value >= 1e12:
            return f"${value/1e12:.1f}T"
        if value >= 1e9:
            return f"${value/1e9:.1f}B"
        return f"${value/1e6:.1f}M"

    def format_timestamp(self, timestamp_str: str, fmt: str = '%m-%d %H:%M') -> str:
        try:
            timestamp = datetime.strptime(timestamp_str, '%a, %d %b %Y %H:%M:%S %z')
            return timestamp.strftime(fmt)
        except (ValueError, TypeError):
            return timestamp_str[:16]

    def log_api_interaction(self, request: dict, response: dict, error: str = None):
        """Log API interactions to JSONL file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": request,
            "response": response,
            "error": error
        }
        with open('market_analysis.jsonl', 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

    def get_market_analysis(self) -> Optional[MarketAnalysis]:
        """Get AI analysis using comprehensive market context"""
        retries = 3
        request_data = {}
        
        while retries > 0:
            try:
                current_ticker = list(self.terminal.tickers)[self.current_ticker_idx]
                
                # Build context from all available data
                market_context = "Market Analysis Context:\n\n"
                
                # Add previous analysis history (currently hardcoded but this should be dynamic and inherited from the model)
                if self.ai_messages:
                    market_context += "Previous Analysis History:\n"
                    for msg in self.ai_messages[-3:]:
                        analysis = msg.model_dump()
                        #market_context += f"[{analysis['time_horizon']}] "
                        market_context += f"Thinking: {analysis['thinking']}\n"
                        market_context += f"Sentiment: {analysis['sentiment']}, "
                        market_context += f"Confidence: {analysis['confidence_level']}/5\n"
                        market_context += f"Summary: {analysis['summary']}\n"
                        market_context += f"Key Factors: {', '.join(analysis['key_factors'])}\n"
                        market_context += f"Recommendation: {analysis['recommendation']}\n\n"
                
                # Current Stock Focus with Technical Analysis
                if current_ticker in self.stock_data:
                    data = self.stock_data[current_ticker]
                    stock = yf.Ticker(current_ticker)
                    info = stock.info
                    
                    market_context += f"Primary Focus - {data['name']} ({current_ticker}):\n"
                    market_context += f"Price: ${data['price']}, Change: {data['change']}%, "
                    market_context += f"Volume: {data['volume']}, Market Cap: {self.format_market_cap(data['market_cap'])}\n"
                    market_context += f"Sector: {info.get('sector', 'N/A')}, Industry: {info.get('industry', 'N/A')}\n"
                    market_context += f"52w Range: ${info.get('fiftyTwoWeekLow', 0):.2f}-${info.get('fiftyTwoWeekHigh', 0):.2f}\n"
                    market_context += f"Beta: {info.get('beta', 'N/A')}, P/E: {info.get('trailingPE', 'N/A')}\n"
                    market_context += f"Avg Volume: {info.get('averageVolume', 0):,}, Shares Outstanding: {info.get('sharesOutstanding', 0):,}\n"
                    market_context += f"Business Summary: {info.get('longBusinessSummary', 'N/A')}\n\n"
                    
                    # Technical Analysis
                    hist = stock.history(period=self.current_period)
                    
                    if not hist.empty:
                        # Calculate simplified technical indicators
                        trend = hist['Close'].pct_change().mean() * 100
                        volatility = (hist['High'] - hist['Low']).mean()
                        volume_trend = ((hist['Volume'].iloc[-1] / hist['Volume'].mean()) - 1) * 100
                        price_range = f"${hist['Low'].min():.2f}-${hist['High'].max():.2f}"
                        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
                        ma50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None

                        market_context += "Technical Analysis:\n"
                        market_context += f"Price Trend: {trend:.1f}%\n"
                        market_context += f"Volatility: ${volatility:.2f}\n"
                        market_context += f"Volume Trend: {volume_trend:.1f}%\n"
                        market_context += f"Trading Range: {price_range}\n"
                        if ma50:
                            market_context += f"MA20/MA50: ${ma20:.2f}/${ma50:.2f}\n"
                        else:
                            market_context += f"MA20: ${ma20:.2f}\n"
                        market_context += "\n"

                        # Price action table
                        market_context += "Price Action:\n"
                        market_context += "| Date | Close | Change % | Volume |\n"
                        market_context += "|------|--------|----------|----------|\n"

                        # Sample key points from history
                        sample_indices = [0, -1]  # Just first and last points
                        if len(hist) > 2:
                            mid = len(hist) // 2
                            sample_indices.insert(1, mid)
                        
                        for idx in sample_indices:
                            row = hist.iloc[idx]
                            date = hist.index[idx]
                            prev_close = hist['Close'].iloc[idx-1] if idx > 0 else row['Close']
                            pct_change = ((row['Close'] - prev_close) / prev_close) * 100
                            vol_fmt = f"{int(row['Volume']):,}"
                            
                            market_context += (
                                f"| {date.strftime('%Y-%m-%d')} "
                                f"| ${row['Close']:.2f} "
                                f"| {pct_change:+.1f}% "
                                f"| {vol_fmt} |\n"
                            )
                        market_context += "\n"
                
                # Other Stocks Context
                market_context += "Related Market Activity:\n"
                for ticker, data in self.stock_data.items():
                    if ticker != current_ticker:
                        market_context += f"{data['name']} ({ticker}): ${data['price']:.2f} ({data['change']}%), "
                market_context += "\n\n"
                
                # Currency Markets
                if self.currency_data:
                    market_context += "Currency Markets:\n"
                    for pair, data in self.currency_data.items():
                        market_context += f"{pair}: {data['rate']:.4f} ({data['change_pct']:+.2f}%), "
                market_context += "\n\n"
                
                # Recent Market News (Mode: {self.news_context_mode.value})
                market_context += f"Recent Market News (Mode: {self.news_context_mode.value}):\n"
                seen_titles = set()
                
                # Use consolidated news item getter
                items, _, _ = self.get_news_items()
                filtered_news = items[:8] if self.search_mode else self.filter_news_for_context()

                # Process news items for context
                for news in filtered_news:
                    if news['title'] not in seen_titles:
                        seen_titles.add(news['title'])
                        # Validate and set default summary
                        if news.get('summary') is None:
                            news['summary'] = news.get('description', '')
                            if not news['summary']:
                                news['summary'] = news.get('content:encoded', '')[:500]
                            if not news['summary']:
                                news['summary'] = "No description available"

                        # Set default source
                        if news.get('source') is None:
                            news['source'] = "Unknown Source"

                        # Set default published date
                        if news.get('published') is None:
                            news['published'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')

                        try:
                            date = datetime.strptime(news['published'], 
                                                   '%a, %d %b %Y %H:%M:%S %z').strftime('%Y-%m-%d %H:%M:%S')
                        except (ValueError, TypeError):
                            date = news['published'][:19]
                        
                        # Build context with validated fields
                        market_context += f"[{date}] {news['title']}"
                        #market_context += f"Source: {news['source']}\n"
                        market_context += f" - {news['summary']}\n"
                        
                        # Add optional enrichment fields if available
                        if news.get('tags'):
                            market_context += f"Tags: {', '.join(news['tags'])}\n"
                        if news.get('category'):
                            market_context += f"Category: {news['category']}\n"
                            
                        market_context += "\n"
                
                focus_ticker = current_ticker + " " + self.stock_data[current_ticker]['name']

                market_context += MARKET_CONTEXT_PROMPT.format(focus_ticker=focus_ticker)                
                
                # Prepare request data for logging
                request_data = {
                    "system_prompt": SYSTEM_PROMPT,
                    "market_context": market_context,
                    "ticker": current_ticker,
                    "model": MODEL_NAME,
                    "schema": MarketAnalysis.model_json_schema()
                }
                
                response = chat(
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': market_context}
                    ],
                    model=MODEL_NAME,
                    format=MarketAnalysis.model_json_schema(),
                )
                
                analysis = MarketAnalysis.model_validate_json(response.message.content)
                self.ai_messages.append(analysis)
                if len(self.ai_messages) > 10:
                    self.ai_messages.pop(0)
                
                # Update sentiment tracking
                current_ticker = list(self.terminal.tickers)[self.current_ticker_idx]
                self.stock_sentiments[current_ticker] = analysis.sentiment
                self.sentiment_tracker.add_sentiment(
                    current_ticker,
                    analysis.sentiment,
                    analysis.confidence_level
                )
                
                # Log successful interaction
                self.log_api_interaction(
                    request=request_data,
                    response={
                        "raw_response": response.message.content,
                        "validated_analysis": analysis.model_dump()
                    }
                )
                
                return analysis
                
            except Exception as e:
                retries -= 1
                self.log_api_interaction(
                    request=request_data, 
                    response=None,
                    error=str(e)
                )
                if retries == 0:
                    raise
                time.sleep(1)

    def get_news_items(self) -> tuple[List[Dict], str, str]:
        """Get current news items for DISPLAY only, with normalized content"""
        # For display, we either show search results OR news items
        base_items = self.search_results if self.search_mode else sorted(
            self.news_items, 
            key=lambda x: x['published'], 
            reverse=True
        )
        
        # Normalize content for display
        items = [{
            **item, 
            'content': (item.get('summary') or item.get('description') or item.get('content:encoded', ''))[:128]
        } for item in base_items]
        
        return (
            items,
            f"Search Results: {self.search_query}" if self.search_mode else "Latest News",
            "yellow" if self.search_mode else "orange3"
        )

    def calculate_item_height(self, item: Dict, width: int) -> int:
        """Calculate display height needed for a news item based on content and terminal width"""
        # Account for padding and borders
        content_width = width - 4  # Adjust for panel borders and padding
        
        # Calculate lines needed for each component
        timestamp = len(f"[{self.format_timestamp(item['published'])}] ")
        title = len(item['title'])
        
        content = (item.get('summary') or 
                  item.get('description') or 
                  item.get('content:encoded', ''))[:128]
        
        source = len(f" - {item['source']}")
        
        # Calculate wrapped lines (including partial lines)
        timestamp_lines = (timestamp + content_width - 1) // content_width
        title_lines = (title + content_width - 1) // content_width
        content_lines = (len(content) + content_width - 1) // content_width
        source_lines = (source + content_width - 1) // content_width
        
        # Add spacing line
        return timestamp_lines + title_lines + content_lines + source_lines + 1

    def get_visible_window(self, items: List[Dict]) -> tuple[List[Dict], int, int]:
        """Get visible items and scroll bounds based on actual content heights"""
        if not items:
            return [], 0, 0
        
        # Get available height for content
        available_height = self.console.height - 6  # Account for borders and status
        width = self.console.width - 4  # Account for panel borders
        
        # Calculate cumulative heights
        item_heights = [self.calculate_item_height(item, width) for item in items]
        cumulative_height = 0
        visible_count = 0
        
        # Find how many items fit
        for height in item_heights[self.news_scroll:]:
            if cumulative_height + height > available_height:
                break
            cumulative_height += height
            visible_count += 1
        
        # Calculate max scroll position
        total_items = len(items)
        start = self.news_scroll
        
        # Adjust scroll position based on mode
        if self.scroll_mode == "infinite":
            start = start % total_items
        else:
            # Find last valid start position that shows at least one item
            max_start = total_items - 1
            while max_start > 0:
                test_height = item_heights[max_start]
                if test_height <= available_height:
                    break
                max_start -= 1
            start = min(max(0, start), max_start)
        
        end = min(start + visible_count, total_items)
        return items[start:end], start, end

    def generate_layout(self) -> Layout:
        """Generate the current layout for Live rendering."""
        base_layout = Layout()
        
        # Main vertical split
        base_layout.split_column(
            Layout(name="upper", ratio=2),
            Layout(name="lower", ratio=1),
            Layout(name="status", size=3)
        )
        
        # Split upper into analysis and market data
        base_layout["upper"].split_row(
            Layout(name="ai_analysis"),  # Moved to left side
            Layout(name="market_data", ratio=1)
        )
        
        # Split market data vertically
        base_layout["market_data"].split_column(
            Layout(name="market_indicators", ratio=2),
            Layout(name="news", ratio=3)  # News moved here
        )
        
        # Split market indicators horizontally for stocks and currencies
        base_layout["market_indicators"].split_row(
            Layout(name="stocks", ratio=3),
            Layout(name="currencies", ratio=2)
        )

        # News panel with consolidated state management
        news_text = Text()
        items, panel_title, border_style = self.get_news_items()
        visible_items, start, end = self.get_visible_window(items)

        # Display items with standard format
        for i, item in enumerate(visible_items, start + 1):
            t_str = self.format_timestamp(item['published'])
            news_text.append(f"[{t_str}] ", style="yellow")
            news_text.append(f"{item['title']} ", style="bold")
            
            content = (item.get('summary')[:128] or 
                      item.get('description')[:128] or 
                      item.get('content:encoded')[:128])
            
            if content:
                news_text.append(f"{content}", style="dim")
            news_text.append(" - ")
            news_text.append(f"{item['source']}\n", style="orange3")
            news_text.append("")

        base_layout["news"].update(Panel(
            news_text,
            title=panel_title,
            subtitle=f"{self.SCROLL_INDICATORS[self.scroll_mode]} | {start+1}-{len(items)}",
            border_style=border_style
        ))

        # Stocks panel with sentiment
        stock_table = Table(box=box.SIMPLE, show_header=True, style="orange3", header_style="bold orange3", padding=(0,1))
        stock_table.add_column("Ticker", justify="left")
        stock_table.add_column("Price", justify="right")
        stock_table.add_column("Change", justify="right")
        stock_table.add_column("Volume", justify="right")
        stock_table.add_column("Cap", justify="right")
        stock_table.add_column("Sentiment", justify="center")  # New column
        
        for ticker, data in self.stock_data.items():
            change_color = "bright_green" if data['change'] >= 0 else "red"
            price = f"${float(data['price']):.2f}"
            change_value = float(data['change'])
            change = f"{change_value:+.2f}"
            price_arrow = TrendArrows.get_price_arrow(change_value)
            volume = f"{int(data['volume']):,}"
            market_cap = self.format_market_cap(data['market_cap'])
            
            # Get averaged sentiment with confidence and trend
            sentiment_data = self.sentiment_tracker.get_average_sentiment(ticker)
            if sentiment_data:
                sentiment, confidence, raw_value = sentiment_data
                sentiment_color = {
                    MarketSentiment.BULLISH: "orange3",
                    MarketSentiment.BEARISH: "red",
                    MarketSentiment.HAWKISH: "yellow",
                    MarketSentiment.CAUTIOUS: "yellow",
                    MarketSentiment.NEUTRAL: "blue"
                }.get(sentiment, "orange3")
                sentiment_arrow = TrendArrows.get_sentiment_arrow(raw_value)
                sentiment_text = f"[{sentiment_color}]{sentiment.value} {sentiment_arrow} ({confidence}%)[/{sentiment_color}]"
            else:
                sentiment_text = ""
            
            stock_table.add_row(
                ticker, 
                f"{price} {price_arrow}",
                f"[{change_color}]{change}[/{change_color}]",
                volume, 
                market_cap,
                sentiment_text
            )
            
        base_layout["stocks"].update(Panel(
            stock_table,
            title="Stock Prices",
            subtitle=f"Updated: {self.last_update.strftime('%H:%M:%S')}",
            border_style="orange3"
        ))

        # Currency and Crypto panel
        currency_table = Table(box=box.SIMPLE, show_header=True, style="orange3", header_style="bold orange3", padding=(0,1))
        currency_table.add_column("Pair", justify="left")
        currency_table.add_column("Rate", justify="right")
        currency_table.add_column("Change", justify="right")

        # Add traditional currency pairs
        if self.currency_data:
            for pair, data in self.currency_data.items():
                change_color = "bright_green" if data['change'] >= 0 else "red"
                rate = f"{float(data['rate']):.4f}"
                change_value = float(data['change_pct'])
                change = f"{change_value:+.2f}%"
                trend_arrow = TrendArrows.get_price_arrow(change_value)
                currency_table.add_row(
                    pair, 
                    f"{rate} {trend_arrow}",
                    f"[{change_color}]{change}[/{change_color}]"
                )

        # Add cryptocurrency pairs if configured
        if self.terminal.crypto and self.crypto_data:
            for pair, data in self.crypto_data.items():
                change_color = "bright_green" if data['change'] >= 0 else "red"
                rate = f"{float(data['price']):.4f}"  # Display as rate instead of price
                change_value = float(data['change'])
                change_pct = (change_value / (data['price'] - change_value)) * 100
                change = f"{change_pct:+.2f}%"
                trend_arrow = TrendArrows.get_price_arrow(change_pct)
                currency_table.add_row(
                    pair,
                    f"{rate} {trend_arrow}",  # Use rate format consistent with forex
                    f"[{change_color}]{change}[/{change_color}]"
                )

        base_layout["currencies"].update(Panel(
            currency_table,
            title="Currency Rates",
            subtitle=f"Updated: {self.last_update.strftime('%H:%M:%S')}",
            border_style="orange3"
        ))

        # AI Analysis Panel with loading state
        ai_content = Text()
        if self.ai_error:
            self.panel_states["ai_analysis"] = LoadingState.ERROR
            ai_content.append(f"Error: {self.ai_error}\n", style="bold red")
        elif self.ai_response:
            self.panel_states["ai_analysis"] = LoadingState.READY
            schema = MarketAnalysis.model_json_schema()
            
            # Get field order and descriptions from schema
            for field_name, field_props in schema['properties'].items():
                # Skip the thinking field
                if field_name == 'thinking':
                    continue
                    
                field_value = getattr(self.ai_response, field_name)
                title = field_props.get('title', field_name.replace('_', ' ').title())
                
                ai_content.append(f"{title}: ", style="bold orange3")
                
                if field_name == 'sentiment':
                    sentiment_color = {
                        MarketSentiment.BULLISH: "bright_green",
                        MarketSentiment.BEARISH: "red",
                        MarketSentiment.HAWKISH: "yellow", 
                        MarketSentiment.CAUTIOUS: "yellow",
                        MarketSentiment.NEUTRAL: "blue"
                    }.get(field_value, "orange3")
                    confidence = getattr(self.ai_response, 'confidence_level')
                    ai_content.append(f"{field_value.value} (Confidence: {confidence}/5)\n\n", style=sentiment_color)
                    
                elif isinstance(field_value, list):
                    ai_content.append("\n")
                    for item in field_value:
                        ai_content.append(f"• {item}\n")
                    ai_content.append("\n")
                else:
                    ai_content.append(f"{field_value}\n\n")
        else:
            self.panel_states["ai_analysis"] = LoadingState.LOADING

        base_layout["ai_analysis"].update(
            self.create_panel_with_loading(
                ai_content,
                title=f"{MarketAnalysis.__doc__} | Context: {self.news_context_mode.value}",
                state=self.panel_states["ai_analysis"],
                loading_message=""
            )
        )

        # Lower panel: Chart view with loading state
        tickers = list(self.terminal.tickers)
        chart_content = ""
        current_ticker = ""
        if tickers:
            current_ticker = tickers[self.current_ticker_idx]
            if current_ticker in self.charts[self.current_chart_type]:
                self.panel_states["chart"] = LoadingState.READY
                chart_content = self.charts[self.current_chart_type][current_ticker]
            else:
                self.panel_states["chart"] = LoadingState.LOADING

        chart_title = f"{current_ticker} {self.current_chart_type.upper()} ({self.current_period})" if tickers else "Chart"
        base_layout["lower"].update(
            self.create_panel_with_loading(
                chart_content,
                title=chart_title,
                state=self.panel_states["chart"],
                loading_message="Generating chart...",
                height=int(self.console.height * self.CHART_HEIGHT_RATIO)
            )
        )

        # Update status panel to include chart type cycling
        status = Text(style="orange3")
        status.append(f"Last update: {self.last_update.strftime('%H:%M:%S')} | ")
        status.append("Commands: (F)ocus  (P)eriod  (C)hart  (M)odulo (A)wareness (S)earch (Q)uit")
        if self.input_mode:
            status.append(" | Input: " + self.command_str, style="bold yellow")
        base_layout["status"].update(Panel(status, border_style="orange3"))
        return base_layout

    def update_charts_for_period(self):
        """Update all chart types for the current period"""
        width, height = self.console.size
        chart_height = max(int(height * self.CHART_HEIGHT_RATIO) - self.PADDING['vertical'], self.MIN_CHART_HEIGHT)
        chart_width = max(width - self.PADDING['horizontal'], self.MIN_CHART_WIDTH)
        
        for ticker in self.terminal.tickers:
            # Price history chart
            history = self.terminal.get_stock_history(
                ticker,
                self.current_period,
                width=chart_width,
                height=chart_height
            )
            if history:
                self.charts["price"][ticker] = history['chart']

            # Volume profile
            volume = self.terminal.get_volume_profile(
                ticker,
                self.current_period,
                width=chart_width,
                height=chart_height
            )
            if volume:
                self.charts["volume"][ticker] = volume['chart']

            # RSI chart
            rsi = self.terminal.get_rsi_chart(
                ticker,
                self.current_period,
                width=chart_width,
                height=chart_height
            )
            if rsi:
                self.charts["rsi"][ticker] = rsi['chart']

            # MACD chart
            macd = self.terminal.get_macd_chart(
                ticker,
                self.current_period,
                width=chart_width,
                height=chart_height
            )
            if macd:
                self.charts["macd"][ticker] = macd['chart']

    def filter_news_for_context(self, limit: int = 8) -> List[Dict]:
        """Filter news items for AI CONTEXT only"""
        # Get awareness items first (unchanged)
        awareness_items = []
        if self.news_context_mode == NewsContextMode.RECENT:
            awareness_items = [{**item, 'context': "Recent News:"} for item in sorted(self.news_items, key=lambda x: x['published'], reverse=True)[:limit]]
        elif self.news_context_mode == NewsContextMode.RANDOM:
            awareness_items = [{**item, 'context': "Random News:"} for item in random.sample(self.news_items, min(limit, len(self.news_items)))]
        elif self.news_context_mode == NewsContextMode.RELEVANT:
            try:
                current_ticker = list(self.terminal.tickers)[self.current_ticker_idx]
                stock_info = self.stock_data.get(current_ticker, {})
                
                search_terms = [
                    (current_ticker, self.awareness_config.weights["ticker"]),
                    (stock_info.get('name', ''), self.awareness_config.weights["name"])
                ]
                
                try:
                    yf_info = yf.Ticker(current_ticker).info
                    if yf_info.get('sector'):
                        search_terms.append((yf_info['sector'], self.awareness_config.weights["sector"]))
                    if yf_info.get('industry'):
                        search_terms.append((yf_info['industry'], self.awareness_config.weights["industry"]))
                except Exception:
                    pass
                
                scored_news = []
                for news in self.news_items:
                    max_score = 0
                    for term, weight in search_terms:
                        if not term:
                            continue
                        try:
                            title_score = fuzz.partial_ratio(str(term).lower(), str(news['title']).lower()) * weight
                            summary = news.get('summary', '')
                            summary_score = fuzz.partial_ratio(str(term).lower(), str(summary).lower()) * weight
                            max_score = max(max_score, title_score, summary_score)
                        except Exception:
                            continue
                    
                    if max_score > self.awareness_config.relevance_threshold:
                        scored_news.append((news, max_score))
                
                awareness_items = [{**news, 'context': f"Relevant to {current_ticker}:"} for news, _ in sorted(scored_news, key=lambda x: (x[1], x[0]['published']), reverse=True)[:limit]]
            except Exception:
                awareness_items = [{**item, 'context': "Recent News (Fallback):"} for item in sorted(self.news_items, key=lambda x: x['published'], reverse=True)[:limit]]
        else:
            awareness_items = []

        # If searching, add search results to context WITHOUT removing awareness items
        if self.search_mode and self.search_results:
            search_items = [{**item, 'context': f"Search Results for '{self.search_query}':"} for item in self.search_results]
            # Combine both lists while respecting limit
            all_items = search_items + awareness_items
            # Remove duplicates while preserving order
            seen = set()
            filtered = []
            for item in all_items:
                if item['title'] not in seen and len(filtered) < limit:
                    filtered.append(item)
                    seen.add(item['title'])
            return filtered

        return awareness_items[:limit]

    def get_scroll_bounds(self) -> tuple[int, int]:
        """Get current items_per_page and max_scroll"""
        items_per_page = max(5, self.console.height // 4)
        total_items = len(self.search_results if self.search_mode else self.news_items)
        max_scroll = max(0, total_items - items_per_page)
        return items_per_page, max_scroll

    def handle_auto_scroll(self, total_items: int, scroll_pos: int) -> int:
        """Handle automatic scrolling with validated bounds"""
        current_time = time.time()
        if current_time - self.last_scroll_time < self.scroll_interval:
            return scroll_pos
        
        # Update last scroll time when we actually scroll
        self.last_scroll_time = current_time
        
        _, max_scroll = self.get_scroll_bounds()
        
        if self.scroll_mode == "static":
            return scroll_pos  # Only manual scrolling
        elif self.scroll_mode == "infinite":
            # Keep within bounds while maintaining sort order
            return min(scroll_pos + 1, max_scroll)
        elif self.scroll_mode == "pingpong":
            new_scroll = scroll_pos + self.scroll_direction
            if new_scroll > max_scroll:
                self.scroll_direction = -1
                return max_scroll
            elif new_scroll < 0:
                self.scroll_direction = 1
                return 0
            return new_scroll
        return scroll_pos

    def create_panel_with_loading(self, content: Any, title: str, state: LoadingState, 
                                loading_message: str = "", **panel_kwargs) -> Panel:
        """Create a panel with loading overlay that respects panel structure"""
        # Get panel dimensions from kwargs or calculate defaults
        panel_width = panel_kwargs.get('width', self.console.width - 4)
        panel_height = panel_kwargs.get('height', self.console.height // 3)
        
        if state == LoadingState.LOADING:
            # For analysis panel, use a simpler centered loading display
            if "analysis" in title.lower():
                loading_content = Text()
                
                # Center the loading animation
                frame = self.loading_animation.get_frame()
                loading_content.append(Text(frame, justify="center", style="bold orange3"))
                
                # Center the message if provided
                if loading_message:
                    msg_padding = " " * ((panel_width - len(loading_message)) // 2)
                    loading_content.append(f"\n{msg_padding}{loading_message}", style="orange3")
                
                return Panel(
                    loading_content,
                    title=title,
                    border_style="orange3",
                    **panel_kwargs
                )
            
            # For chart panel, use the full ASCII art animation
            else:
                loading_content = self.loading_animation.position_in_panel(
                    panel_width - 4,  # Account for panel borders
                    panel_height - 2   # Account for panel borders
                )
                
                if loading_message:
                    msg_padding = ' ' * ((panel_width - len(loading_message)) // 2)
                    loading_content.append(f"\n{msg_padding}{loading_message}", style="orange3")
                
                return Panel(
                    loading_content,
                    title=title,
                    border_style="orange3",
                    box=box.DOUBLE,
                    **panel_kwargs
                )
        
        # Return normal panel if not loading
        return Panel(content, title=title, border_style="orange3", **panel_kwargs)

    def run(self):
        update_thread = threading.Thread(target=self.update_data, daemon=True)
        update_thread.start()

        with Live(self.generate_layout(), 
                 refresh_per_second=self.refresh_config.screen_fps, 
                 screen=True) as live:
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    # ESC cancels search or input mode.
                    if key == b'\x1b':
                        if self.search_mode:
                            self.search_mode = False
                            self.search_query = ""
                            self.search_results = []
                        if self.input_mode:
                            self.input_mode = False
                            self.command_str = ""
                        continue

                    # If in input mode, capture keystrokes for the command.
                    if self.input_mode:
                        if key == b'\r':  # Enter key: process command
                            cmd = self.command_str.strip().lower()
                            # Process quit command
                            if cmd in ("q", "quit", "exit"):
                                break
                            # Process next chart command
                            elif cmd == "n":
                                self.current_ticker_idx = (self.current_ticker_idx + 1) % len(self.terminal.tickers)
                            # Process period change command
                            elif cmd.startswith("p"):
                                current_idx = self.AVAILABLE_PERIODS.index(self.current_period)
                                self.current_period = self.AVAILABLE_PERIODS[(current_idx + 1) % len(self.AVAILABLE_PERIODS)]
                                # Update all charts with new period
                                self.update_charts_for_period()
                            # Process search command (e.g., "s apple")
                            elif cmd.startswith("s "):
                                term = cmd[2:].strip()
                                if term:
                                    self.search_mode = True
                                    self.search_query = term
                                    self.search_results = self.search(term)
                            # Exit input mode after processing
                            self.input_mode = False
                            self.command_str = ""
                        elif key in (b'\x08', b'\x7f'):  # Backspace
                            self.command_str = self.command_str[:-1]
                        else:
                            try:
                                ch = key.decode('utf-8')
                                if ch.isprintable():
                                    self.command_str += ch
                            except UnicodeDecodeError:
                                pass
                    else:
                        # Not in input mode, process single-key commands
                        try:
                            key_char = key.decode('utf-8').lower()
                        except UnicodeDecodeError:
                            continue
                        if key_char == 'q':
                            break
                        elif key_char == 'f':
                            self.current_ticker_idx = (self.current_ticker_idx + 1) % len(self.terminal.tickers)
                        elif key_char == 'c':  # Add chart type cycling
                            current_idx = self.CHART_TYPES.index(self.current_chart_type)
                            self.current_chart_type = self.CHART_TYPES[(current_idx + 1) % len(self.CHART_TYPES)]
                        
                        #elif key_char == 'i':
                        #    self.input_mode = True
                        #    self.command_str = ""
                        elif key_char == 's':
                            # Start input mode with "s " prefilled for search.
                            self.input_mode = True
                            self.command_str = "s "
                        elif key_char == 'm':  # New command to cycle scroll modes
                            current_idx = self.SCROLL_MODES.index(self.scroll_mode)
                            self.scroll_mode = self.SCROLL_MODES[(current_idx + 1) % len(self.SCROLL_MODES)]
                        elif key_char == ' ' and self.scroll_mode == "static":
                            # In static mode, space advances one item
                            items_length = len(self.search_results if self.search_mode else self.news_items)
                            self.news_scroll = (self.news_scroll + 1) % items_length
                        elif key_char == 'p':
                            current_idx = self.AVAILABLE_PERIODS.index(self.current_period)
                            self.current_period = self.AVAILABLE_PERIODS[(current_idx + 1) % len(self.AVAILABLE_PERIODS)]
                            # Update all charts with new period
                            self.update_charts_for_period()
                        elif key_char == 'a':  # Cycle news context mode
                            modes = list(NewsContextMode)
                            current_idx = modes.index(self.news_context_mode)
                            self.news_context_mode = modes[(current_idx + 1) % len(modes)]
                            # Force immediate analysis update with new context
                            self.analysis_manager.request_analysis(self.get_market_analysis)

                # Use single scroll handler with total items only
                items_length = len(self.search_results if self.search_mode else self.news_items)
                self.news_scroll = self.handle_auto_scroll(items_length, self.news_scroll)

                # Check for analysis results
                result = self.analysis_manager.check_results()
                if result:
                    status, data = result
                    if status == "success":
                        self.ai_response = data
                        self.ai_error = None
                    else:  # error
                        self.ai_error = data
                        self.ai_response = None

                # Request new analysis when data updates
                if not self.update_queue.empty():
                    try:
                        self.update_queue.get_nowait()
                        # Request analysis asynchronously
                        self.analysis_manager.request_analysis(self.get_market_analysis)
                    except queue.Empty:
                        pass

                live.update(self.generate_layout())
                time.sleep(1.0 / self.refresh_config.screen_fps)  # This is running at screen refresh rate

if __name__ == '__main__':
    dashboard = FinancialDashboard()
    dashboard.run()
