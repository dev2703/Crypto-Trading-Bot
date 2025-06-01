"""
Real-time Data Streaming Module
- Implements WebSocket connection for live market data
- Handles real-time price updates and order book changes
- Manages connection state and error handling
"""
import json
import websocket
import threading
import time
from typing import Callable, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeStreamer:
    """Real-time market data streamer using WebSocket."""
    
    def __init__(self, api_key: str):
        """
        Initialize the real-time streamer.
        
        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self.ws = None
        self.is_connected = False
        self.callbacks = {
            'price': [],
            'orderbook': [],
            'trade': []
        }
        self.symbols = set()
    
    def connect(self) -> None:
        """Establish WebSocket connection."""
        try:
            # For demonstration, we'll use Alpha Vantage's WebSocket endpoint
            # In production, you might want to use a different provider with better WebSocket support
            websocket.enableTrace(True)
            self.ws = websocket.WebSocketApp(
                "wss://stream.alphavantage.co/v2/stream",
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Start WebSocket connection in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection to establish
            timeout = 10
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self.is_connected:
                raise ConnectionError("Failed to establish WebSocket connection")
                
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            raise
    
    def subscribe(self, symbol: str, data_types: List[str] = ['price']) -> None:
        """
        Subscribe to real-time data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            data_types: List of data types to subscribe to ('price', 'orderbook', 'trade')
        """
        if not self.is_connected:
            raise ConnectionError("WebSocket not connected")
        
        self.symbols.add(symbol)
        subscription_msg = {
            "action": "subscribe",
            "symbols": symbol,
            "apikey": self.api_key
        }
        
        try:
            self.ws.send(json.dumps(subscription_msg))
            logger.info(f"Subscribed to {symbol} for {data_types}")
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {str(e)}")
            raise
    
    def add_callback(self, data_type: str, callback: Callable) -> None:
        """
        Add a callback function for a specific data type.
        
        Args:
            data_type: Type of data ('price', 'orderbook', 'trade')
            callback: Function to call when data is received
        """
        if data_type not in self.callbacks:
            raise ValueError(f"Invalid data type: {data_type}")
        self.callbacks[data_type].append(callback)
    
    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            data_type = data.get('type', 'price')
            
            # Call registered callbacks
            for callback in self.callbacks.get(data_type, []):
                callback(data)
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {str(error)}")
        self.is_connected = False
    
    def _on_close(self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str) -> None:
        """Handle WebSocket connection close."""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
    
    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket connection open."""
        logger.info("WebSocket connection established")
        self.is_connected = True
    
    def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            self.ws.close()
            self.is_connected = False
            logger.info("WebSocket connection closed")

# Example usage
if __name__ == "__main__":
    def price_callback(data: Dict) -> None:
        """Example callback for price updates."""
        print(f"Price update: {data}")
    
    # Initialize streamer
    streamer = RealTimeStreamer(api_key="YOUR_API_KEY")
    
    try:
        # Connect to WebSocket
        streamer.connect()
        
        # Add callback for price updates
        streamer.add_callback('price', price_callback)
        
        # Subscribe to AAPL price updates
        streamer.subscribe('AAPL', ['price'])
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        streamer.disconnect() 