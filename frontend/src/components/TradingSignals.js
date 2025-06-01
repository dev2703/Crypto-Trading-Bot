import React, { useState, useEffect } from 'react';
import axios from 'axios';

function TradingSignals() {
  const [signals, setSignals] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSignals = async () => {
      try {
        const response = await axios.get('/api/trading-signals');
        setSignals(response.data);
        setLoading(false);
      } catch (err) {
        setError('Error fetching trading signals');
        setLoading(false);
      }
    };

    fetchSignals();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Trading Signals</h1>
      {signals && (
        <div>
          <h2 className="text-xl mb-2">RSI Signal: {signals.rsi}</h2>
          <h2 className="text-xl mb-2">MACD Signal: {signals.macd}</h2>
          <h2 className="text-xl mb-2">MA Signal: {signals.ma}</h2>
          <h2 className="text-xl mb-2">BB Signal: {signals.bb}</h2>
        </div>
      )}
    </div>
  );
}

export default TradingSignals; 