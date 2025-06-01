import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './dashboard.css';

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

  if (loading) return <div className="flex justify-center items-center h-screen bg-gradient-to-br from-blue-400 via-purple-300 to-pink-200 font-montserrat dashboard-bg"><div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-pink-500"></div></div>;
  if (error) return <div className="text-red-500 text-center p-4 bg-red-100 rounded-lg shadow-md font-montserrat">{error}</div>;

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-400 via-purple-300 to-pink-200 font-montserrat dashboard-bg">
      <div className="p-8 bg-white/80 rounded-3xl shadow-2xl max-w-md w-full border border-purple-200 backdrop-blur-md">
        <h1 className="text-3xl font-extrabold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 via-pink-500 to-blue-500 drop-shadow-lg tracking-tight">Trading Signals</h1>
        {signals && (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold text-blue-700">RSI Signal: <span className="font-bold text-pink-600">{signals.rsi}</span></h2>
            <h2 className="text-xl font-semibold text-blue-700">MACD Signal: <span className="font-bold text-purple-600">{signals.macd}</span></h2>
            <h2 className="text-xl font-semibold text-blue-700">MA Signal: <span className="font-bold text-green-600">{signals.ma}</span></h2>
            <h2 className="text-xl font-semibold text-blue-700">BB Signal: <span className="font-bold text-yellow-500">{signals.bb}</span></h2>
          </div>
        )}
      </div>
    </div>
  );
}

export default TradingSignals; 