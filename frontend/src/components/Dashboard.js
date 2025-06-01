import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './dashboard.css';

function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('/api/market-data');
        setData(response.data);
        setLoading(false);
      } catch (err) {
        setError('Error fetching market data. Please try again later.');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div className="flex justify-center items-center h-screen"><div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div></div>;
  if (error) return <div className="text-red-500 text-center p-4 bg-red-100 rounded-lg shadow-md">{error}</div>;

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-400 via-purple-300 to-pink-200 font-montserrat dashboard-bg">
      <div className="p-8 bg-white/80 rounded-3xl shadow-2xl max-w-md w-full border border-blue-200 backdrop-blur-md">
        <h1 className="text-4xl font-extrabold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-purple-500 to-pink-500 drop-shadow-lg tracking-tight">Market Data</h1>
        {data && (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold text-gray-800 flex items-center">Current Price: <span className="ml-2 text-green-600 font-bold text-3xl drop-shadow">${data.price}</span></h2>
            <h3 className="text-xl text-gray-700 flex items-center">24h Change: <span className={data.change >= 0 ? 'ml-2 text-green-600 font-bold' : 'ml-2 text-red-600 font-bold'}>{data.change}%</span></h3>
            <h3 className="text-xl text-gray-700 flex items-center">24h High: <span className="ml-2 text-green-500 font-semibold">${data.high}</span></h3>
            <h3 className="text-xl text-gray-700 flex items-center">24h Low: <span className="ml-2 text-red-500 font-semibold">${data.low}</span></h3>
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard; 