import React, { useState, useEffect } from 'react';
import axios from 'axios';

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
        setError('Error fetching market data');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Market Data</h1>
      {data && (
        <div>
          <h2 className="text-xl mb-2">Current Price: ${data.price}</h2>
          <h3 className="text-lg mb-2">24h Change: {data.change}%</h3>
          <h3 className="text-lg mb-2">24h High: ${data.high}</h3>
          <h3 className="text-lg mb-2">24h Low: ${data.low}</h3>
        </div>
      )}
    </div>
  );
}

export default Dashboard; 