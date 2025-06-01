import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function PerformanceMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await axios.get('/api/performance-metrics');
        setMetrics(response.data);
        setLoading(false);
      } catch (err) {
        setError('Error fetching performance metrics');
        setLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  const chartData = {
    labels: metrics?.dates || [],
    datasets: [
      {
        label: 'Equity Curve',
        data: metrics?.equity || [],
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Performance Metrics</h1>
      {metrics && (
        <div>
          <h2 className="text-xl mb-2">Sharpe Ratio: {metrics.sharpeRatio}</h2>
          <h2 className="text-xl mb-2">Drawdown: {metrics.drawdown}%</h2>
          <h2 className="text-xl mb-2">Returns: {metrics.returns}%</h2>
          <div className="mt-4">
            <h3 className="text-lg mb-2">Equity Curve</h3>
            <Line data={chartData} />
          </div>
        </div>
      )}
    </div>
  );
}

export default PerformanceMetrics; 