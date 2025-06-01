import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import './dashboard.css';

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

  if (loading) return <div className="flex justify-center items-center h-screen bg-gradient-to-br from-blue-400 via-purple-300 to-pink-200 font-montserrat dashboard-bg"><div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div></div>;
  if (error) return <div className="text-red-500 text-center p-4 bg-red-100 rounded-lg shadow-md font-montserrat">{error}</div>;

  const chartData = {
    labels: metrics?.dates || [],
    datasets: [
      {
        label: 'Equity Curve',
        data: metrics?.equity || [],
        borderColor: 'rgb(236, 72, 153)', // pink-500
        backgroundColor: 'rgba(236, 72, 153, 0.1)',
        tension: 0.3,
        pointBackgroundColor: 'rgb(59, 130, 246)', // blue-500
        pointBorderColor: 'rgb(59, 130, 246)',
        pointRadius: 4,
      },
    ],
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-400 via-purple-300 to-pink-200 font-montserrat dashboard-bg">
      <div className="p-8 bg-white/80 rounded-3xl shadow-2xl max-w-lg w-full border border-pink-200 backdrop-blur-md">
        <h1 className="text-3xl font-extrabold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-pink-500 via-purple-500 to-blue-600 drop-shadow-lg tracking-tight">Performance Metrics</h1>
        {metrics && (
          <div>
            <h2 className="text-xl mb-2 font-semibold text-blue-700">Sharpe Ratio: <span className="font-bold text-pink-600">{metrics.sharpeRatio}</span></h2>
            <h2 className="text-xl mb-2 font-semibold text-blue-700">Drawdown: <span className="font-bold text-purple-600">{metrics.drawdown}%</span></h2>
            <h2 className="text-xl mb-2 font-semibold text-blue-700">Returns: <span className="font-bold text-green-600">{metrics.returns}%</span></h2>
            <div className="mt-6 bg-white/60 rounded-xl p-4 shadow-inner">
              <h3 className="text-lg mb-2 font-semibold text-gray-700">Equity Curve</h3>
              <Line data={chartData} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default PerformanceMetrics; 