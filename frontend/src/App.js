import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import TradingSignals from './components/TradingSignals';
import PerformanceMetrics from './components/PerformanceMetrics';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="bg-gray-800 text-white p-4">
          <ul className="flex space-x-4">
            <li><Link to="/">Dashboard</Link></li>
            <li><Link to="/trading-signals">Trading Signals</Link></li>
            <li><Link to="/performance-metrics">Performance Metrics</Link></li>
          </ul>
        </nav>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/trading-signals" element={<TradingSignals />} />
          <Route path="/performance-metrics" element={<PerformanceMetrics />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 