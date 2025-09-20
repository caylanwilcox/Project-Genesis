import React, { useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import { useStore } from './store/useStore';
import { mockSignals, mockEngines, mockMarketData, mockReports } from './services/mockData';

function App() {
  const { setSignals, setEngines, setMarketData, setReports, setSelectedSymbol } = useStore();

  useEffect(() => {
    setSignals(mockSignals);
    setEngines(mockEngines);
    setMarketData('AAPL', mockMarketData);
    setMarketData('TSLA', mockMarketData.map(d => ({
      ...d,
      open: d.open * 1.3,
      high: d.high * 1.3,
      low: d.low * 1.3,
      close: d.close * 1.3,
    })));
    setReports(mockReports);
    setSelectedSymbol('AAPL');
  }, [setSignals, setEngines, setMarketData, setReports, setSelectedSymbol]);

  return (
    <div className="App">
      <Dashboard />
    </div>
  );
}

export default App;
