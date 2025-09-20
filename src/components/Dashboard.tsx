import React from 'react';
import { useStore } from '../store/useStore';
import SignalList from './SignalList';
import EnginePanel from './EnginePanel';
import ChartView from './ChartView';
import ReportViewer from './ReportViewer';

const Dashboard: React.FC = () => {
  const { selectedSymbol, regime } = useStore();

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>MVP Trading System</h1>
        <div className="regime-indicator">
          <span>Market Regime: </span>
          <span className={`regime-${regime}`}>{regime.toUpperCase()}</span>
        </div>
      </header>

      <div className="dashboard-grid">
        <div className="panel engines-panel">
          <h2>Trading Engines</h2>
          <EnginePanel />
        </div>

        <div className="panel chart-panel">
          <h2>Market Chart {selectedSymbol && `- ${selectedSymbol}`}</h2>
          <ChartView />
        </div>

        <div className="panel signals-panel">
          <h2>Active Signals</h2>
          <SignalList />
        </div>

        <div className="panel reports-panel">
          <h2>Reports</h2>
          <ReportViewer />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;