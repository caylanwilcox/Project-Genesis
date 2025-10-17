import React from 'react';
import { useStore } from '../store/useStore';
import SignalList from './SignalList';
import EnginePanel from './EnginePanel';
import ChartView from './ChartView';
import ReportViewer from './ReportViewer';

const Dashboard: React.FC = () => {
  const { selectedSymbol, regime } = useStore();

  return (
    <div className="dashboard dashboard__root">
      <header className="dashboard-header dashboard__header">
        <h1 className="dashboard__title">MVP Trading System</h1>
        <div className="regime-indicator dashboard__regimeIndicator">
          <span className="dashboard__regimeLabel">Market Regime: </span>
          <span className={`regime-${regime} dashboard__regimeValue`}>{regime.toUpperCase()}</span>
        </div>
      </header>

      <div className="dashboard-grid dashboard__grid">
        <div className="panel engines-panel dashboard__panel dashboard__panel--engines">
          <h2 className="dashboard__panelTitle dashboard__panelTitle--engines">Trading Engines</h2>
          <div className="dashboard__enginePanelBody">
            <EnginePanel />
          </div>
        </div>

        <div className="panel chart-panel dashboard__panel dashboard__panel--chart">
          <h2 className="dashboard__panelTitle dashboard__panelTitle--chart">Market Chart {selectedSymbol && `- ${selectedSymbol}`}</h2>
          <div className="dashboard__chartPanelBody">
            <ChartView />
          </div>
        </div>

        <div className="panel signals-panel dashboard__panel dashboard__panel--signals">
          <h2 className="dashboard__panelTitle dashboard__panelTitle--signals">Active Signals</h2>
          <div className="dashboard__signalsPanelBody">
            <SignalList />
          </div>
        </div>

        <div className="panel reports-panel dashboard__panel dashboard__panel--reports">
          <h2 className="dashboard__panelTitle dashboard__panelTitle--reports">Reports</h2>
          <div className="dashboard__reportsPanelBody">
            <ReportViewer />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;