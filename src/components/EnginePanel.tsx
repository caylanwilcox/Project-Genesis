import React from 'react';
import { useStore } from '../store/useStore';

const EnginePanel: React.FC = () => {
  const { engines, updateEngine, setSelectedEngine, selectedEngine } = useStore();

  const coreEngines = engines.filter(e => e.type === 'core');
  const backgroundEngines = engines.filter(e => e.type === 'background');

  const handleToggleEngine = (engineId: string, currentActive: boolean) => {
    updateEngine(engineId, { active: !currentActive });
  };

  const handleSelectEngine = (engineId: string) => {
    setSelectedEngine(engineId === selectedEngine ? null : engineId);
  };

  const renderEngineGroup = (engineList: typeof engines, title: string) => (
    <div className="engine-group">
      <h3>{title}</h3>
      <div className="engine-list">
        {engineList.map((engine) => (
          <div
            key={engine.id}
            className={`engine-item ${selectedEngine === engine.id ? 'selected' : ''}`}
            onClick={() => handleSelectEngine(engine.id)}
          >
            <div className="engine-header">
              <span className="engine-name">{engine.name}</span>
              <label className="switch" onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={engine.active}
                  onChange={() => handleToggleEngine(engine.id, engine.active)}
                />
                <span className="slider"></span>
              </label>
            </div>
            <div className="engine-info">
              <div className="weight-bar">
                <div
                  className="weight-fill"
                  style={{ width: `${engine.weight * 100}%` }}
                />
              </div>
              <span className="weight-text">
                Weight: {(engine.weight * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="engine-panel">
      {renderEngineGroup(coreEngines, 'Core Engines')}
      {renderEngineGroup(backgroundEngines, 'Background Engines')}

      <div className="engine-stats">
        <div className="stat">
          <span className="stat-label">Active Engines:</span>
          <span className="stat-value">
            {engines.filter(e => e.active).length} / {engines.length}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Total Weight:</span>
          <span className="stat-value">
            {(engines.filter(e => e.active).reduce((sum, e) => sum + e.weight, 0) * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
};

export default EnginePanel;