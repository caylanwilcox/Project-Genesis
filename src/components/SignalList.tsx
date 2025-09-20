import React, { useMemo } from 'react';
import { useStore } from '../store/useStore';
import { Signal } from '../types/Signal';
import { format } from 'date-fns';

const SignalList: React.FC = () => {
  const { signals, selectedSymbol, selectedEngine } = useStore();

  const filteredSignals = useMemo(() => {
    let filtered = [...signals];

    if (selectedSymbol) {
      filtered = filtered.filter(s => s.symbol === selectedSymbol);
    }

    if (selectedEngine) {
      filtered = filtered.filter(s => s.engine === selectedEngine);
    }

    return filtered.sort((a, b) =>
      new Date(b.ts_emit).getTime() - new Date(a.ts_emit).getTime()
    ).slice(0, 20);
  }, [signals, selectedSymbol, selectedEngine]);

  const getDirectionColor = (direction: Signal['direction']) => {
    switch (direction) {
      case 'long': return '#4CAF50';
      case 'short': return '#f44336';
      default: return '#9E9E9E';
    }
  };

  return (
    <div className="signal-list">
      {filteredSignals.length === 0 ? (
        <div className="no-signals">No signals available</div>
      ) : (
        <div className="signals-container">
          {filteredSignals.map((signal) => (
            <div key={signal.id} className="signal-card">
              <div className="signal-header">
                <span className="signal-symbol">{signal.symbol}</span>
                <span
                  className="signal-direction"
                  style={{ color: getDirectionColor(signal.direction) }}
                >
                  {signal.direction.toUpperCase()}
                </span>
              </div>

              <div className="signal-details">
                <div className="signal-row">
                  <span className="label">Engine:</span>
                  <span className="value">{signal.engine}</span>
                </div>
                <div className="signal-row">
                  <span className="label">Confidence:</span>
                  <span className="value">{(signal.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="signal-row">
                  <span className="label">Horizon:</span>
                  <span className="value">{signal.horizon}</span>
                </div>
                <div className="signal-row">
                  <span className="label">Time:</span>
                  <span className="value">
                    {format(new Date(signal.ts_emit), 'MMM dd HH:mm')}
                  </span>
                </div>
              </div>

              {signal.targets.length > 0 && (
                <div className="signal-targets">
                  <span className="targets-label">Targets:</span>
                  {signal.targets.map((target, idx) => (
                    <span key={idx} className="target">
                      ${target.tp.toFixed(2)} ({(target.prob * 100).toFixed(0)}%)
                    </span>
                  ))}
                </div>
              )}

              {signal.explain && (
                <div className="signal-explain">{signal.explain}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SignalList;