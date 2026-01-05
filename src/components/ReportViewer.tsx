import React, { useState } from 'react';
import { useStore } from '../store/useStore';
import { format } from 'date-fns';

const ReportViewer: React.FC = () => {
  const { reports } = useStore();
  const [selectedReport, setSelectedReport] = useState<string | null>(null);

  const reportTypes = {
    premarket: { label: 'Premarket', time: '02:30 CT', color: '#4CAF50' },
    midday: { label: 'Midday', time: '12:00 CT', color: '#2196F3' },
    eod: { label: 'End of Day', time: '16:30 CT', color: '#FF9800' },
  };

  const sortedReports = [...reports].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  const selectedReportData = reports.find(r => r.id === selectedReport);

  return (
    <div className="report-viewer">
      <div className="report-list">
        {sortedReports.length === 0 ? (
          <div className="no-reports">No reports available</div>
        ) : (
          sortedReports.map((report) => {
            const typeInfo = reportTypes[report.type as keyof typeof reportTypes];
            return (
              <div
                key={report.id}
                className={`report-item ${selectedReport === report.id ? 'selected' : ''}`}
                onClick={() => setSelectedReport(report.id)}
              >
                <div className="report-header">
                  <span
                    className="report-type"
                    style={{ backgroundColor: typeInfo.color }}
                  >
                    {typeInfo.label}
                  </span>
                  <span className="report-date">
                    {format(new Date(report.timestamp), 'MMM dd, yyyy')}
                  </span>
                </div>
                <div className="report-meta">
                  <span className="signal-count">
                    {report.signals.length} signals
                  </span>
                  <span className="report-time">
                    {format(new Date(report.timestamp), 'HH:mm')}
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>

      {selectedReportData && (
        <div className="report-content">
          <div className="report-content-header">
            <h3>{reportTypes[selectedReportData.type as keyof typeof reportTypes].label} Report</h3>
            <span className="report-timestamp">
              {format(new Date(selectedReportData.timestamp), 'PPpp')}
            </span>
          </div>

          <div className="report-body">
            <pre>{selectedReportData.content}</pre>
          </div>

          {selectedReportData.signals.length > 0 && (
            <div className="report-signals">
              <h4>Associated Signals</h4>
              <div className="signal-summary">
                {selectedReportData.signals.map((signal) => (
                  <div key={signal.id} className="signal-summary-item">
                    <span className="signal-symbol">{signal.symbol}</span>
                    <span className={`signal-direction-${signal.direction}`}>
                      {signal.direction}
                    </span>
                    <span className="signal-confidence">
                      {(signal.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ReportViewer;