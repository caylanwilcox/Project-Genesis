import React, { useMemo } from 'react';
import { useStore } from '../store/useStore';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
  Bar,
  BarChart,
} from 'recharts';
import { format } from 'date-fns';

const ChartView: React.FC = () => {
  const { marketData, selectedSymbol, signals } = useStore();

  const chartData = useMemo(() => {
    if (!selectedSymbol) return [];

    const data = marketData.get(selectedSymbol) || [];
    return data.map(d => ({
      ...d,
      date: format(new Date(d.date), 'MMM dd'),
      signals: signals.filter(
        s => s.symbol === selectedSymbol &&
        format(new Date(s.ts_emit), 'yyyy-MM-dd') === d.date
      ).length
    }));
  }, [marketData, selectedSymbol, signals]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="label">{`Date: ${label}`}</p>
          <p className="value">{`Open: $${payload[0]?.payload?.open?.toFixed(2)}`}</p>
          <p className="value">{`High: $${payload[0]?.payload?.high?.toFixed(2)}`}</p>
          <p className="value">{`Low: $${payload[0]?.payload?.low?.toFixed(2)}`}</p>
          <p className="value">{`Close: $${payload[0]?.payload?.close?.toFixed(2)}`}</p>
          <p className="value">{`Volume: ${payload[0]?.payload?.volume?.toLocaleString()}`}</p>
          {payload[0]?.payload?.signals > 0 && (
            <p className="signals">{`Signals: ${payload[0]?.payload?.signals}`}</p>
          )}
        </div>
      );
    }
    return null;
  };

  if (!selectedSymbol) {
    return (
      <div className="chart-placeholder">
        <p>Select a symbol to view chart</p>
      </div>
    );
  }

  if (chartData.length === 0) {
    return (
      <div className="chart-placeholder">
        <p>No data available for {selectedSymbol}</p>
      </div>
    );
  }

  return (
    <div className="chart-container">
      <div className="chart-section">
        <h4>Price Chart</h4>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="date" stroke="#888" />
            <YAxis stroke="#888" />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="close"
              stroke="#8884d8"
              fill="#8884d8"
              fillOpacity={0.3}
            />
            <Line
              type="monotone"
              dataKey="high"
              stroke="#82ca9d"
              strokeWidth={1}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="low"
              stroke="#ff7777"
              strokeWidth={1}
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-section">
        <h4>Volume</h4>
        <ResponsiveContainer width="100%" height={150}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="date" stroke="#888" />
            <YAxis stroke="#888" />
            <Tooltip />
            <Bar dataKey="volume" fill="#4CAF50" opacity={0.7} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {chartData.some(d => d.signals > 0) && (
        <div className="chart-section">
          <h4>Signal Activity</h4>
          <ResponsiveContainer width="100%" height={100}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="date" stroke="#888" />
              <YAxis stroke="#888" />
              <Tooltip />
              <Bar dataKey="signals" fill="#FF9800" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default ChartView;