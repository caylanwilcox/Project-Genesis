import React from 'react';
import { useStore } from '../store/useStore';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  Bar,
  BarChart,
  Line,
} from 'recharts';
import { ChartTooltip } from './ChartView/ChartTooltip';
import { ChartPlaceholder } from './ChartView/ChartPlaceholder';
import { useChartData } from './ChartView/useChartData';

const ChartView: React.FC = () => {
  const { marketData, selectedSymbol, signals } = useStore();
  const chartData = useChartData(marketData, selectedSymbol, signals);

  if (!selectedSymbol) {
    return <ChartPlaceholder message="Select a symbol to view chart" />;
  }

  if (chartData.length === 0) {
    return <ChartPlaceholder message={`No data available for ${selectedSymbol}`} />;
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
            <Tooltip content={<ChartTooltip />} />
            <Area type="monotone" dataKey="close" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} />
            <Line type="monotone" dataKey="high" stroke="#82ca9d" strokeWidth={1} dot={false} />
            <Line type="monotone" dataKey="low" stroke="#ff7777" strokeWidth={1} dot={false} />
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