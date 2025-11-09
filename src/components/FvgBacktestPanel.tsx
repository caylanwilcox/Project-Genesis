'use client'

import React, { useState, useMemo } from 'react'
import { CandleData } from './ProfessionalChart/types'
import { backtestFvgPatterns, BacktestResults, FvgTradeResult } from '@/services/fvgBacktestService'
import { TrendingUp, TrendingDown, Target, Clock, Percent, BarChart3 } from 'lucide-react'

interface FvgBacktestPanelProps {
  data: CandleData[]
  timeframe?: string
  displayTimeframe?: string
  isOpen: boolean
  onClose: () => void
}

// Calculate default lookAheadBars based on timeframe
function getDefaultLookAheadBars(timeframe?: string): number {
  if (!timeframe) return 50

  // For intraday timeframes, look ahead fewer bars (shorter time window)
  if (timeframe === '1m' || timeframe === '5m') return 20 // ~20-100 minutes
  if (timeframe === '15m' || timeframe === '30m') return 30 // ~7-15 hours
  if (timeframe === '1h' || timeframe === '2h') return 50 // ~2-4 days
  if (timeframe === '4h') return 100 // ~2-3 weeks

  // For daily and above, look ahead more bars (longer time window)
  if (timeframe === '1d') return 100 // ~3-4 months
  if (timeframe === '1w') return 50 // ~1 year
  if (timeframe === '1M') return 24 // ~2 years

  return 50 // default
}

export function FvgBacktestPanel({ data, timeframe, displayTimeframe, isOpen, onClose }: FvgBacktestPanelProps) {
  const defaultLookAhead = getDefaultLookAheadBars(timeframe)
  const [lookAheadBars, setLookAheadBars] = useState(defaultLookAhead)
  const [entryStrategy, setEntryStrategy] = useState<'continuation' | 'immediate'>('continuation')
  const [marketHoursOnly, setMarketHoursOnly] = useState(false)

  // Update lookAheadBars when timeframe changes
  React.useEffect(() => {
    setLookAheadBars(getDefaultLookAheadBars(timeframe))
  }, [timeframe])

  // Run backtest
  const results: BacktestResults | null = useMemo(() => {
    if (!data || data.length < 100) return null
    return backtestFvgPatterns(data, { lookAheadBars, entryStrategy, marketHoursOnly })
  }, [data, lookAheadBars, entryStrategy, marketHoursOnly])

  if (!isOpen || !results) return null

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <BarChart3 size={20} className="text-blue-400" />
            <div>
              <h2 className="text-lg font-bold text-white">FVG Backtest Results</h2>
              {timeframe && (
                <div className="text-xs text-gray-500 mt-0.5">
                  Timeframe: {timeframe.toUpperCase()} {displayTimeframe && `(${displayTimeframe})`}
                </div>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-xl leading-none px-2"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="overflow-y-auto flex-1 p-6">
          {/* Market Hours Toggle */}
          <div className="mb-4 p-3 bg-gray-800/50 border border-gray-700 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Clock size={14} className="text-blue-400" />
                <span className="text-xs text-gray-400 font-semibold">Market Hours Filter</span>
              </div>
              <button
                onClick={() => setMarketHoursOnly(!marketHoursOnly)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  marketHoursOnly
                    ? 'bg-green-500/20 text-green-400 border border-green-500'
                    : 'bg-gray-700/50 text-gray-400 border border-gray-600'
                }`}
              >
                {marketHoursOnly ? 'ON (9:30am-4pm ET)' : 'OFF (All Hours)'}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {marketHoursOnly
                ? 'Analyzing only regular trading hours (9:30am - 4:00pm ET)'
                : 'Analyzing all hours including pre/post-market'
              }
            </p>
          </div>

          {/* Settings */}
          <div className="mb-6 space-y-4">
            {/* Entry Strategy Toggle */}
            <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
              <label className="block text-xs font-medium text-gray-400 mb-3">
                Entry Strategy
              </label>
              <div className="flex gap-3">
                <button
                  onClick={() => setEntryStrategy('continuation')}
                  className={`flex-1 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                    entryStrategy === 'continuation'
                      ? 'bg-blue-500/20 text-blue-400 border border-blue-500'
                      : 'bg-gray-700/50 text-gray-400 border border-gray-600 hover:bg-gray-700'
                  }`}
                >
                  Continuation
                  <div className="text-xs font-normal mt-1 opacity-80">Wait for retracement</div>
                </button>
                <button
                  onClick={() => setEntryStrategy('immediate')}
                  className={`flex-1 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                    entryStrategy === 'immediate'
                      ? 'bg-blue-500/20 text-blue-400 border border-blue-500'
                      : 'bg-gray-700/50 text-gray-400 border border-gray-600 hover:bg-gray-700'
                  }`}
                >
                  Immediate
                  <div className="text-xs font-normal mt-1 opacity-80">Enter next bar</div>
                </button>
              </div>
              <div className="text-xs text-gray-500 mt-2">
                {entryStrategy === 'continuation'
                  ? '✓ Waits for price to retrace to gap before entering'
                  : '✓ Enters immediately on next bar after FVG detection'
                }
              </div>
            </div>

            {/* Look-Ahead Bars */}
            <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
              <label className="block text-xs font-medium text-gray-400 mb-2">
                Look-Ahead Bars (how many bars to analyze after detection)
              </label>
              <input
                type="range"
                min="20"
                max="200"
                step="10"
                value={lookAheadBars}
                onChange={(e) => setLookAheadBars(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 mt-1">{lookAheadBars} bars</div>
            </div>
          </div>

          {/* Overall Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <StatCard
              label="Total Patterns"
              value={results.totalPatterns.toString()}
              icon={<Target size={16} />}
              color="blue"
            />
            <StatCard
              label="Win Rate"
              value={`${results.winRate.toFixed(1)}%`}
              icon={<Percent size={16} />}
              color={results.winRate >= 60 ? 'green' : results.winRate >= 50 ? 'yellow' : 'red'}
            />
            <StatCard
              label="Total Trades"
              value={results.totalTrades.toString()}
              subtitle={`${results.wins}W / ${results.losses}L`}
              icon={<BarChart3 size={16} />}
              color="gray"
            />
            <StatCard
              label="Entry Rate"
              value={`${((results.patternsWithEntry / results.totalPatterns) * 100).toFixed(1)}%`}
              subtitle={`${results.patternsWithEntry} entries`}
              icon={<Clock size={16} />}
              color="gray"
            />
          </div>

          {/* Target Hit Rates */}
          <div className="mb-6 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <Target size={14} className="text-blue-400" />
              Target Hit Rates
            </h3>
            <div className="space-y-2">
              <ProgressBar label="TP1 (0.5:1 R:R)" percentage={results.tp1HitRate} />
              <ProgressBar label="TP2 (1:1 R:R)" percentage={results.tp2HitRate} />
              <ProgressBar label="TP3 (2:1 R:R)" percentage={results.tp3HitRate} />
            </div>
          </div>

          {/* Performance by Type */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <TrendingUp size={14} className="text-green-400" />
                Bullish FVG
              </h3>
              <div className="text-2xl font-bold text-green-400">
                {results.bullishWinRate.toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">win rate</div>
            </div>

            <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <TrendingDown size={14} className="text-red-400" />
                Bearish FVG
              </h3>
              <div className="text-2xl font-bold text-red-400">
                {results.bearishWinRate.toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">win rate</div>
            </div>
          </div>

          {/* Performance by Confidence */}
          <div className="mb-6 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
            <h3 className="text-sm font-semibold text-white mb-3">Performance by Confidence Score</h3>
            <div className="space-y-3">
              <ConfidenceRow
                label="High Confidence (≥85%)"
                winRate={results.highConfidenceWinRate}
                color="green"
              />
              <ConfidenceRow
                label="Med Confidence (65-84%)"
                winRate={results.medConfidenceWinRate}
                color="yellow"
              />
              <ConfidenceRow
                label="Low Confidence (<65%)"
                winRate={results.lowConfidenceWinRate}
                color="red"
              />
            </div>
          </div>

          {/* Time Metrics */}
          <div className="grid grid-cols-3 gap-4">
            <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Avg Bars to Entry</div>
              <div className="text-lg font-bold text-white">{results.avgBarsToEntry.toFixed(1)}</div>
            </div>
            <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Avg Bars to Win</div>
              <div className="text-lg font-bold text-green-400">{results.avgBarsToWin.toFixed(1)}</div>
            </div>
            <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Avg Bars to Loss</div>
              <div className="text-lg font-bold text-red-400">{results.avgBarsToLoss.toFixed(1)}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Utility Components
function StatCard({
  label,
  value,
  subtitle,
  icon,
  color
}: {
  label: string
  value: string
  subtitle?: string
  icon: React.ReactNode
  color: 'blue' | 'green' | 'yellow' | 'red' | 'gray'
}) {
  const colorMap = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    red: 'text-red-400',
    gray: 'text-gray-400'
  }

  return (
    <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
      <div className="flex items-center gap-2 mb-2">
        <div className={colorMap[color]}>{icon}</div>
        <div className="text-xs text-gray-400">{label}</div>
      </div>
      <div className={`text-xl font-bold ${colorMap[color]}`}>{value}</div>
      {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
    </div>
  )
}

function ProgressBar({ label, percentage }: { label: string; percentage: number }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-400">{label}</span>
        <span className="text-xs font-semibold text-white">{percentage.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 transition-all duration-300"
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  )
}

function ConfidenceRow({
  label,
  winRate,
  color
}: {
  label: string
  winRate: number
  color: 'green' | 'yellow' | 'red'
}) {
  const colorMap = {
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500'
  }

  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-gray-400">{label}</span>
      <div className="flex items-center gap-2">
        <div className="h-2 w-32 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full ${colorMap[color]} transition-all duration-300`}
            style={{ width: `${Math.min(winRate, 100)}%` }}
          />
        </div>
        <span className="text-xs font-semibold text-white w-12 text-right">
          {winRate.toFixed(1)}%
        </span>
      </div>
    </div>
  )
}
