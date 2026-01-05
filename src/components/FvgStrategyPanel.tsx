'use client'

import React, { useMemo } from 'react'
import { CandleData } from './ProfessionalChart/types'
import {
  generateFvgSignals,
  getBestSignal,
  calculateStrategyStats,
  FvgStrategySignal,
  FvgStrategyConfig,
} from '@/services/fvgStrategyService'
import { TrendingUp, TrendingDown, Target, AlertCircle, CheckCircle2, XCircle, Activity } from 'lucide-react'

interface FvgStrategyPanelProps {
  data: CandleData[]
  timeframe?: string
  config?: Partial<FvgStrategyConfig>
}

export function FvgStrategyPanel({ data, timeframe, config }: FvgStrategyPanelProps) {
  // Generate signals
  const signals = useMemo(() => {
    if (!data || data.length < 30) return []
    return generateFvgSignals(data, config)
  }, [data, config])

  const bestSignal = useMemo(() => getBestSignal(signals), [signals])
  const stats = useMemo(() => calculateStrategyStats(signals), [signals])

  if (!data || data.length < 30) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="text-sm text-gray-500 text-center">
          Insufficient data for strategy analysis
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Strategy Stats Overview */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-4">
          <Activity size={18} className="text-blue-400" />
          <h3 className="text-sm font-semibold text-white">FVG Strategy Overview</h3>
          {timeframe && (
            <span className="text-xs text-gray-500">({timeframe.toUpperCase()})</span>
          )}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatBox
            label="Total Signals"
            value={stats.totalSignals.toString()}
            color="blue"
          />
          <StatBox
            label="High Quality"
            value={stats.highQuality.toString()}
            subtitle={`${stats.totalSignals > 0 ? ((stats.highQuality / stats.totalSignals) * 100).toFixed(0) : 0}%`}
            color="green"
          />
          <StatBox
            label="Avg Confidence"
            value={`${stats.avgConfidence.toFixed(0)}%`}
            color="purple"
          />
          <StatBox
            label="HTF Confirm"
            value={`${stats.htfConfirmationRate.toFixed(0)}%`}
            color="cyan"
          />
        </div>

        <div className="grid grid-cols-2 gap-3 mt-3">
          <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">Bullish Signals</span>
              <TrendingUp size={14} className="text-green-400" />
            </div>
            <div className="text-lg font-bold text-green-400 mt-1">{stats.bullishSignals}</div>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">Bearish Signals</span>
              <TrendingDown size={14} className="text-red-400" />
            </div>
            <div className="text-lg font-bold text-red-400 mt-1">{stats.bearishSignals}</div>
          </div>
        </div>
      </div>

      {/* Best Signal Display */}
      {bestSignal && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Target size={18} className="text-blue-400" />
              <h3 className="text-sm font-semibold text-white">Best Trading Signal</h3>
            </div>
            <QualityBadge signal={bestSignal} />
          </div>

          <SignalCard signal={bestSignal} />
        </div>
      )}

      {/* All Signals List */}
      {signals.length > 1 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-white mb-3">
            All Signals ({signals.length})
          </h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {signals.slice(0, 10).map((signal, idx) => (
              <SignalRow key={signal.id} signal={signal} rank={idx + 1} />
            ))}
          </div>
          {signals.length > 10 && (
            <div className="text-xs text-gray-500 text-center mt-2">
              +{signals.length - 10} more signals
            </div>
          )}
        </div>
      )}

      {signals.length === 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <div className="text-center">
            <AlertCircle size={32} className="text-gray-600 mx-auto mb-2" />
            <div className="text-sm text-gray-500">No high-quality FVG signals found</div>
            <div className="text-xs text-gray-600 mt-1">
              Try adjusting filters or wait for better setups
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Sub-components

function StatBox({
  label,
  value,
  subtitle,
  color,
}: {
  label: string
  value: string
  subtitle?: string
  color: 'blue' | 'green' | 'purple' | 'cyan'
}) {
  const colorMap = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    purple: 'text-purple-400',
    cyan: 'text-cyan-400',
  }

  return (
    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={`text-lg font-bold ${colorMap[color]}`}>{value}</div>
      {subtitle && <div className="text-xs text-gray-500 mt-0.5">{subtitle}</div>}
    </div>
  )
}

function QualityBadge({ signal }: { signal: FvgStrategySignal }) {
  const badges = {
    high: {
      label: 'HIGH QUALITY',
      bg: 'bg-green-500/20',
      border: 'border-green-500',
      text: 'text-green-400',
    },
    medium: {
      label: 'MEDIUM',
      bg: 'bg-yellow-500/20',
      border: 'border-yellow-500',
      text: 'text-yellow-400',
    },
    low: {
      label: 'LOW',
      bg: 'bg-gray-500/20',
      border: 'border-gray-500',
      text: 'text-gray-400',
    },
  }

  const badge = badges[signal.strength]

  return (
    <div className={`px-3 py-1 rounded-lg border ${badge.bg} ${badge.border}`}>
      <div className="flex items-center gap-1.5">
        {signal.mtfConfirmation ? (
          <CheckCircle2 size={12} className={badge.text} />
        ) : (
          <XCircle size={12} className="text-gray-500" />
        )}
        <span className={`text-xs font-semibold ${badge.text}`}>
          {badge.label}
        </span>
      </div>
    </div>
  )
}

function SignalCard({ signal }: { signal: FvgStrategySignal }) {
  const isLong = signal.type === 'long'

  return (
    <div className="space-y-3">
      {/* Signal Type & Confidence */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isLong ? (
            <TrendingUp size={20} className="text-green-400" />
          ) : (
            <TrendingDown size={20} className="text-red-400" />
          )}
          <div>
            <div className={`text-lg font-bold ${isLong ? 'text-green-400' : 'text-red-400'}`}>
              {signal.type.toUpperCase()} SIGNAL
            </div>
            <div className="text-xs text-gray-500">
              {signal.confidence.toFixed(0)}% confidence •{' '}
              {signal.mtfConfirmation ? (
                <span className="text-green-400">✓ HTF Confirmed</span>
              ) : (
                <span className="text-gray-500">✗ No HTF Confirm</span>
              )}
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-400">R:R Ratio</div>
          <div className="text-lg font-bold text-blue-400">
            {signal.riskRewardRatio.toFixed(2)}:1
          </div>
        </div>
      </div>

      {/* Entry Zone */}
      <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
        <div className="text-xs text-gray-400 mb-2 font-semibold">ENTRY ZONE</div>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-gray-500">Entry Range</div>
            <div className="text-base font-bold text-white">
              ${signal.entryZoneLow.toFixed(2)} - ${signal.entryZoneHigh.toFixed(2)}
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500">Target Entry</div>
            <div className="text-base font-bold text-cyan-400">
              ${signal.entryPrice.toFixed(2)}
            </div>
          </div>
        </div>
      </div>

      {/* Stop Loss & Take Profits */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-red-500/10 rounded-lg p-3 border border-red-500/30">
          <div className="text-xs text-gray-400 mb-1">Stop Loss</div>
          <div className="text-lg font-bold text-red-400">
            ${signal.stopLoss.toFixed(2)}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            -${signal.riskAmount.toFixed(2)} risk
          </div>
        </div>

        <div className="bg-green-500/10 rounded-lg p-3 border border-green-500/30">
          <div className="text-xs text-gray-400 mb-1">Final Target (TP3)</div>
          <div className="text-lg font-bold text-green-400">
            ${signal.tp3.toFixed(2)}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            +${(signal.riskAmount * 2).toFixed(2)} reward
          </div>
        </div>
      </div>

      {/* Take Profit Levels */}
      <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
        <div className="text-xs text-gray-400 mb-2 font-semibold">TAKE PROFIT LEVELS</div>
        <div className="space-y-2">
          <TPLevel label="TP1 (50% position)" price={signal.tp1} percentage="50%" color="green" />
          <TPLevel label="TP2 (30% position)" price={signal.tp2} percentage="30%" color="green" />
          <TPLevel label="TP3 (20% position)" price={signal.tp3} percentage="100%" color="green" />
        </div>
        {signal.trailStopAfterTP1 && (
          <div className="text-xs text-blue-400 mt-2">
            ✓ Trailing stop activated after TP1 hit
          </div>
        )}
      </div>

      {/* Risk Management */}
      <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
        <div className="text-xs text-gray-400 mb-2 font-semibold">RISK MANAGEMENT</div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <div className="text-xs text-gray-500">Position Size</div>
            <div className="text-sm font-bold text-white">
              {signal.positionSizePercent.toFixed(1)}% of capital
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Gap Size</div>
            <div className="text-sm font-bold text-white">
              {signal.gapSizePercent.toFixed(2)}%
            </div>
          </div>
        </div>
      </div>

      {/* Pattern Details */}
      <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
        <div className="text-xs text-gray-400 mb-2 font-semibold">PATTERN ANALYSIS</div>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-500">Volume Profile:</span>
            <span className="text-white font-medium">{signal.volumeProfile}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Market Structure:</span>
            <span className="text-white font-medium">{signal.marketStructure}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">HTF Bias:</span>
            <span className={`font-medium ${
              signal.htfBias === 'bullish' ? 'text-green-400' :
              signal.htfBias === 'bearish' ? 'text-red-400' :
              'text-gray-400'
            }`}>
              {signal.htfBias || 'neutral'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

function TPLevel({
  label,
  price,
  percentage,
  color,
}: {
  label: string
  price: number
  percentage: string
  color: string
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-gray-400">{label}</span>
      <div className="flex items-center gap-2">
        <span className={`text-sm font-bold text-${color}-400`}>
          ${price.toFixed(2)}
        </span>
        <span className="text-xs text-gray-500">({percentage})</span>
      </div>
    </div>
  )
}

function SignalRow({ signal, rank }: { signal: FvgStrategySignal; rank: number }) {
  const isLong = signal.type === 'long'

  return (
    <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50 hover:border-gray-600 transition-colors">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="text-xs font-mono text-gray-500">#{rank}</div>
          <div className="flex items-center gap-2">
            {isLong ? (
              <TrendingUp size={14} className="text-green-400" />
            ) : (
              <TrendingDown size={14} className="text-red-400" />
            )}
            <div>
              <div className={`text-sm font-semibold ${isLong ? 'text-green-400' : 'text-red-400'}`}>
                {signal.type.toUpperCase()}
              </div>
              <div className="text-xs text-gray-500">
                ${signal.entryPrice.toFixed(2)} • {signal.confidence.toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {signal.mtfConfirmation && (
            <CheckCircle2 size={14} className="text-green-400" />
          )}
          <div className="text-right">
            <div className="text-xs text-gray-500">R:R</div>
            <div className="text-sm font-bold text-blue-400">
              {signal.riskRewardRatio.toFixed(1)}:1
            </div>
          </div>
          <div className={`px-2 py-0.5 rounded text-xs font-semibold ${
            signal.strength === 'high' ? 'bg-green-500/20 text-green-400' :
            signal.strength === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
            'bg-gray-500/20 text-gray-400'
          }`}>
            {signal.strength.charAt(0).toUpperCase() + signal.strength.slice(1)}
          </div>
        </div>
      </div>
    </div>
  )
}
