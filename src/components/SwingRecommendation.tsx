'use client'

import { useState } from 'react'

// Types for swing data from ML API
export interface SwingV6Data {
  prob_1d_up: number
  prob_3d_up: number
  prob_5d_up: number
  prob_10d_up: number
  signal_1d: string
  signal_3d: string
  signal_5d: string
  signal_10d: string
}

export interface SwingLevels {
  current_price: number
  sma_20: number
  sma_50: number
  daily_high_5d: number
  daily_low_5d: number
  atr_14: number
  weekly_high?: number
  weekly_low?: number
  weekly_mid?: number
}

export interface SwingPhase4 {
  allowed: boolean
  bias: string
  holding_period: string
  invalidation_levels: string[]
  mode: string
}

export interface SwingData {
  v6_swing: SwingV6Data
  phase1: {
    direction: string
    confidence: string
    levels: SwingLevels
    momentum?: {
      rsi_14: number
      macd_signal: string
      momentum_5d: number
      momentum_10d: number
    }
    trend_alignment?: {
      aligned: boolean
      daily_trend: string
      weekly_trend: string
    }
  }
  phase4: SwingPhase4
  intraday_bias?: string
  intraday_execution_mode?: string
}

interface SwingRecommendationProps {
  symbol: string
  swingData: SwingData
  intradayBias?: string
  intradayExecutionMode?: string
}

// Generate actionable recommendation based on swing model outputs
function generateRecommendation(data: SwingData): {
  action: 'BUY_NOW' | 'WAIT_FOR_PULLBACK' | 'HOLD' | 'AVOID' | 'TAKE_PROFIT'
  headline: string
  reasoning: string[]
  entryStrategy: string
  stopLoss: string
  targets: string[]
  holdingPeriod: string
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH'
  confidence: number
} {
  const { v6_swing, phase1, phase4 } = data
  const levels = phase1.levels

  // Calculate aggregate probabilities
  const shortTermBullish = v6_swing.prob_1d_up > 0.5
  const mediumTermBullish = v6_swing.prob_3d_up > 0.55
  const swingBullish = v6_swing.prob_5d_up > 0.5 || v6_swing.prob_10d_up > 0.55

  const intradayBearish = data.intraday_bias === 'NEUTRAL' || data.intraday_execution_mode === 'NO_TRADE'
  const swingAllowed = phase4.allowed
  const swingBias = phase4.bias

  // Confidence based on alignment
  let confidence = 50
  if (v6_swing.prob_3d_up > 0.65) confidence += 15
  if (v6_swing.prob_10d_up > 0.6) confidence += 10
  if (phase1.confidence === 'HIGH') confidence += 10
  if (phase1.trend_alignment?.aligned) confidence += 10
  if (intradayBearish) confidence -= 10
  confidence = Math.min(95, Math.max(30, confidence))

  // Calculate key levels
  const stopLossLevel = levels.daily_low_5d
  const sma20 = levels.sma_20
  const sma50 = levels.sma_50
  const currentPrice = levels.current_price
  const target1 = levels.daily_high_5d
  const target2 = currentPrice + (levels.atr_14 * 1.5)

  // Decision logic

  // Case 1: Short-term bearish but medium/long-term bullish - WAIT FOR PULLBACK
  if (!shortTermBullish && (mediumTermBullish || swingBullish) && swingAllowed) {
    const pullbackZone = sma20 > sma50
      ? `$${(sma20 - 2).toFixed(2)} - $${sma20.toFixed(2)}`
      : `$${(currentPrice * 0.99).toFixed(2)} - $${(currentPrice * 0.995).toFixed(2)}`

    return {
      action: 'WAIT_FOR_PULLBACK',
      headline: 'Wait for Better Entry',
      reasoning: [
        `1-Day model is BEARISH (${Math.round(v6_swing.prob_1d_up * 100)}% up) - short-term weakness expected`,
        `3-Day model is ${v6_swing.signal_3d} (${Math.round(v6_swing.prob_3d_up * 100)}% up probability)`,
        `10-Day model is ${v6_swing.signal_10d} (${Math.round(v6_swing.prob_10d_up * 100)}% up probability)`,
        swingBias === 'LONG' ? 'Swing bias remains bullish - look for entry on dip' : 'Mixed signals - patience recommended'
      ],
      entryStrategy: `Look for pullback to ${pullbackZone} (near SMA 20 at $${sma20.toFixed(2)})`,
      stopLoss: `Below $${stopLossLevel.toFixed(2)} (5-day low)`,
      targets: [
        `T1: $${target1.toFixed(2)} (recent high)`,
        `T2: $${target2.toFixed(2)} (+1.5 ATR)`
      ],
      holdingPeriod: phase4.holding_period || '5-10 days',
      riskLevel: 'LOW',
      confidence
    }
  }

  // Case 2: All timeframes aligned bullish - BUY NOW
  if (shortTermBullish && mediumTermBullish && swingAllowed && swingBias === 'LONG') {
    return {
      action: 'BUY_NOW',
      headline: 'Strong Buy Signal',
      reasoning: [
        `All timeframes aligned BULLISH`,
        `1-Day: ${Math.round(v6_swing.prob_1d_up * 100)}% up probability`,
        `3-Day: ${Math.round(v6_swing.prob_3d_up * 100)}% up probability`,
        `10-Day: ${Math.round(v6_swing.prob_10d_up * 100)}% up probability`,
        phase1.trend_alignment?.aligned ? 'Daily & Weekly trends aligned' : ''
      ].filter(Boolean),
      entryStrategy: `Enter at market or near $${currentPrice.toFixed(2)}`,
      stopLoss: `Below $${stopLossLevel.toFixed(2)} (5-day low) or SMA50 at $${sma50.toFixed(2)}`,
      targets: [
        `T1: $${target1.toFixed(2)} (recent high)`,
        `T2: $${target2.toFixed(2)} (+1.5 ATR)`
      ],
      holdingPeriod: phase4.holding_period || '5-10 days',
      riskLevel: 'MEDIUM',
      confidence
    }
  }

  // Case 3: Already in position and showing strength - HOLD
  if (swingBullish && phase1.direction === 'BULLISH' && !phase4.allowed) {
    return {
      action: 'HOLD',
      headline: 'Hold Current Position',
      reasoning: [
        'Swing models remain bullish',
        `10-Day probability: ${Math.round(v6_swing.prob_10d_up * 100)}%`,
        'Wait for clearer entry if not in position'
      ],
      entryStrategy: 'Already positioned or wait for next setup',
      stopLoss: phase4.invalidation_levels?.[0] || `Below $${stopLossLevel.toFixed(2)}`,
      targets: [`T1: $${target1.toFixed(2)}`],
      holdingPeriod: phase4.holding_period || '5-10 days',
      riskLevel: 'LOW',
      confidence
    }
  }

  // Case 4: Bearish across timeframes - AVOID
  if (!shortTermBullish && !mediumTermBullish && !swingBullish) {
    return {
      action: 'AVOID',
      headline: 'Stay on Sidelines',
      reasoning: [
        'Multiple timeframes showing bearish signals',
        `1-Day: ${v6_swing.signal_1d} (${Math.round(v6_swing.prob_1d_up * 100)}%)`,
        `3-Day: ${v6_swing.signal_3d} (${Math.round(v6_swing.prob_3d_up * 100)}%)`,
        'Wait for reversal signals before entering'
      ],
      entryStrategy: 'No entry - wait for conditions to improve',
      stopLoss: 'N/A',
      targets: [],
      holdingPeriod: 'N/A',
      riskLevel: 'HIGH',
      confidence: 100 - confidence // High confidence in avoiding
    }
  }

  // Case 5: Near resistance with weakening momentum - TAKE PROFIT
  if (currentPrice >= target1 * 0.99 && !shortTermBullish) {
    return {
      action: 'TAKE_PROFIT',
      headline: 'Consider Taking Profits',
      reasoning: [
        'Price near recent highs',
        'Short-term momentum weakening',
        `1-Day turning bearish: ${Math.round(v6_swing.prob_1d_up * 100)}%`,
        'Lock in gains before potential pullback'
      ],
      entryStrategy: 'Scale out of existing position',
      stopLoss: 'Trail stop to breakeven',
      targets: ['Current levels - take profits'],
      holdingPeriod: 'Exit now',
      riskLevel: 'MEDIUM',
      confidence
    }
  }

  // Default: WAIT FOR PULLBACK (conservative)
  return {
    action: 'WAIT_FOR_PULLBACK',
    headline: 'Mixed Signals - Be Patient',
    reasoning: [
      'Conflicting signals across timeframes',
      `1-Day: ${v6_swing.signal_1d}`,
      `3-Day: ${v6_swing.signal_3d}`,
      `Swing bias: ${swingBias || 'NEUTRAL'}`
    ],
    entryStrategy: 'Wait for clearer setup',
    stopLoss: `Below $${stopLossLevel.toFixed(2)}`,
    targets: [`T1: $${target1.toFixed(2)}`],
    holdingPeriod: phase4.holding_period || '3-5 days',
    riskLevel: 'MEDIUM',
    confidence
  }
}

export function SwingRecommendation({ symbol, swingData, intradayBias, intradayExecutionMode }: SwingRecommendationProps) {
  const [expanded, setExpanded] = useState(false)

  // Add intraday context to swing data
  const enrichedData: SwingData = {
    ...swingData,
    intraday_bias: intradayBias,
    intraday_execution_mode: intradayExecutionMode
  }

  const rec = generateRecommendation(enrichedData)
  const v6 = swingData.v6_swing

  const getActionColor = (action: string) => {
    switch (action) {
      case 'BUY_NOW': return 'bg-green-500 text-white'
      case 'WAIT_FOR_PULLBACK': return 'bg-yellow-500 text-black'
      case 'HOLD': return 'bg-blue-500 text-white'
      case 'TAKE_PROFIT': return 'bg-purple-500 text-white'
      case 'AVOID': return 'bg-red-500 text-white'
      default: return 'bg-gray-500 text-white'
    }
  }

  const getActionBorder = (action: string) => {
    switch (action) {
      case 'BUY_NOW': return 'border-green-500/50 bg-green-900/20'
      case 'WAIT_FOR_PULLBACK': return 'border-yellow-500/50 bg-yellow-900/20'
      case 'HOLD': return 'border-blue-500/50 bg-blue-900/20'
      case 'TAKE_PROFIT': return 'border-purple-500/50 bg-purple-900/20'
      case 'AVOID': return 'border-red-500/50 bg-red-900/20'
      default: return 'border-gray-500/50 bg-gray-900/20'
    }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'LOW': return 'text-green-400'
      case 'MEDIUM': return 'text-yellow-400'
      case 'HIGH': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  return (
    <div className={`rounded-xl border p-4 transition-all ${getActionBorder(rec.action)}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className={`px-3 py-1.5 rounded-lg font-bold text-sm ${getActionColor(rec.action)}`}>
            {rec.action.replace(/_/g, ' ')}
          </span>
          <span className="text-white font-semibold">{rec.headline}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gray-400 text-xs">Confidence:</span>
          <span className={`font-bold ${rec.confidence >= 70 ? 'text-green-400' : rec.confidence >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
            {rec.confidence}%
          </span>
        </div>
      </div>

      {/* Swing Model Probabilities Grid */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        {[
          { label: '1-Day', prob: v6.prob_1d_up, signal: v6.signal_1d },
          { label: '3-Day', prob: v6.prob_3d_up, signal: v6.signal_3d },
          { label: '5-Day', prob: v6.prob_5d_up, signal: v6.signal_5d },
          { label: '10-Day', prob: v6.prob_10d_up, signal: v6.signal_10d },
        ].map(({ label, prob, signal }) => (
          <div key={label} className="bg-gray-800/50 rounded-lg p-2 text-center">
            <div className="text-gray-500 text-xs mb-1">{label}</div>
            <div className={`text-lg font-bold ${
              signal === 'BULLISH' ? 'text-green-400' :
              signal === 'BEARISH' ? 'text-red-400' :
              'text-yellow-400'
            }`}>
              {Math.round(prob * 100)}%
            </div>
            <div className={`text-xs ${
              signal === 'BULLISH' ? 'text-green-400' :
              signal === 'BEARISH' ? 'text-red-400' :
              'text-yellow-400'
            }`}>
              {signal === 'BULLISH' ? '▲' : signal === 'BEARISH' ? '▼' : '◆'} {signal}
            </div>
          </div>
        ))}
      </div>

      {/* Quick Summary */}
      <div className="bg-gray-800/30 rounded-lg p-3 mb-3">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div>
            <span className="text-gray-500 text-xs block">Entry</span>
            <span className="text-white font-medium">{rec.entryStrategy.split(' ').slice(0, 4).join(' ')}...</span>
          </div>
          <div>
            <span className="text-gray-500 text-xs block">Stop Loss</span>
            <span className="text-red-400 font-medium">{rec.stopLoss.split(' ').slice(0, 3).join(' ')}</span>
          </div>
          <div>
            <span className="text-gray-500 text-xs block">Hold Period</span>
            <span className="text-cyan-400 font-medium">{rec.holdingPeriod}</span>
          </div>
          <div>
            <span className="text-gray-500 text-xs block">Risk Level</span>
            <span className={`font-medium ${getRiskColor(rec.riskLevel)}`}>{rec.riskLevel}</span>
          </div>
        </div>
      </div>

      {/* Expandable Details */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left text-gray-400 text-xs hover:text-white transition-colors flex items-center gap-1"
      >
        <span>{expanded ? '▼' : '▶'}</span>
        <span>{expanded ? 'Hide' : 'Show'} Full Analysis</span>
      </button>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-gray-700 space-y-3">
          {/* Reasoning */}
          <div>
            <div className="text-gray-500 text-xs mb-1">REASONING</div>
            <ul className="text-sm space-y-1">
              {rec.reasoning.map((reason, i) => (
                <li key={i} className="text-gray-300 flex items-start gap-2">
                  <span className="text-gray-600">•</span>
                  {reason}
                </li>
              ))}
            </ul>
          </div>

          {/* Entry Strategy */}
          <div>
            <div className="text-gray-500 text-xs mb-1">ENTRY STRATEGY</div>
            <p className="text-white text-sm">{rec.entryStrategy}</p>
          </div>

          {/* Stop Loss */}
          <div>
            <div className="text-gray-500 text-xs mb-1">STOP LOSS</div>
            <p className="text-red-400 text-sm font-medium">{rec.stopLoss}</p>
          </div>

          {/* Targets */}
          {rec.targets.length > 0 && (
            <div>
              <div className="text-gray-500 text-xs mb-1">PROFIT TARGETS</div>
              <ul className="text-sm space-y-1">
                {rec.targets.map((target, i) => (
                  <li key={i} className="text-green-400">{target}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Key Levels */}
          {swingData.phase1.levels && (
            <div>
              <div className="text-gray-500 text-xs mb-1">KEY LEVELS</div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="bg-gray-800/50 rounded p-2">
                  <span className="text-gray-500">SMA 20</span>
                  <div className="text-blue-400 font-mono">${swingData.phase1.levels.sma_20.toFixed(2)}</div>
                </div>
                <div className="bg-gray-800/50 rounded p-2">
                  <span className="text-gray-500">SMA 50</span>
                  <div className="text-purple-400 font-mono">${swingData.phase1.levels.sma_50.toFixed(2)}</div>
                </div>
                <div className="bg-gray-800/50 rounded p-2">
                  <span className="text-gray-500">5D Low</span>
                  <div className="text-red-400 font-mono">${swingData.phase1.levels.daily_low_5d.toFixed(2)}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
