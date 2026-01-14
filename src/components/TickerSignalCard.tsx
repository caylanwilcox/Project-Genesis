'use client'

import { useState } from 'react'

// ========================
// TYPES
// ========================

export interface V6Signal {
  action: 'LONG' | 'SHORT' | 'NO_TRADE'
  reason: string
  probability_a: number | null
  probability_b: number | null
  session: 'early' | 'late'
  price_11am: number | null
}

export interface NorthstarKeyLevels {
  recent_high: number
  recent_low: number
  mid_point: number
  pivot: number
  pivot_r1: number
  pivot_s1: number
  current_price: number
  today_open: number
  prev_close: number
  retest_high?: number
  retest_low?: number
}

export interface NorthstarData {
  phase1: {
    direction: string
    confidence_band: string
    dominant_timeframe: string
    acceptance: {
      accepted: boolean
      acceptance_strength: string
      acceptance_reason: string
      failed_levels: string[]
    }
    range: {
      state: string
      rotation_complete: boolean
      expansion_quality: string
    }
    mtf: {
      aligned: boolean
      dominant_tf: string
      conflict_tf: string | null
    }
    participation: {
      conviction: string
      effort_result_match: boolean
    }
    failure: {
      present: boolean
      failure_types: string[]
    }
    key_levels: NorthstarKeyLevels
    volatility_expansion?: {
      probability: number
      signal: string
      expansion_likely: boolean
      reasons: string[]
    }
  }
  phase2: {
    health_score: number
    tier: string
    stand_down: boolean
    reasons: string[]
    dimensions: {
      structural_integrity: number
      time_persistence: number
      volatility_alignment: number
      participation_consistency: number
      failure_risk: number
    }
  }
  phase3: {
    throttle: string
    density_score: number
    allowed_signals: number
    reasons: string[]
  }
  phase4: {
    allowed: boolean
    bias: string
    execution_mode: string
    risk_state: string
    invalidation_context: string[]
  }
}

export interface TickerSignalCardProps {
  symbol: string
  currentPrice: number
  todayOpen: number
  todayChangePct: number
  barsAnalyzed: number
  v6: V6Signal
  northstar?: NorthstarData | null
  session: 'early' | 'late'
  isLoading?: boolean
}

// Historical accuracy from REAL 2025 backtest data
const getTargetBAccuracy = (prob: number): number => {
  const confidence = Math.max(prob, 1 - prob) * 100
  if (confidence >= 85) return 96
  if (confidence >= 80) return 81
  if (confidence >= 75) return 78
  if (confidence >= 70) return 68
  if (confidence >= 65) return 64
  if (confidence >= 60) return 61
  return 54
}

const getTargetAAccuracy = (prob: number): number => {
  const confidence = Math.max(prob, 1 - prob) * 100
  if (confidence >= 85) return 96
  if (confidence >= 80) return 85
  if (confidence >= 75) return 79
  if (confidence >= 70) return 73
  if (confidence >= 65) return 68
  if (confidence >= 60) return 63
  return 56
}

const getAccuracyColor = (accuracy: number) => {
  if (accuracy >= 80) return 'text-green-400'
  if (accuracy >= 70) return 'text-yellow-400'
  return 'text-gray-500'
}

// Dimension score bar component
function DimensionBar({ label, score, weight }: { label: string; score: number; weight: number }) {
  const getScoreColor = (s: number) => {
    if (s >= 80) return 'bg-green-500'
    if (s >= 60) return 'bg-yellow-500'
    if (s >= 40) return 'bg-orange-500'
    return 'bg-red-500'
  }

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-gray-400 w-32 truncate">{label}</span>
      <div className="flex-1 h-2 bg-gray-700 rounded overflow-hidden">
        <div
          className={`h-full ${getScoreColor(score)} transition-all`}
          style={{ width: `${score}%` }}
        />
      </div>
      <span className={`w-10 text-right font-mono ${score >= 70 ? 'text-green-400' : score >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
        {score}%
      </span>
      <span className="text-gray-600 w-10 text-right">x{(weight * 100).toFixed(0)}%</span>
    </div>
  )
}

// Northstar Details Component with expandable explanations
function NorthstarDetails({ ns }: { ns: NorthstarData }) {
  const [expanded, setExpanded] = useState<string | null>(null)

  const toggleExpand = (section: string) => {
    setExpanded(expanded === section ? null : section)
  }

  // Safe access to dimensions with defaults
  const dims = ns.phase2?.dimensions || {
    structural_integrity: 0,
    time_persistence: 0,
    volatility_alignment: 0,
    participation_consistency: 0,
    failure_risk: 0
  }

  // Safe access to phase4 with defaults
  const phase4Allowed = ns.phase4?.allowed ?? false

  return (
    <div className="pt-3 border-t border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-500 text-xs">MARKET STRUCTURE</span>
        <span className={`text-xs px-2 py-0.5 rounded ${
          phase4Allowed
            ? 'bg-green-500/20 text-green-400'
            : 'bg-red-500/20 text-red-400'
        }`}>
          {phase4Allowed ? 'TRADE ALLOWED' : 'STAND DOWN'}
        </span>
      </div>

      {/* Direction - Clickable */}
      <div
        className={`rounded-lg p-2 mb-2 cursor-pointer transition-all ${
          ns.phase1?.direction === 'UP' ? 'bg-green-900/30 border border-green-500/30 hover:bg-green-900/40' :
          ns.phase1?.direction === 'DOWN' ? 'bg-red-900/30 border border-red-500/30 hover:bg-red-900/40' :
          'bg-gray-800/50 border border-gray-700 hover:bg-gray-800/70'
        }`}
        onClick={() => toggleExpand('direction')}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className={`text-lg ${
              ns.phase1?.direction === 'UP' ? 'text-green-400' :
              ns.phase1?.direction === 'DOWN' ? 'text-red-400' :
              'text-gray-400'
            }`}>
              {ns.phase1?.direction === 'UP' ? '▲' : ns.phase1?.direction === 'DOWN' ? '▼' : '◆'}
            </span>
            <span className="text-white font-medium">
              {ns.phase1?.direction === 'UP' ? 'Bullish Structure' :
               ns.phase1?.direction === 'DOWN' ? 'Bearish Structure' :
               'Balanced/Choppy'}
            </span>
            <span className="text-gray-600 text-xs">i</span>
          </div>
          <span className={`text-xs ${
            ns.phase1?.confidence_band === 'STRUCTURAL_EDGE' ? 'text-green-400' :
            ns.phase1?.confidence_band === 'CONTEXT_ONLY' ? 'text-yellow-400' :
            'text-red-400'
          }`}>
            {ns.phase1?.confidence_band === 'STRUCTURAL_EDGE' ? 'Strong Edge' :
             ns.phase1?.confidence_band === 'CONTEXT_ONLY' ? 'Weak Edge' :
             'No Edge'}
          </span>
        </div>
        {expanded === 'direction' && (
          <div className="mt-2 pt-2 border-t border-gray-700 text-xs space-y-2">
            {/* Acceptance Details */}
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${ns.phase1?.acceptance?.accepted ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-gray-300">Price Acceptance:</span>
              <span className={ns.phase1?.acceptance?.accepted ? 'text-green-400' : 'text-red-400'}>
                {ns.phase1?.acceptance?.accepted
                  ? `YES - ${ns.phase1?.acceptance?.acceptance_strength || 'MODERATE'}`
                  : 'NO'}
              </span>
            </div>

            {ns.phase1?.acceptance?.acceptance_reason && (
              <div className="text-gray-400 pl-4 text-xs italic">
                {ns.phase1?.acceptance?.acceptance_reason}
              </div>
            )}

            {ns.phase1?.acceptance?.failed_levels?.length > 0 && (
              <div className="text-red-400 pl-4">
                Warning: Failed levels: {ns.phase1?.acceptance?.failed_levels?.join(', ')}
              </div>
            )}

            {/* Range State */}
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                ns.phase1?.range?.state === 'TREND' ? 'bg-green-500' :
                ns.phase1?.range?.state === 'BALANCE' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-gray-300">Range State:</span>
              <span className={
                ns.phase1?.range?.state === 'TREND' ? 'text-green-400' :
                ns.phase1?.range?.state === 'BALANCE' ? 'text-yellow-400' : 'text-red-400'
              }>
                {ns.phase1?.range?.state || 'UNKNOWN'}
              </span>
            </div>

            {/* MTF Alignment */}
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${ns.phase1?.mtf?.aligned ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-gray-300">Multi-Timeframe:</span>
              <span className={ns.phase1?.mtf?.aligned ? 'text-green-400' : 'text-red-400'}>
                {ns.phase1?.mtf?.aligned
                  ? `ALIGNED - ${ns.phase1?.mtf?.dominant_tf || 'INTRADAY'} dominant`
                  : `CONFLICT - ${ns.phase1?.mtf?.conflict_tf || 'Timeframes disagree'}`}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* KEY LEVELS */}
      {ns.phase1?.key_levels && ns.phase1.key_levels.current_price > 0 && (
        <div
          className="rounded-lg p-2 mb-2 bg-blue-900/20 border border-blue-500/30 cursor-pointer hover:bg-blue-900/30 transition-all"
          onClick={() => toggleExpand('levels')}
        >
          <div className="flex items-center justify-between">
            <span className="text-blue-400 text-xs font-medium">KEY LEVELS</span>
            <span className="text-gray-600 text-xs">i</span>
          </div>
          <div className="mt-1 grid grid-cols-3 gap-2 text-xs">
            <div>
              <span className="text-red-400">R1: </span>
              <span className="text-white font-mono">${ns.phase1.key_levels.pivot_r1?.toFixed(2) || '--'}</span>
            </div>
            <div>
              <span className="text-yellow-400">P: </span>
              <span className="text-white font-mono">${ns.phase1.key_levels.pivot?.toFixed(2) || '--'}</span>
            </div>
            <div>
              <span className="text-green-400">S1: </span>
              <span className="text-white font-mono">${ns.phase1.key_levels.pivot_s1?.toFixed(2) || '--'}</span>
            </div>
          </div>
          {expanded === 'levels' && (
            <div className="mt-2 pt-2 border-t border-blue-500/30 space-y-2 text-xs">
              <div className="grid grid-cols-2 gap-2 mt-2">
                <div>
                  <div className="text-gray-500 mb-1">Pivot Levels:</div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-red-400">R1:</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.pivot_r1?.toFixed(2) || '--'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-yellow-400">Pivot:</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.pivot?.toFixed(2) || '--'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-green-400">S1:</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.pivot_s1?.toFixed(2) || '--'}</span>
                    </div>
                  </div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">Reference:</div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Open:</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.today_open?.toFixed(2) || '--'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Prev Close:</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.prev_close?.toFixed(2) || '--'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Mid:</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.mid_point?.toFixed(2) || '--'}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Health and Window */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div
          className={`rounded p-2 cursor-pointer transition-all ${
            ns.phase2?.tier === 'HEALTHY' ? 'bg-green-500/10 hover:bg-green-500/20' :
            ns.phase2?.tier === 'DEGRADED' ? 'bg-yellow-500/10 hover:bg-yellow-500/20' :
            'bg-red-500/10 hover:bg-red-500/20'
          }`}
          onClick={() => toggleExpand('health')}
        >
          <div className="flex items-center justify-between">
            <span className="text-gray-500">Signal Health</span>
            <span className="text-gray-600">i</span>
          </div>
          <div className={`font-medium ${
            ns.phase2?.tier === 'HEALTHY' ? 'text-green-400' :
            ns.phase2?.tier === 'DEGRADED' ? 'text-yellow-400' :
            'text-red-400'
          }`}>
            {ns.phase2?.tier === 'HEALTHY' ? 'Healthy' :
             ns.phase2?.tier === 'DEGRADED' ? 'Degraded' :
             'Unstable'} ({ns.phase2?.health_score ?? 0}%)
          </div>
          {expanded === 'health' && (
            <div className="mt-2 pt-2 border-t border-gray-700 space-y-2">
              <div className="text-gray-400 mb-1">Health Breakdown:</div>
              <DimensionBar label="Structure" score={dims.structural_integrity} weight={0.30} />
              <DimensionBar label="Time" score={dims.time_persistence} weight={0.15} />
              <DimensionBar label="Volatility" score={dims.volatility_alignment} weight={0.15} />
              <DimensionBar label="Participation" score={dims.participation_consistency} weight={0.20} />
              <DimensionBar label="Failure Risk" score={dims.failure_risk} weight={0.20} />
            </div>
          )}
        </div>

        <div
          className={`rounded p-2 cursor-pointer transition-all ${
            ns.phase3?.throttle === 'OPEN' ? 'bg-green-500/10 hover:bg-green-500/20' :
            ns.phase3?.throttle === 'LIMITED' ? 'bg-yellow-500/10 hover:bg-yellow-500/20' :
            'bg-red-500/10 hover:bg-red-500/20'
          }`}
          onClick={() => toggleExpand('window')}
        >
          <div className="flex items-center justify-between">
            <span className="text-gray-500">Trade Window</span>
            <span className="text-gray-600">i</span>
          </div>
          <div className={`font-medium ${
            ns.phase3?.throttle === 'OPEN' ? 'text-green-400' :
            ns.phase3?.throttle === 'LIMITED' ? 'text-yellow-400' :
            'text-red-400'
          }`}>
            {ns.phase3?.throttle === 'OPEN' ? 'Open' :
             ns.phase3?.throttle === 'LIMITED' ? 'Limited' :
             'Blocked'}
          </div>
          {expanded === 'window' && (
            <div className="mt-2 pt-2 border-t border-gray-700 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Density:</span>
                <span className="font-mono">{ns.phase3?.density_score ?? 0}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Signals:</span>
                <span className="font-mono">
                  {(ns.phase3?.allowed_signals ?? 0) === 999 ? 'Unlimited' :
                   (ns.phase3?.allowed_signals ?? 0) === 0 ? 'None' :
                   ns.phase3?.allowed_signals ?? 0}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Execution guidance or Stand Down */}
      {phase4Allowed ? (
        <div className="mt-2 p-2 bg-gray-800/50 rounded text-xs">
          <div className="flex justify-between items-center">
            <span className="text-gray-400">
              Play: <span className="text-white">
                {ns.phase4?.execution_mode === 'TREND_CONTINUATION' ? 'Trend Following' :
                 ns.phase4?.execution_mode === 'MEAN_REVERSION' ? 'Mean Reversion' :
                 ns.phase4?.execution_mode === 'SCALP' ? 'Quick Scalp' :
                 'Standard'}
              </span>
            </span>
            <span className={`${
              ns.phase4?.risk_state === 'NORMAL' ? 'text-green-400' :
              ns.phase4?.risk_state === 'REDUCED' ? 'text-yellow-400' :
              'text-red-400'
            }`}>
              Risk: {ns.phase4?.risk_state === 'NORMAL' ? 'Normal' :
                     ns.phase4?.risk_state === 'REDUCED' ? 'Half' :
                     'Quarter'}
            </span>
          </div>
        </div>
      ) : (
        <div className="mt-2 p-2 bg-red-900/20 border border-red-500/30 rounded text-xs text-red-400">
          <span>Warning: Market conditions unfavorable - wait for better setup</span>
        </div>
      )}
    </div>
  )
}

// Helper to calculate hours remaining until 12 PM ET
function getHoursUntilNoon(): string {
  const now = new Date()
  const etTime = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }))
  const hours = etTime.getHours()
  const minutes = etTime.getMinutes()

  if (hours >= 12) return 'Active'

  const hoursLeft = 12 - hours - 1
  const minutesLeft = 59 - minutes

  if (hoursLeft > 0) {
    return `${hoursLeft}h ${minutesLeft}m`
  }
  return `${minutesLeft}m`
}

// Main Component
export function TickerSignalCard({
  symbol,
  currentPrice,
  todayOpen,
  todayChangePct,
  barsAnalyzed,
  v6,
  northstar,
  session,
  isLoading = false
}: TickerSignalCardProps) {
  const probA = v6.probability_a || 0.5
  const probB = v6.probability_b || 0.5

  const getActionColor = (action: string) => {
    switch (action) {
      case 'LONG': return 'bg-green-500 text-white'
      case 'SHORT': return 'bg-red-500 text-white'
      default: return 'bg-gray-600 text-gray-300'
    }
  }

  const getActionStyle = (action: string) => {
    switch (action) {
      case 'LONG': return 'bg-green-900/40 border-green-500/50'
      case 'SHORT': return 'bg-red-900/40 border-red-500/50'
      default: return 'bg-gray-900/40 border-gray-700'
    }
  }

  if (isLoading) {
    return (
      <div className="border rounded-xl p-5 bg-gray-900/40 border-gray-700">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-2xl font-bold text-white">{symbol}</h2>
            <div className="text-gray-500 text-sm mt-1">Loading signals...</div>
          </div>
        </div>
        <div className="animate-pulse space-y-3">
          <div className="h-24 bg-gray-800/50 rounded-lg"></div>
          <div className="h-16 bg-gray-800/50 rounded-lg"></div>
        </div>
      </div>
    )
  }

  return (
    <div className={`border rounded-xl p-5 transition-all ${getActionStyle(v6.action)}`}>
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-bold text-white flex items-center gap-1">
              {symbol}
              <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </h2>
          </div>
          <div className="text-3xl font-light text-white mt-1">
            ${currentPrice.toFixed(2)}
          </div>
          <div className={`text-sm ${todayChangePct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {todayChangePct >= 0 ? '▲' : '▼'} {Math.abs(todayChangePct).toFixed(2)}%
          </div>
        </div>
        <div className={`px-4 py-2 rounded-lg font-bold text-lg ${getActionColor(v6.action)}`}>
          {v6.action === 'NO_TRADE' ? 'WAIT' : v6.action}
        </div>
      </div>

      {/* Target Dominance Boxes */}
      {session === 'late' ? (
        <div className="mb-4 space-y-2">
          {/* Target B - Primary for Late Session */}
          {(() => {
            const isBullish = probB >= 0.5
            const confidence = Math.round(Math.max(probB, 1 - probB) * 100)
            const direction = isBullish ? 'BULLISH' : 'BEARISH'
            const bgColor = isBullish ? 'bg-green-900/40 border-green-500/50' : 'bg-red-900/40 border-red-500/50'
            const textColor = isBullish ? 'text-green-400' : 'text-red-400'

            return (
              <div className={`rounded-lg p-3 border ${bgColor}`}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-400 text-xs font-medium">TARGET B: Close vs 11AM</span>
                  {v6.price_11am && (
                    <span className="text-xs text-cyan-400 font-mono">
                      ${v6.price_11am.toFixed(2)}
                    </span>
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={`text-xl font-bold ${textColor}`}>{direction}</span>
                    <span className={`text-lg ${textColor}`}>{isBullish ? '▲' : '▼'}</span>
                  </div>
                  <div className="text-right">
                    <div className={`text-xl font-bold ${textColor}`}>{confidence}%</div>
                    <div className="text-gray-500 text-xs">model confidence</div>
                  </div>
                </div>
              </div>
            )
          })()}

          {/* Target A - Secondary */}
          {(() => {
            const isBullish = probA >= 0.5
            const confidence = Math.round(Math.max(probA, 1 - probA) * 100)
            const direction = isBullish ? 'BULLISH' : 'BEARISH'
            const bgColor = isBullish ? 'bg-green-900/20 border-green-500/30' : 'bg-red-900/20 border-red-500/30'
            const textColor = isBullish ? 'text-green-400/80' : 'text-red-400/80'

            return (
              <div className={`rounded-lg p-2 border ${bgColor}`}>
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <span className="text-gray-500 text-xs">TARGET A: Close vs Open</span>
                    <span className="text-xs text-gray-600 font-mono">
                      ${todayOpen.toFixed(2)}
                    </span>
                  </div>
                  <span className={`text-sm font-bold ${textColor}`}>
                    {direction} {isBullish ? '▲' : '▼'} {confidence}%
                  </span>
                </div>
              </div>
            )
          })()}
        </div>
      ) : (
        /* Early Session - Target A Primary */
        <div className="mb-4">
          {(() => {
            const isBullish = probA >= 0.5
            const confidence = Math.round(Math.max(probA, 1 - probA) * 100)
            const direction = isBullish ? 'BULLISH' : 'BEARISH'
            const bgColor = isBullish ? 'bg-green-900/40 border-green-500/50' : 'bg-red-900/40 border-red-500/50'
            const textColor = isBullish ? 'text-green-400' : 'text-red-400'

            return (
              <div className={`rounded-lg p-3 border ${bgColor}`}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-400 text-xs font-medium">TARGET A: Close vs Open</span>
                  <span className="text-xs text-cyan-400 font-mono">
                    ${todayOpen.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={`text-xl font-bold ${textColor}`}>{direction}</span>
                    <span className={`text-lg ${textColor}`}>{isBullish ? '▲' : '▼'}</span>
                  </div>
                  <div className="text-right">
                    <div className={`text-xl font-bold ${textColor}`}>{confidence}%</div>
                    <div className="text-gray-500 text-xs">model confidence</div>
                  </div>
                </div>
              </div>
            )
          })()}
          <div className="mt-2 p-2 bg-gray-800/50 rounded border border-gray-700">
            <div className="flex justify-between items-center">
              <span className="text-gray-500 text-xs">Target B activates at 12 PM</span>
              <span className="text-yellow-400 text-sm font-mono">
                {getHoursUntilNoon()}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Historical Win Rates */}
      {v6.action !== 'NO_TRADE' && (
        <div className="mb-4 pt-3 border-t border-gray-800">
          <div className="text-gray-500 text-xs mb-2">If you traded at this confidence in 2025:</div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-gray-800/50 rounded p-2">
              <div className="text-gray-500 text-xs mb-1">Target A</div>
              <div className={`text-lg font-bold ${getAccuracyColor(getTargetAAccuracy(probA))}`}>
                {getTargetAAccuracy(probA)}%
              </div>
              <div className="text-xs text-gray-600">of trades won</div>
            </div>
            <div className={`rounded p-2 ${session === 'late' ? 'bg-cyan-900/30 border border-cyan-500/30' : 'bg-gray-800/50'}`}>
              <div className={`text-xs mb-1 ${session === 'late' ? 'text-cyan-400' : 'text-gray-500'}`}>
                Target B {session === 'late' && '★'}
              </div>
              <div className={`text-lg font-bold ${getAccuracyColor(getTargetBAccuracy(probB))}`}>
                {getTargetBAccuracy(probB)}%
              </div>
              <div className="text-xs text-gray-600">of trades won</div>
            </div>
          </div>
        </div>
      )}

      {/* Northstar Pipeline Summary */}
      {northstar && <NorthstarDetails ns={northstar} />}

      {/* Footer */}
      <div className="mt-3 pt-2 border-t border-gray-800 text-xs text-gray-600">
        {barsAnalyzed} bars analyzed | {v6.reason}
      </div>
    </div>
  )
}
