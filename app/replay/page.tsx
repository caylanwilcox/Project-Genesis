'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { ProfessionalChart } from '@/components/ProfessionalChart'
import type { CandleData, V6Prediction } from '@/components/ProfessionalChart/types'

interface ReplayData {
  mode: string
  replay_date: string
  replay_time: string
  simulated_time_et: string
  simulated_hour: number
  simulated_minute: number
  market_open: boolean
  session: 'early' | 'late'
  tickers: Record<string, TickerData>
  v6_signals: Record<string, V6Signal>
  northstar: Record<string, NorthstarData>
  summary: {
    best_ticker: string | null
    allowed_tickers: string[]
    v6_actionable: string[]
    recommendation: string
  }
}

interface TickerData {
  current_price: number
  today_open: number
  today_change_pct: number
  price_11am: number | null
  bars_analyzed: number
  v6: V6Signal
  northstar: NorthstarData
  position_pct?: number
  stop_loss?: number | null
  take_profit?: number | null
  bucket?: string
  confidence?: number
  error?: string
}

interface V6Signal {
  action: 'LONG' | 'SHORT' | 'NO_TRADE'
  reason: string
  probability_a: number | null
  probability_b: number | null
  session: string
  price_11am: number | null
}

interface NorthstarData {
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
    key_levels: {
      recent_high: number
      recent_low: number
      mid_point: number
      pivot: number
      pivot_r1: number
      pivot_s1: number
      current_price: number
      today_open: number
      prev_close: number
      retest_high: number
      retest_low: number
    }
    volatility_expansion: {
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

const SPEEDS = [
  { label: '1x', value: 1000 },
  { label: '2x', value: 500 },
  { label: '5x', value: 200 },
  { label: '10x', value: 100 },
  { label: '30x', value: 33 },
  { label: '60x', value: 16 },
]

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
      <span className="text-gray-600 w-10 text-right">×{(weight * 100).toFixed(0)}%</span>
    </div>
  )
}

// Northstar Details Component with expandable explanations showing SPECIFIC data
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

  return (
    <div className="pt-3 border-t border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-500 text-xs">MARKET STRUCTURE</span>
        <span className={`text-xs px-2 py-0.5 rounded ${
          ns.phase4.allowed
            ? 'bg-green-500/20 text-green-400'
            : 'bg-red-500/20 text-red-400'
        }`}>
          {ns.phase4.allowed ? 'TRADE ALLOWED' : 'STAND DOWN'}
        </span>
      </div>

      {/* Direction - Clickable with SPECIFIC data */}
      <div
        className={`rounded-lg p-2 mb-2 cursor-pointer transition-all ${
          ns.phase1.direction === 'UP' ? 'bg-green-900/30 border border-green-500/30 hover:bg-green-900/40' :
          ns.phase1.direction === 'DOWN' ? 'bg-red-900/30 border border-red-500/30 hover:bg-red-900/40' :
          'bg-gray-800/50 border border-gray-700 hover:bg-gray-800/70'
        }`}
        onClick={() => toggleExpand('direction')}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className={`text-lg ${
              ns.phase1.direction === 'UP' ? 'text-green-400' :
              ns.phase1.direction === 'DOWN' ? 'text-red-400' :
              'text-gray-400'
            }`}>
              {ns.phase1.direction === 'UP' ? '▲' : ns.phase1.direction === 'DOWN' ? '▼' : '◆'}
            </span>
            <span className="text-white font-medium">
              {ns.phase1.direction === 'UP' ? 'Bullish Structure' :
               ns.phase1.direction === 'DOWN' ? 'Bearish Structure' :
               'Balanced/Choppy'}
            </span>
            <span className="text-gray-600 text-xs">ⓘ</span>
          </div>
          <span className={`text-xs ${
            ns.phase1.confidence_band === 'STRUCTURAL_EDGE' ? 'text-green-400' :
            ns.phase1.confidence_band === 'CONTEXT_ONLY' ? 'text-yellow-400' :
            'text-red-400'
          }`}>
            {ns.phase1.confidence_band === 'STRUCTURAL_EDGE' ? 'Strong Edge' :
             ns.phase1.confidence_band === 'CONTEXT_ONLY' ? 'Weak Edge' :
             'No Edge'}
          </span>
        </div>
        {expanded === 'direction' && (
          <div className="mt-2 pt-2 border-t border-gray-700 text-xs space-y-2">
            {/* Acceptance Details */}
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${ns.phase1.acceptance?.accepted ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-gray-300">Price Acceptance:</span>
              <span className={ns.phase1.acceptance?.accepted ? 'text-green-400' : 'text-red-400'}>
                {ns.phase1.acceptance?.accepted
                  ? `YES - ${ns.phase1.acceptance?.acceptance_strength || 'MODERATE'}`
                  : 'NO'}
              </span>
            </div>

            {/* Acceptance Reason - WHY */}
            {ns.phase1.acceptance?.acceptance_reason && (
              <div className="text-gray-400 pl-4 text-xs italic">
                {ns.phase1.acceptance.acceptance_reason}
              </div>
            )}

            {/* Failed Levels */}
            {ns.phase1.acceptance?.failed_levels?.length > 0 && (
              <div className="text-red-400 pl-4">
                ⚠ Failed levels: {ns.phase1.acceptance.failed_levels.join(', ')}
              </div>
            )}

            {/* Range State */}
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                ns.phase1.range?.state === 'TREND' ? 'bg-green-500' :
                ns.phase1.range?.state === 'BALANCE' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-gray-300">Range State:</span>
              <span className={
                ns.phase1.range?.state === 'TREND' ? 'text-green-400' :
                ns.phase1.range?.state === 'BALANCE' ? 'text-yellow-400' : 'text-red-400'
              }>
                {ns.phase1.range?.state || 'UNKNOWN'}
                {ns.phase1.range?.expansion_quality && ns.phase1.range.expansion_quality !== 'NONE' &&
                  ` (${ns.phase1.range.expansion_quality} expansion)`}
                {ns.phase1.range?.rotation_complete && ' - Rotation complete'}
              </span>
            </div>

            {/* MTF Alignment */}
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${ns.phase1.mtf?.aligned ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-gray-300">Multi-Timeframe:</span>
              <span className={ns.phase1.mtf?.aligned ? 'text-green-400' : 'text-red-400'}>
                {ns.phase1.mtf?.aligned
                  ? `ALIGNED - ${ns.phase1.mtf?.dominant_tf || 'INTRADAY'} dominant`
                  : `CONFLICT - ${ns.phase1.mtf?.conflict_tf || 'Timeframes disagree'}`}
              </span>
            </div>

            {/* Participation */}
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                ns.phase1.participation?.conviction === 'HIGH' ? 'bg-green-500' :
                ns.phase1.participation?.conviction === 'MEDIUM' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-gray-300">Volume Conviction:</span>
              <span className={
                ns.phase1.participation?.conviction === 'HIGH' ? 'text-green-400' :
                ns.phase1.participation?.conviction === 'MEDIUM' ? 'text-yellow-400' : 'text-red-400'
              }>
                {ns.phase1.participation?.conviction || 'LOW'}
                {ns.phase1.participation?.effort_result_match === false && ' ⚠ Effort/result mismatch'}
              </span>
            </div>

            {/* Failure Patterns */}
            {ns.phase1.failure?.present && (
              <div className="text-red-400 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-red-500" />
                <span>FAILURE DETECTED: {ns.phase1.failure?.failure_types?.join(', ') || 'Unknown pattern'}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* KEY LEVELS - Always visible */}
      {ns.phase1.key_levels && ns.phase1.key_levels.current_price > 0 && (
        <div
          className="rounded-lg p-2 mb-2 bg-blue-900/20 border border-blue-500/30 cursor-pointer hover:bg-blue-900/30 transition-all"
          onClick={() => toggleExpand('levels')}
        >
          <div className="flex items-center justify-between">
            <span className="text-blue-400 text-xs font-medium">KEY LEVELS</span>
            <span className="text-gray-600 text-xs">ⓘ</span>
          </div>
          <div className="mt-1 grid grid-cols-3 gap-2 text-xs">
            <div>
              <span className="text-red-400">R1: </span>
              <span className="text-white font-mono">${ns.phase1.key_levels.pivot_r1?.toFixed(2) || '—'}</span>
            </div>
            <div>
              <span className="text-yellow-400">P: </span>
              <span className="text-white font-mono">${ns.phase1.key_levels.pivot?.toFixed(2) || '—'}</span>
            </div>
            <div>
              <span className="text-green-400">S1: </span>
              <span className="text-white font-mono">${ns.phase1.key_levels.pivot_s1?.toFixed(2) || '—'}</span>
            </div>
          </div>
          {expanded === 'levels' && (
            <div className="mt-2 pt-2 border-t border-blue-500/30 space-y-2 text-xs">
              {/* Visual Price Ladder with Key Levels */}
              <div className="bg-gray-800/50 rounded p-3">
                <div className="text-gray-400 mb-2">Price Ladder with Key Levels:</div>
                {(() => {
                  const lvl = ns.phase1.key_levels
                  // Calculate price range for visualization
                  const allLevels = [lvl.pivot_r1, lvl.pivot, lvl.pivot_s1, lvl.current_price, lvl.recent_high, lvl.recent_low].filter(p => p > 0)
                  const maxPrice = Math.max(...allLevels) * 1.002
                  const minPrice = Math.min(...allLevels) * 0.998
                  const priceRange = maxPrice - minPrice

                  const priceToPosition = (price: number) => {
                    if (priceRange === 0) return 50
                    return ((price - minPrice) / priceRange) * 100
                  }

                  return (
                    <div className="relative h-32 bg-gray-900 rounded border border-gray-700">
                      {/* R1 Line - Thick Red */}
                      {lvl.pivot_r1 > 0 && (
                        <div
                          className="absolute w-full h-1 bg-red-500 left-0"
                          style={{ bottom: `${priceToPosition(lvl.pivot_r1)}%` }}
                        >
                          <span className="absolute right-1 -top-4 text-red-400 font-mono text-xs">
                            R1 ${lvl.pivot_r1.toFixed(2)}
                          </span>
                        </div>
                      )}

                      {/* Pivot Line - Thick Yellow */}
                      {lvl.pivot > 0 && (
                        <div
                          className="absolute w-full h-1 bg-yellow-500 left-0"
                          style={{ bottom: `${priceToPosition(lvl.pivot)}%` }}
                        >
                          <span className="absolute right-1 -top-4 text-yellow-400 font-mono text-xs">
                            P ${lvl.pivot.toFixed(2)}
                          </span>
                        </div>
                      )}

                      {/* S1 Line - Thick Green */}
                      {lvl.pivot_s1 > 0 && (
                        <div
                          className="absolute w-full h-1 bg-green-500 left-0"
                          style={{ bottom: `${priceToPosition(lvl.pivot_s1)}%` }}
                        >
                          <span className="absolute right-1 -top-4 text-green-400 font-mono text-xs">
                            S1 ${lvl.pivot_s1.toFixed(2)}
                          </span>
                        </div>
                      )}

                      {/* Recent High - Dashed Red */}
                      {lvl.recent_high > 0 && (
                        <div
                          className="absolute w-full h-0.5 border-t-2 border-dashed border-red-400/50 left-0"
                          style={{ bottom: `${priceToPosition(lvl.recent_high)}%` }}
                        >
                          <span className="absolute left-1 -top-4 text-red-400/70 font-mono text-xs">
                            High ${lvl.recent_high.toFixed(2)}
                          </span>
                        </div>
                      )}

                      {/* Recent Low - Dashed Green */}
                      {lvl.recent_low > 0 && (
                        <div
                          className="absolute w-full h-0.5 border-t-2 border-dashed border-green-400/50 left-0"
                          style={{ bottom: `${priceToPosition(lvl.recent_low)}%` }}
                        >
                          <span className="absolute left-1 -top-4 text-green-400/70 font-mono text-xs">
                            Low ${lvl.recent_low.toFixed(2)}
                          </span>
                        </div>
                      )}

                      {/* Current Price - White Circle Marker */}
                      <div
                        className="absolute left-1/2 transform -translate-x-1/2 flex items-center"
                        style={{ bottom: `${priceToPosition(lvl.current_price)}%` }}
                      >
                        <div className="w-3 h-3 bg-white rounded-full border-2 border-blue-400 shadow-lg shadow-blue-400/50"></div>
                        <span className="ml-2 bg-blue-500 text-white px-1.5 py-0.5 rounded font-mono text-xs whitespace-nowrap">
                          ${lvl.current_price.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  )
                })()}
              </div>

              {/* Pivot Levels Detail */}
              <div className="grid grid-cols-2 gap-2 mt-2">
                <div>
                  <div className="text-gray-500 mb-1">Pivot Levels (from prev day):</div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-red-400">Resistance 1 (R1):</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.pivot_r1?.toFixed(2) || '—'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-yellow-400">Pivot Point (P):</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.pivot?.toFixed(2) || '—'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-green-400">Support 1 (S1):</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.pivot_s1?.toFixed(2) || '—'}</span>
                    </div>
                  </div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">Reference Prices:</div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Today Open:</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.today_open?.toFixed(2) || '—'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Prev Close:</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.prev_close?.toFixed(2) || '—'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Mid Point:</span>
                      <span className="text-white font-mono">${ns.phase1.key_levels.mid_point?.toFixed(2) || '—'}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Distance to levels */}
              {(() => {
                const lvl = ns.phase1.key_levels
                const distToR1 = lvl.pivot_r1 ? ((lvl.pivot_r1 - lvl.current_price) / lvl.current_price * 100) : 0
                const distToS1 = lvl.pivot_s1 ? ((lvl.current_price - lvl.pivot_s1) / lvl.current_price * 100) : 0
                const distToPivot = lvl.pivot ? ((lvl.pivot - lvl.current_price) / lvl.current_price * 100) : 0

                return (
                  <div className="mt-2 pt-2 border-t border-blue-500/30">
                    <div className="text-gray-400 mb-1">Distance to Levels:</div>
                    <div className="grid grid-cols-3 gap-2">
                      <div className={`text-center p-1 rounded ${Math.abs(distToR1) < 0.3 ? 'bg-red-500/30' : ''}`}>
                        <div className="text-red-400 text-xs">To R1</div>
                        <div className={`font-mono ${distToR1 > 0 ? 'text-red-400' : 'text-green-400'}`}>
                          {distToR1 > 0 ? '+' : ''}{distToR1.toFixed(2)}%
                        </div>
                      </div>
                      <div className={`text-center p-1 rounded ${Math.abs(distToPivot) < 0.2 ? 'bg-yellow-500/30' : ''}`}>
                        <div className="text-yellow-400 text-xs">To Pivot</div>
                        <div className={`font-mono ${distToPivot > 0 ? 'text-yellow-400' : 'text-gray-400'}`}>
                          {distToPivot > 0 ? '+' : ''}{distToPivot.toFixed(2)}%
                        </div>
                      </div>
                      <div className={`text-center p-1 rounded ${Math.abs(distToS1) < 0.3 ? 'bg-green-500/30' : ''}`}>
                        <div className="text-green-400 text-xs">To S1</div>
                        <div className={`font-mono ${distToS1 > 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {distToS1 > 0 ? '+' : ''}{distToS1.toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })()}
            </div>
          )}
        </div>
      )}

      {/* Health and Window - Clickable with SPECIFIC data */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div
          className={`rounded p-2 cursor-pointer transition-all ${
            ns.phase2.tier === 'HEALTHY' ? 'bg-green-500/10 hover:bg-green-500/20' :
            ns.phase2.tier === 'DEGRADED' ? 'bg-yellow-500/10 hover:bg-yellow-500/20' :
            'bg-red-500/10 hover:bg-red-500/20'
          }`}
          onClick={() => toggleExpand('health')}
        >
          <div className="flex items-center justify-between">
            <span className="text-gray-500">Signal Health</span>
            <span className="text-gray-600">ⓘ</span>
          </div>
          <div className={`font-medium ${
            ns.phase2.tier === 'HEALTHY' ? 'text-green-400' :
            ns.phase2.tier === 'DEGRADED' ? 'text-yellow-400' :
            'text-red-400'
          }`}>
            {ns.phase2.tier === 'HEALTHY' ? 'Healthy' :
             ns.phase2.tier === 'DEGRADED' ? 'Degraded' :
             'Unstable'} ({ns.phase2.health_score}%)
          </div>
          {expanded === 'health' && (
            <div className="mt-2 pt-2 border-t border-gray-700 space-y-2">
              {/* Health Dimensions Breakdown */}
              <div className="text-gray-400 mb-1">Health Score Breakdown:</div>
              <DimensionBar label="Structure" score={dims.structural_integrity} weight={0.30} />
              <DimensionBar label="Time Persist" score={dims.time_persistence} weight={0.15} />
              <DimensionBar label="Volatility" score={dims.volatility_alignment} weight={0.15} />
              <DimensionBar label="Participation" score={dims.participation_consistency} weight={0.20} />
              <DimensionBar label="Failure Risk" score={dims.failure_risk} weight={0.20} />

              {/* Specific Reasons */}
              {ns.phase2.reasons?.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-700">
                  <div className="text-gray-400 mb-1">Issues Detected:</div>
                  {ns.phase2.reasons.map((reason, i) => (
                    <div key={i} className="text-red-400 flex items-center gap-1">
                      <span>•</span> {reason}
                    </div>
                  ))}
                </div>
              )}

              {ns.phase2.stand_down && (
                <div className="mt-2 p-2 bg-red-500/20 rounded text-red-400 font-medium">
                  🛑 STAND DOWN - Health score too low to trade
                </div>
              )}
            </div>
          )}
        </div>

        <div
          className={`rounded p-2 cursor-pointer transition-all ${
            ns.phase3.throttle === 'OPEN' ? 'bg-green-500/10 hover:bg-green-500/20' :
            ns.phase3.throttle === 'LIMITED' ? 'bg-yellow-500/10 hover:bg-yellow-500/20' :
            'bg-red-500/10 hover:bg-red-500/20'
          }`}
          onClick={() => toggleExpand('window')}
        >
          <div className="flex items-center justify-between">
            <span className="text-gray-500">Trade Window</span>
            <span className="text-gray-600">ⓘ</span>
          </div>
          <div className={`font-medium ${
            ns.phase3.throttle === 'OPEN' ? 'text-green-400' :
            ns.phase3.throttle === 'LIMITED' ? 'text-yellow-400' :
            'text-red-400'
          }`}>
            {ns.phase3.throttle === 'OPEN' ? 'Open' :
             ns.phase3.throttle === 'LIMITED' ? 'Limited' :
             'Blocked'}
          </div>
          {expanded === 'window' && (
            <div className="mt-2 pt-2 border-t border-gray-700 space-y-2">
              {/* Density Details */}
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Density Score:</span>
                <span className={`font-mono ${
                  ns.phase3.density_score >= 70 ? 'text-green-400' :
                  ns.phase3.density_score >= 40 ? 'text-yellow-400' : 'text-red-400'
                }`}>{ns.phase3.density_score}%</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-400">Signals Allowed:</span>
                <span className={`font-mono ${
                  ns.phase3.allowed_signals > 1 ? 'text-green-400' :
                  ns.phase3.allowed_signals === 1 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {ns.phase3.allowed_signals === 999 ? 'Unlimited' :
                   ns.phase3.allowed_signals === 0 ? 'None' :
                   ns.phase3.allowed_signals}
                </span>
              </div>

              {/* Specific Reasons */}
              {ns.phase3.reasons?.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-700">
                  <div className="text-gray-400 mb-1">Throttle Reasons:</div>
                  {ns.phase3.reasons.map((reason, i) => (
                    <div key={i} className="text-yellow-400 flex items-center gap-1">
                      <span>•</span> {reason}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Execution guidance - Clickable with SPECIFIC data */}
      {ns.phase4.allowed && (
        <div
          className="mt-2 p-2 bg-gray-800/50 rounded text-xs cursor-pointer hover:bg-gray-800/70 transition-all"
          onClick={() => toggleExpand('execution')}
        >
          <div className="flex justify-between items-center">
            <span className="text-gray-400">
              Play: <span className="text-white">
                {ns.phase4.execution_mode === 'TREND_CONTINUATION' ? 'Trend Following' :
                 ns.phase4.execution_mode === 'MEAN_REVERSION' ? 'Mean Reversion' :
                 ns.phase4.execution_mode === 'SCALP' ? 'Quick Scalp' :
                 'No Trade'}
              </span>
            </span>
            <div className="flex items-center gap-2">
              <span className={`${
                ns.phase4.risk_state === 'NORMAL' ? 'text-green-400' :
                ns.phase4.risk_state === 'REDUCED' ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                Risk: {ns.phase4.risk_state === 'NORMAL' ? 'Normal Size' :
                       ns.phase4.risk_state === 'REDUCED' ? 'Half Size' :
                       'Quarter Size'}
              </span>
              <span className="text-gray-600">ⓘ</span>
            </div>
          </div>
          {expanded === 'execution' && (
            <div className="mt-2 pt-2 border-t border-gray-700 space-y-2">
              {/* Specific Execution Parameters */}
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <div className="text-gray-500">Bias</div>
                  <div className={`font-medium ${
                    ns.phase4.bias === 'LONG' ? 'text-green-400' :
                    ns.phase4.bias === 'SHORT' ? 'text-red-400' : 'text-gray-400'
                  }`}>{ns.phase4.bias}</div>
                </div>
                <div>
                  <div className="text-gray-500">Mode</div>
                  <div className="text-white">{ns.phase4.execution_mode?.replace('_', ' ')}</div>
                </div>
              </div>

              {/* Invalidation Context */}
              {ns.phase4.invalidation_context?.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-700">
                  <div className="text-gray-400 mb-1">Exit if you see:</div>
                  {ns.phase4.invalidation_context.map((ctx, i) => (
                    <div key={i} className="text-orange-400 flex items-center gap-1">
                      <span>⚠</span> {ctx}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {!ns.phase4.allowed && (
        <div
          className="mt-2 p-2 bg-red-900/20 border border-red-500/30 rounded text-xs text-red-400 cursor-pointer hover:bg-red-900/30"
          onClick={() => toggleExpand('blocked')}
        >
          <div className="flex items-center justify-between">
            <span>⚠ Market conditions unfavorable - wait for better setup</span>
            <span className="text-gray-600">ⓘ</span>
          </div>
          {expanded === 'blocked' && (
            <div className="mt-2 pt-2 border-t border-red-500/30 space-y-2">
              <div className="text-gray-400 mb-1">Blocking Factors:</div>

              {/* Show specific scores that caused the block */}
              {ns.phase2.health_score < 45 && (
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-500" />
                  <span>Health score {ns.phase2.health_score}% (needs ≥45%)</span>
                </div>
              )}

              {ns.phase2.stand_down && (
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-500" />
                  <span>Stand-down flag active from health gate</span>
                </div>
              )}

              {ns.phase3.throttle === 'BLOCKED' && (
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-500" />
                  <span>Density score {ns.phase3.density_score}% (needs ≥40%)</span>
                </div>
              )}

              {ns.phase1.confidence_band === 'NO_TRADE' && (
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-500" />
                  <span>No structural edge - confidence band: {ns.phase1.confidence_band}</span>
                </div>
              )}

              {ns.phase1.failure?.present && (
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-500" />
                  <span>Failure patterns: {ns.phase1.failure?.failure_types?.join(', ') || 'detected'}</span>
                </div>
              )}

              {/* Show what needs to improve */}
              <div className="mt-2 pt-2 border-t border-red-500/30 text-gray-400">
                Wait for: Higher health score, clearer structure, or failure patterns to resolve.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function ReplayPage() {
  const router = useRouter()
  const [date, setDate] = useState('2025-12-30')
  const [hour, setHour] = useState(9)
  const [minute, setMinute] = useState(30)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1000)
  const [data, setData] = useState<ReplayData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  // Chart state
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null)
  const [chartData, setChartData] = useState<CandleData[]>([])
  const [chartLoading, setChartLoading] = useState(false)
  const [chartTimeframe, setChartTimeframe] = useState<'1m' | '5m' | '15m' | '1h'>('5m')

  // Fetch chart data when ticker is selected
  // Fetches only the current replay day's data
  const fetchChartData = useCallback(async (ticker: string, timeframe: string) => {
    setChartLoading(true)
    try {
      const endTime = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`

      // Fetch single day data for the replay date
      const response = await fetch(
        `/api/v2/data/market?ticker=${ticker}&timeframe=${timeframe}&date=${date}&endTime=${endTime}&limit=500`
      )
      if (!response.ok) throw new Error('Failed to fetch chart data')
      const result = await response.json()

      if (result.success && result.data) {
        setChartData(result.data)
        console.log(`[Chart] Loaded ${result.data.length} ${timeframe} bars for ${date} (source: ${result.source})`)
      }
    } catch (err) {
      console.error('Error fetching chart data:', err)
    } finally {
      setChartLoading(false)
    }
  }, [date, hour, minute])

  // Fetch chart data when selected ticker, timeframe, or time changes
  // Debounced to prevent excessive API calls during fast playback
  useEffect(() => {
    if (!selectedTicker) return

    const timer = setTimeout(() => {
      fetchChartData(selectedTicker, chartTimeframe)
    }, 300) // 300ms debounce

    return () => clearTimeout(timer)
  }, [selectedTicker, chartTimeframe, hour, minute, fetchChartData])

  const totalMinutes = hour * 60 + minute
  const marketOpenMinutes = 9 * 60 + 30
  const marketCloseMinutes = 16 * 60

  const formatTime = (h: number, m: number) => {
    const period = h >= 12 ? 'PM' : 'AM'
    const displayHour = h > 12 ? h - 12 : h === 0 ? 12 : h
    return `${displayHour}:${m.toString().padStart(2, '0')} ${period}`
  }

  const fetchReplayData = useCallback(async () => {
    setLoading(true)
    try {
      const timeStr = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`
      const response = await fetch(`/api/v2/replay?date=${date}&time=${timeStr}`)
      if (!response.ok) throw new Error('Failed to fetch replay data')
      const result = await response.json()
      if (result.error) {
        setError(result.error)
      } else {
        setData(result)
        setError(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [date, hour, minute])

  useEffect(() => {
    const timer = setTimeout(fetchReplayData, 300)
    return () => clearTimeout(timer)
  }, [fetchReplayData])

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setMinute(prev => {
          if (prev >= 59) {
            setHour(h => {
              if (h >= 16) {
                setIsPlaying(false)
                return 16
              }
              return h + 1
            })
            return 0
          }
          return prev + 1
        })
      }, speed)
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [isPlaying, speed])

  const handleSliderChange = (value: number) => {
    const h = Math.floor(value / 60)
    const m = value % 60
    setHour(h)
    setMinute(m)
  }

  const getActionColor = (action: string) => {
    switch (action) {
      case 'LONG': return 'bg-green-500 text-white'
      case 'SHORT': return 'bg-red-500 text-white'
      default: return 'bg-gray-600 text-gray-300'
    }
  }

  const getActionStyle = (action: string, isBest: boolean) => {
    const base = isBest ? 'ring-2 ring-cyan-400' : ''
    switch (action) {
      case 'LONG': return `bg-green-900/40 border-green-500/50 ${base}`
      case 'SHORT': return `bg-red-900/40 border-red-500/50 ${base}`
      default: return `bg-gray-900/40 border-gray-700 ${base}`
    }
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold flex items-center gap-3">
                <span className="text-purple-400">REPLAY MODE</span>
                <span className="text-sm bg-purple-500/20 text-purple-300 px-2 py-1 rounded">
                  Time Travel Testing
                </span>
              </h1>
              <p className="text-gray-400 text-sm mt-1">
                Validate signals by replaying historical trading days
              </p>
            </div>
            <button
              onClick={() => router.push('/dashboard')}
              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm"
            >
              Back to Live
            </button>
          </div>
        </div>
      </div>

      {/* Time Controls */}
      <div className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center gap-6 mb-4">
            <div className="flex items-center gap-2">
              <label className="text-gray-400 text-sm">Date:</label>
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-white"
              />
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => { setHour(9); setMinute(30) }}
                className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
              >
                Reset
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`px-4 py-2 rounded font-medium ${
                  isPlaying ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
                }`}
              >
                {isPlaying ? '⏸ Pause' : '▶ Play'}
              </button>
            </div>

            <div className="flex items-center gap-2">
              <label className="text-gray-400 text-sm">Speed:</label>
              {SPEEDS.map((s) => (
                <button
                  key={s.label}
                  onClick={() => setSpeed(s.value)}
                  className={`px-2 py-1 rounded text-xs ${
                    speed === s.value ? 'bg-purple-500 text-white' : 'bg-gray-800 text-gray-400'
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-4xl font-mono text-cyan-400 w-40">
              {formatTime(hour, minute)}
            </div>
            <div className="flex-1">
              <input
                type="range"
                min={marketOpenMinutes}
                max={marketCloseMinutes}
                value={totalMinutes}
                onChange={(e) => handleSliderChange(parseInt(e.target.value))}
                className="w-full h-3 bg-gray-800 rounded-lg appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5
                  [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-cyan-400 [&::-webkit-slider-thumb]:cursor-pointer"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>9:30 AM</span>
                <span>11:00 AM</span>
                <span>12:00 PM (Late)</span>
                <span>2:00 PM</span>
                <span>4:00 PM</span>
              </div>
            </div>
            <div className={`px-3 py-1 rounded text-sm font-medium ${
              data?.session === 'late' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
            }`}>
              {data?.session === 'late' ? 'LATE SESSION' : 'EARLY SESSION'}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        {loading && !data && (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin h-8 w-8 border-4 border-purple-400 border-t-transparent rounded-full"></div>
            <span className="ml-3 text-gray-400">Loading replay data...</span>
          </div>
        )}

        {error && (
          <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 mb-6">
            <div className="text-red-400">{error}</div>
          </div>
        )}

        {data && (
          <>
            {/* Chart Section - Shows when a ticker is selected */}
            {selectedTicker && data.tickers[selectedTicker] && (
              <div className="mb-6 bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 bg-gray-800/50">
                  <div className="flex items-center gap-3">
                    <span className="text-xl font-bold text-cyan-400">{selectedTicker}</span>
                    {/* Timeframe selector */}
                    <div className="flex items-center gap-1 bg-gray-900 rounded-lg p-1">
                      {(['1m', '5m', '15m', '1h'] as const).map((tf) => (
                        <button
                          key={tf}
                          onClick={() => setChartTimeframe(tf)}
                          className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                            chartTimeframe === tf
                              ? 'bg-cyan-600 text-white'
                              : 'text-gray-400 hover:text-white hover:bg-gray-700'
                          }`}
                        >
                          {tf}
                        </button>
                      ))}
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedTicker(null)}
                    className="text-gray-400 hover:text-white p-1"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <div className="h-[500px] relative">
                  {chartLoading ? (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="animate-spin h-8 w-8 border-4 border-cyan-400 border-t-transparent rounded-full"></div>
                    </div>
                  ) : chartData.length > 0 ? (
                    <ProfessionalChart
                      symbol={selectedTicker}
                      data={chartData}
                      currentPrice={data.tickers[selectedTicker].current_price}
                      stopLoss={data.tickers[selectedTicker].northstar?.phase1?.key_levels?.pivot_s1}
                      entryPoint={data.tickers[selectedTicker].northstar?.phase1?.key_levels?.pivot}
                      targets={data.tickers[selectedTicker].northstar?.phase1?.key_levels?.pivot_r1
                        ? [data.tickers[selectedTicker].northstar.phase1.key_levels.pivot_r1]
                        : []}
                      showFvg={true}
                      fvgPercentage={0.2}
                      showMLOverlay={true}
                      v6Prediction={(() => {
                        const v6 = data.tickers[selectedTicker]?.v6
                        if (!v6) return undefined
                        const probA = v6.probability_a || 0.5
                        const probB = v6.probability_b || 0.5
                        const activeProb = data.session === 'early' ? probA : probB
                        return {
                          direction: v6.action === 'LONG' ? 'BULLISH' : v6.action === 'SHORT' ? 'BEARISH' : 'NEUTRAL',
                          probability_a: probA,
                          probability_b: probB,
                          confidence: Math.round(Math.abs(activeProb - 0.5) * 200),
                          session: data.session,
                          action: v6.action === 'LONG' ? 'BUY_CALL' : v6.action === 'SHORT' ? 'BUY_PUT' : 'NO_TRADE',
                        } as V6Prediction
                      })()}
                      price11am={data.tickers[selectedTicker]?.v6?.price_11am || undefined}
                      rangeHigh={data.tickers[selectedTicker].northstar?.phase1?.key_levels?.recent_high}
                      rangeLow={data.tickers[selectedTicker].northstar?.phase1?.key_levels?.recent_low}
                      rangeMid={data.tickers[selectedTicker].northstar?.phase1?.key_levels?.mid_point}
                      retestHigh={data.tickers[selectedTicker].northstar?.phase1?.key_levels?.retest_high}
                      retestLow={data.tickers[selectedTicker].northstar?.phase1?.key_levels?.retest_low}
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                      No chart data available for this date/time
                    </div>
                  )}
                </div>
                {/* Key Levels Legend */}
                {data.tickers[selectedTicker].northstar?.phase1?.key_levels && (
                  <div className="px-4 py-2 border-t border-gray-800 bg-gray-800/30 flex flex-wrap items-center gap-4 text-xs">
                    {/* Pivot levels */}
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-1 bg-red-500 rounded"></div>
                      <span className="text-gray-400">R1:</span>
                      <span className="text-red-400 font-mono">
                        ${data.tickers[selectedTicker].northstar.phase1.key_levels.pivot_r1?.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-1 bg-cyan-500 rounded"></div>
                      <span className="text-gray-400">Pivot:</span>
                      <span className="text-cyan-400 font-mono">
                        ${data.tickers[selectedTicker].northstar.phase1.key_levels.pivot?.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-1 bg-green-500 rounded"></div>
                      <span className="text-gray-400">S1:</span>
                      <span className="text-green-400 font-mono">
                        ${data.tickers[selectedTicker].northstar.phase1.key_levels.pivot_s1?.toFixed(2)}
                      </span>
                    </div>
                    {/* Range levels (acceptance zone) */}
                    <div className="flex items-center gap-2 border-l border-gray-700 pl-4">
                      <div className="w-4 h-1 bg-amber-500 rounded" style={{ borderStyle: 'dashed' }}></div>
                      <span className="text-gray-400">Range:</span>
                      <span className="text-amber-400 font-mono">
                        ${data.tickers[selectedTicker].northstar.phase1.key_levels.recent_low?.toFixed(2)} - ${data.tickers[selectedTicker].northstar.phase1.key_levels.recent_high?.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-400">Mid:</span>
                      <span className="text-amber-300 font-mono">
                        ${data.tickers[selectedTicker].northstar.phase1.key_levels.mid_point?.toFixed(2)}
                      </span>
                    </div>
                    <div className="ml-auto text-gray-500">
                      Click ticker to close
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Summary Banner */}
            <div className={`mb-6 p-4 rounded-lg border ${
              data.summary.allowed_tickers.length > 0 ? 'bg-green-500/10 border-green-500/50' : 'bg-red-500/10 border-red-500/50'
            }`}>
              <div className="flex justify-between items-center">
                <div>
                  <div className="text-lg font-bold">{data.summary.recommendation}</div>
                  <div className="text-sm text-gray-400">
                    V6: {data.summary.v6_actionable.join(', ') || 'None'} |
                    Northstar: {data.summary.allowed_tickers.join(', ') || 'None'}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-gray-400 text-sm">{data.replay_date}</div>
                  <div className="text-cyan-400 font-mono text-xl">{data.simulated_time_et}</div>
                </div>
              </div>
            </div>

            {/* Ticker Cards - Matching Dashboard Layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(data.tickers).map(([ticker, tickerData]) => {
                if (tickerData.error) {
                  return (
                    <div key={ticker} className="bg-gray-900 border border-gray-800 rounded-xl p-5">
                      <h3 className="text-xl font-bold text-white mb-2">{ticker}</h3>
                      <div className="text-red-400">{tickerData.error}</div>
                    </div>
                  )
                }

                const v6 = tickerData.v6
                const ns = tickerData.northstar
                const isBest = ticker === data.summary.best_ticker
                const session = data.session
                const probA = v6.probability_a || 0.5
                const probB = v6.probability_b || 0.5

                return (
                  <div
                    key={ticker}
                    className={`border rounded-xl p-5 transition-all ${getActionStyle(v6.action, isBest)}`}
                  >
                    {/* Header */}
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setSelectedTicker(selectedTicker === ticker ? null : ticker)}
                            className={`text-2xl font-bold transition-colors flex items-center gap-1 ${
                              selectedTicker === ticker
                                ? 'text-cyan-400'
                                : 'text-purple-400 hover:text-purple-300'
                            }`}
                            title="Click to view chart"
                          >
                            {ticker}
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                          </button>
                          {isBest && <span className="text-cyan-400 text-xs">BEST</span>}
                        </div>
                        <div className="text-3xl font-light text-white mt-1">
                          ${tickerData.current_price.toFixed(2)}
                        </div>
                        <div className={`text-sm ${tickerData.today_change_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {tickerData.today_change_pct >= 0 ? '▲' : '▼'} {Math.abs(tickerData.today_change_pct).toFixed(2)}%
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
                                {tickerData.price_11am && (
                                  <span className="text-xs text-cyan-400 font-mono">
                                    ${tickerData.price_11am.toFixed(2)}
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
                                    ${tickerData.today_open.toFixed(2)}
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
                                  ${tickerData.today_open.toFixed(2)}
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
                              {hour < 12 ? `${12 - hour}h ${60 - minute}m` : 'Active'}
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

                    {/* Northstar Pipeline Summary - User Friendly with Expandable Details */}
                    <NorthstarDetails ns={ns} />

                    {/* Footer */}
                    <div className="mt-3 pt-2 border-t border-gray-800 text-xs text-gray-600">
                      {tickerData.bars_analyzed} bars analyzed | {v6.reason}
                    </div>
                  </div>
                )
              })}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
