'use client'

import { useState, useEffect } from 'react'

// Key Levels type for support/resistance
interface KeyLevels {
  recent_high: number
  recent_low: number
  mid_point: number
  pivot: number
  pivot_r1: number
  pivot_s1: number
  current_price: number
  today_open: number
  prev_close: number
}

// Phase 1 types
interface Phase1 {
  resolved: boolean
  direction: 'UP' | 'DOWN' | 'BALANCED'
  dominant_timeframe: string
  confidence_band: 'NO_TRADE' | 'CONTEXT_ONLY' | 'STRUCTURAL_EDGE'
  acceptance: {
    accepted: boolean
    acceptance_strength: 'WEAK' | 'MODERATE' | 'STRONG'
    acceptance_reason?: string
    failed_levels: string[]
  }
  range: {
    state: 'TREND' | 'BALANCE' | 'FAILED_EXPANSION'
    rotation_complete: boolean
    expansion_quality: 'CLEAN' | 'DIRTY' | 'NONE'
  }
  mtf: {
    aligned: boolean
    dominant_tf: string
    conflict_tf: string | null
  }
  participation: {
    conviction: 'HIGH' | 'MEDIUM' | 'LOW'
    effort_result_match: boolean
  }
  failure: {
    present: boolean
    failure_types: string[]
  }
  key_levels?: KeyLevels
}

// Phase 2 types
interface Phase2 {
  health_score: number
  tier: 'HEALTHY' | 'DEGRADED' | 'UNSTABLE'
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

// Phase 3 types
interface Phase3 {
  density_score: number
  throttle: 'OPEN' | 'LIMITED' | 'BLOCKED'
  allowed_signals: number
  reasons: string[]
}

// Phase 4 types
interface Phase4 {
  allowed: boolean
  bias: 'LONG' | 'SHORT' | 'NEUTRAL'
  execution_mode: 'TREND_CONTINUATION' | 'MEAN_REVERSION' | 'SCALP' | 'NO_TRADE'
  risk_state: 'NORMAL' | 'REDUCED' | 'DEFENSIVE'
  invalidation_context: string[]
}

interface TickerAnalysis {
  timestamp: string
  symbol: string
  phase1: Phase1
  phase2: Phase2
  phase3: Phase3
  phase4: Phase4
  current_price: number
  today_open: number
  today_change_pct: number
  bars_analyzed: number
  error?: string
}

interface NorthstarData {
  generated_at: string
  current_time_et: string
  current_hour: number
  market_open: boolean
  tickers: Record<string, TickerAnalysis>
  pipeline_version: string
  summary: {
    execution_allowed: boolean
    allowed_tickers: Array<{
      ticker: string
      bias: string
      mode: string
      risk: string
    }>
    recommendation: string
  }
}

export function NorthstarPanel() {
  const [data, setData] = useState<NorthstarData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedTicker, setSelectedTicker] = useState<string>('SPY')
  const [expandedPhase, setExpandedPhase] = useState<number | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/api/v2/northstar')
        if (!response.ok) throw new Error('Failed to fetch Northstar data')
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
    }

    fetchData()
    const interval = setInterval(fetchData, 60000) // Refresh every minute
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="flex items-center gap-2">
          <div className="animate-spin h-4 w-4 border-2 border-purple-400 border-t-transparent rounded-full"></div>
          <span className="text-gray-400 text-sm">Loading Northstar pipeline...</span>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="text-red-400 text-sm">{error || 'No data available'}</div>
        <div className="text-gray-500 text-xs mt-1">Ensure ML server is running</div>
      </div>
    )
  }

  const tickerData = data.tickers[selectedTicker]

  const getPhaseColor = (phase: number) => {
    if (!tickerData) return 'border-gray-700'
    switch (phase) {
      case 1:
        return tickerData.phase1.confidence_band === 'STRUCTURAL_EDGE' ? 'border-green-500' :
               tickerData.phase1.confidence_band === 'CONTEXT_ONLY' ? 'border-yellow-500' : 'border-red-500'
      case 2:
        return tickerData.phase2.tier === 'HEALTHY' ? 'border-green-500' :
               tickerData.phase2.tier === 'DEGRADED' ? 'border-yellow-500' : 'border-red-500'
      case 3:
        return tickerData.phase3.throttle === 'OPEN' ? 'border-green-500' :
               tickerData.phase3.throttle === 'LIMITED' ? 'border-yellow-500' : 'border-red-500'
      case 4:
        return tickerData.phase4.allowed ? 'border-green-500' : 'border-red-500'
      default:
        return 'border-gray-700'
    }
  }

  const getDirectionIcon = (dir: string) => {
    if (dir === 'UP') return '▲'
    if (dir === 'DOWN') return '▼'
    return '◆'
  }

  const getDirectionColor = (dir: string) => {
    if (dir === 'UP') return 'text-green-400'
    if (dir === 'DOWN') return 'text-red-400'
    return 'text-gray-400'
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/30 to-indigo-900/30 px-4 py-3 border-b border-gray-800">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-white font-semibold flex items-center gap-2">
              <span className="text-purple-400">Northstar</span> Phase Pipeline
              <span className="px-2 py-0.5 rounded text-xs bg-purple-500/20 text-purple-400">
                v{data.pipeline_version}
              </span>
            </h3>
            <p className="text-gray-400 text-xs mt-0.5">{data.current_time_et}</p>
          </div>
          <div className="flex gap-2">
            {Object.keys(data.tickers).map((ticker) => (
              <button
                key={ticker}
                onClick={() => setSelectedTicker(ticker)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  selectedTicker === ticker
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {ticker}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Summary Banner */}
      <div className={`px-4 py-2 border-b border-gray-800 ${
        data.summary.execution_allowed ? 'bg-green-900/20' : 'bg-red-900/20'
      }`}>
        <div className="flex items-center justify-between">
          <span className={`text-sm font-medium ${
            data.summary.execution_allowed ? 'text-green-400' : 'text-red-400'
          }`}>
            {data.summary.execution_allowed ? 'EXECUTION ALLOWED' : 'STAND DOWN'}
          </span>
          <span className="text-gray-400 text-sm">{data.summary.recommendation}</span>
        </div>
      </div>

      {tickerData && !tickerData.error && (
        <div className="p-4">
          {/* Price Info with Clickable Ticker */}
          <div className="flex justify-between items-center mb-4 pb-3 border-b border-gray-800">
            <div className="flex items-center gap-3">
              <a
                href={`https://www.tradingview.com/chart/?symbol=${selectedTicker}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-purple-400 hover:text-purple-300 transition-colors flex items-center gap-1"
                title="Open TradingView chart"
              >
                <span className="text-lg font-bold">{selectedTicker}</span>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </a>
              <span className="text-white text-2xl font-bold">${tickerData.current_price}</span>
              <span className={`text-sm ${tickerData.today_change_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {tickerData.today_change_pct >= 0 ? '+' : ''}{tickerData.today_change_pct?.toFixed(2)}%
              </span>
            </div>
            <div className="text-gray-500 text-xs">
              {tickerData.bars_analyzed} bars analyzed
            </div>
          </div>

          {/* 4 Phase Cards */}
          <div className="space-y-3">
            {/* Phase 1: Reality State (Truth) */}
            <div
              className={`border rounded-lg p-3 cursor-pointer transition-all ${getPhaseColor(1)} ${
                expandedPhase === 1 ? 'bg-gray-800/50' : 'hover:bg-gray-800/30'
              }`}
              onClick={() => setExpandedPhase(expandedPhase === 1 ? null : 1)}
            >
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs font-mono">P1</span>
                  <span className="text-white font-medium">TRUTH</span>
                  <span className={`text-sm ${getDirectionColor(tickerData.phase1.direction)}`}>
                    {getDirectionIcon(tickerData.phase1.direction)} {tickerData.phase1.direction}
                  </span>
                </div>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  tickerData.phase1.confidence_band === 'STRUCTURAL_EDGE' ? 'bg-green-500/20 text-green-400' :
                  tickerData.phase1.confidence_band === 'CONTEXT_ONLY' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-red-500/20 text-red-400'
                }`}>
                  {tickerData.phase1.confidence_band.replace('_', ' ')}
                </span>
              </div>

              {expandedPhase === 1 && (
                <div className="mt-3 pt-3 border-t border-gray-700 space-y-2 text-xs">
                  {/* Acceptance Details */}
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${tickerData.phase1.acceptance.accepted ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className="text-gray-400">Acceptance:</span>
                    <span className={tickerData.phase1.acceptance.accepted ? 'text-green-400' : 'text-red-400'}>
                      {tickerData.phase1.acceptance.accepted
                        ? `YES - ${tickerData.phase1.acceptance.acceptance_strength}`
                        : 'NO'}
                    </span>
                  </div>

                  {/* Acceptance Reason - WHY */}
                  {tickerData.phase1.acceptance.acceptance_reason && (
                    <div className="text-gray-400 pl-4 text-xs italic">
                      {tickerData.phase1.acceptance.acceptance_reason}
                    </div>
                  )}

                  {/* Failed Levels */}
                  {tickerData.phase1.acceptance.failed_levels?.length > 0 && (
                    <div className="text-red-400 pl-4">
                      ⚠ {tickerData.phase1.acceptance.failed_levels.join(', ')}
                    </div>
                  )}

                  {/* Range State */}
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${
                      tickerData.phase1.range.state === 'TREND' ? 'bg-green-500' :
                      tickerData.phase1.range.state === 'BALANCE' ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                    <span className="text-gray-400">Range:</span>
                    <span className={
                      tickerData.phase1.range.state === 'TREND' ? 'text-green-400' :
                      tickerData.phase1.range.state === 'BALANCE' ? 'text-yellow-400' : 'text-red-400'
                    }>
                      {tickerData.phase1.range.state}
                      {tickerData.phase1.range.expansion_quality !== 'NONE' && ` (${tickerData.phase1.range.expansion_quality})`}
                      {tickerData.phase1.range.rotation_complete && ' - Rotation complete'}
                    </span>
                  </div>

                  {/* MTF Alignment */}
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${tickerData.phase1.mtf.aligned ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className="text-gray-400">MTF:</span>
                    <span className={tickerData.phase1.mtf.aligned ? 'text-green-400' : 'text-red-400'}>
                      {tickerData.phase1.mtf.aligned
                        ? `ALIGNED - ${tickerData.phase1.mtf.dominant_tf} dominant`
                        : `CONFLICT - ${tickerData.phase1.mtf.conflict_tf || 'Timeframes disagree'}`}
                    </span>
                  </div>

                  {/* Participation */}
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${
                      tickerData.phase1.participation.conviction === 'HIGH' ? 'bg-green-500' :
                      tickerData.phase1.participation.conviction === 'MEDIUM' ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                    <span className="text-gray-400">Conviction:</span>
                    <span className={
                      tickerData.phase1.participation.conviction === 'HIGH' ? 'text-green-400' :
                      tickerData.phase1.participation.conviction === 'MEDIUM' ? 'text-yellow-400' : 'text-red-400'
                    }>
                      {tickerData.phase1.participation.conviction}
                      {!tickerData.phase1.participation.effort_result_match && ' ⚠ Effort/result mismatch'}
                    </span>
                  </div>

                  {/* Failure Patterns */}
                  {tickerData.phase1.failure.present && (
                    <div className="text-red-400 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-red-500" />
                      FAILURE: {tickerData.phase1.failure.failure_types.join(', ')}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* KEY LEVELS - Between Phase 1 and 2 */}
            {tickerData.phase1.key_levels && tickerData.phase1.key_levels.current_price > 0 && (
              <div
                className={`border rounded-lg p-3 cursor-pointer transition-all border-blue-500/50 ${
                  expandedPhase === 5 ? 'bg-blue-900/20' : 'hover:bg-blue-900/10'
                }`}
                onClick={() => setExpandedPhase(expandedPhase === 5 ? null : 5)}
              >
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <span className="text-blue-400 text-xs font-medium">KEY LEVELS</span>
                  </div>
                  <div className="flex gap-3 text-xs">
                    <span><span className="text-red-400">R1:</span> <span className="text-white font-mono">${tickerData.phase1.key_levels.pivot_r1?.toFixed(2)}</span></span>
                    <span><span className="text-yellow-400">P:</span> <span className="text-white font-mono">${tickerData.phase1.key_levels.pivot?.toFixed(2)}</span></span>
                    <span><span className="text-green-400">S1:</span> <span className="text-white font-mono">${tickerData.phase1.key_levels.pivot_s1?.toFixed(2)}</span></span>
                  </div>
                </div>

                {expandedPhase === 5 && (
                  <div className="mt-3 pt-3 border-t border-blue-500/30 space-y-3 text-xs">
                    {/* Visual Price Ladder with Key Levels */}
                    {(() => {
                      const lvl = tickerData.phase1.key_levels!
                      // Calculate price range for visualization (from S1 to R1 with padding)
                      const allLevels = [lvl.pivot_r1, lvl.pivot, lvl.pivot_s1, lvl.current_price, lvl.recent_high, lvl.recent_low].filter(p => p > 0)
                      const maxPrice = Math.max(...allLevels) * 1.002
                      const minPrice = Math.min(...allLevels) * 0.998
                      const priceRange = maxPrice - minPrice

                      // Convert price to percentage position (0 = bottom, 100 = top)
                      const priceToPosition = (price: number) => {
                        if (priceRange === 0) return 50
                        return ((price - minPrice) / priceRange) * 100
                      }

                      return (
                        <div className="bg-gray-800/50 rounded p-3">
                          <div className="text-gray-400 mb-2">Price Ladder with Key Levels:</div>
                          <div className="relative h-40 bg-gray-900 rounded border border-gray-700">
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
                              <div className="w-4 h-4 bg-white rounded-full border-2 border-blue-400 shadow-lg shadow-blue-400/50"></div>
                              <span className="ml-2 bg-blue-500 text-white px-2 py-0.5 rounded font-mono text-xs whitespace-nowrap">
                                ${lvl.current_price.toFixed(2)}
                              </span>
                            </div>
                          </div>
                        </div>
                      )
                    })()}

                    {/* Levels Grid */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="text-gray-500 mb-1">Pivot Levels:</div>
                        <div className="space-y-1">
                          <div className="flex justify-between"><span className="text-red-400">R1:</span><span className="font-mono">${tickerData.phase1.key_levels.pivot_r1?.toFixed(2)}</span></div>
                          <div className="flex justify-between"><span className="text-yellow-400">Pivot:</span><span className="font-mono">${tickerData.phase1.key_levels.pivot?.toFixed(2)}</span></div>
                          <div className="flex justify-between"><span className="text-green-400">S1:</span><span className="font-mono">${tickerData.phase1.key_levels.pivot_s1?.toFixed(2)}</span></div>
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-500 mb-1">Reference:</div>
                        <div className="space-y-1">
                          <div className="flex justify-between"><span className="text-gray-400">Open:</span><span className="font-mono">${tickerData.phase1.key_levels.today_open?.toFixed(2)}</span></div>
                          <div className="flex justify-between"><span className="text-gray-400">Prev Close:</span><span className="font-mono">${tickerData.phase1.key_levels.prev_close?.toFixed(2)}</span></div>
                          <div className="flex justify-between"><span className="text-gray-400">Mid:</span><span className="font-mono">${tickerData.phase1.key_levels.mid_point?.toFixed(2)}</span></div>
                        </div>
                      </div>
                    </div>

                    {/* Distance to Levels */}
                    {(() => {
                      const lvl = tickerData.phase1.key_levels!
                      const distToR1 = lvl.pivot_r1 ? ((lvl.pivot_r1 - lvl.current_price) / lvl.current_price * 100) : 0
                      const distToS1 = lvl.pivot_s1 ? ((lvl.current_price - lvl.pivot_s1) / lvl.current_price * 100) : 0
                      const distToPivot = lvl.pivot ? ((lvl.pivot - lvl.current_price) / lvl.current_price * 100) : 0
                      return (
                        <div className="pt-2 border-t border-blue-500/30">
                          <div className="text-gray-400 mb-1">Distance:</div>
                          <div className="grid grid-cols-3 gap-2">
                            <div className={`text-center p-1 rounded ${Math.abs(distToR1) < 0.3 ? 'bg-red-500/30' : ''}`}>
                              <div className="text-red-400">To R1</div>
                              <div className={`font-mono ${distToR1 > 0 ? 'text-red-400' : 'text-green-400'}`}>
                                {distToR1 > 0 ? '+' : ''}{distToR1.toFixed(2)}%
                              </div>
                            </div>
                            <div className={`text-center p-1 rounded ${Math.abs(distToPivot) < 0.2 ? 'bg-yellow-500/30' : ''}`}>
                              <div className="text-yellow-400">To P</div>
                              <div className="font-mono text-yellow-400">{distToPivot > 0 ? '+' : ''}{distToPivot.toFixed(2)}%</div>
                            </div>
                            <div className={`text-center p-1 rounded ${Math.abs(distToS1) < 0.3 ? 'bg-green-500/30' : ''}`}>
                              <div className="text-green-400">To S1</div>
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

            {/* Phase 2: Health Gate */}
            <div
              className={`border rounded-lg p-3 cursor-pointer transition-all ${getPhaseColor(2)} ${
                expandedPhase === 2 ? 'bg-gray-800/50' : 'hover:bg-gray-800/30'
              }`}
              onClick={() => setExpandedPhase(expandedPhase === 2 ? null : 2)}
            >
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs font-mono">P2</span>
                  <span className="text-white font-medium">HEALTH GATE</span>
                  <span className="text-gray-400 text-sm">{tickerData.phase2.health_score}%</span>
                </div>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  tickerData.phase2.tier === 'HEALTHY' ? 'bg-green-500/20 text-green-400' :
                  tickerData.phase2.tier === 'DEGRADED' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-red-500/20 text-red-400'
                }`}>
                  {tickerData.phase2.tier}
                </span>
              </div>

              {expandedPhase === 2 && (
                <div className="mt-3 pt-3 border-t border-gray-700 text-xs">
                  <div className="grid grid-cols-5 gap-2 mb-2">
                    {Object.entries(tickerData.phase2.dimensions).map(([key, value]) => (
                      <div key={key} className="text-center">
                        <div className="text-gray-500 text-[10px]">{key.replace('_', ' ').slice(0, 8)}</div>
                        <div className={`font-mono ${value >= 75 ? 'text-green-400' : value >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                          {value}
                        </div>
                      </div>
                    ))}
                  </div>
                  {tickerData.phase2.reasons.length > 0 && (
                    <div className="text-gray-400 text-[10px]">
                      Issues: {tickerData.phase2.reasons.join(', ')}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Phase 3: Density Control */}
            <div
              className={`border rounded-lg p-3 cursor-pointer transition-all ${getPhaseColor(3)} ${
                expandedPhase === 3 ? 'bg-gray-800/50' : 'hover:bg-gray-800/30'
              }`}
              onClick={() => setExpandedPhase(expandedPhase === 3 ? null : 3)}
            >
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs font-mono">P3</span>
                  <span className="text-white font-medium">DENSITY</span>
                  <span className="text-gray-400 text-sm">{tickerData.phase3.density_score}%</span>
                </div>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  tickerData.phase3.throttle === 'OPEN' ? 'bg-green-500/20 text-green-400' :
                  tickerData.phase3.throttle === 'LIMITED' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-red-500/20 text-red-400'
                }`}>
                  {tickerData.phase3.throttle}
                </span>
              </div>

              {expandedPhase === 3 && (
                <div className="mt-3 pt-3 border-t border-gray-700 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Allowed Signals</span>
                    <span className="text-white">{tickerData.phase3.allowed_signals}</span>
                  </div>
                  {tickerData.phase3.reasons.length > 0 && (
                    <div className="text-gray-400 text-[10px] mt-1">
                      {tickerData.phase3.reasons.join(', ')}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Phase 4: Execution Permission */}
            <div
              className={`border rounded-lg p-3 cursor-pointer transition-all ${getPhaseColor(4)} ${
                expandedPhase === 4 ? 'bg-gray-800/50' : 'hover:bg-gray-800/30'
              }`}
              onClick={() => setExpandedPhase(expandedPhase === 4 ? null : 4)}
            >
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs font-mono">P4</span>
                  <span className="text-white font-medium">EXECUTION</span>
                  {tickerData.phase4.bias !== 'NEUTRAL' && (
                    <span className={`text-sm font-bold ${
                      tickerData.phase4.bias === 'LONG' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {tickerData.phase4.bias}
                    </span>
                  )}
                </div>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  tickerData.phase4.allowed ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                }`}>
                  {tickerData.phase4.allowed ? 'ALLOWED' : 'DENIED'}
                </span>
              </div>

              {expandedPhase === 4 && (
                <div className="mt-3 pt-3 border-t border-gray-700 grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <div className="text-gray-500">Mode</div>
                    <div className="text-white">{tickerData.phase4.execution_mode.replace('_', ' ')}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Risk State</div>
                    <div className={`${
                      tickerData.phase4.risk_state === 'NORMAL' ? 'text-green-400' :
                      tickerData.phase4.risk_state === 'REDUCED' ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {tickerData.phase4.risk_state}
                    </div>
                  </div>
                  {tickerData.phase4.invalidation_context.length > 0 && (
                    <div className="col-span-2">
                      <div className="text-gray-500">Invalidation</div>
                      <div className="text-gray-400 text-[10px]">
                        {tickerData.phase4.invalidation_context.slice(0, 3).join(' | ')}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {tickerData?.error && (
        <div className="p-4 text-red-400 text-sm">{tickerData.error}</div>
      )}
    </div>
  )
}
