'use client'

import { useState, useEffect } from 'react'

// ========================
// TOOLTIP DEFINITIONS - Beginner-friendly explanations
// ========================

const TOOLTIPS = {
  // Phase names
  phase1: "TRUTH: What is the market actually doing right now? This phase reads the raw price action to determine if buyers or sellers are in control.",
  phase2: "HEALTH GATE: Is this a good quality setup? Checks if the current market conditions are stable enough to trade.",
  phase3: "DENSITY: How crowded is this trade? Controls how many signals we take to avoid overtrading.",
  phase4: "EXECUTION: Should you trade? The final yes/no decision with specific entry guidance.",

  // Phase 1 signals
  direction_UP: "BULLISH: Buyers are in control. Price is making higher highs and higher lows.",
  direction_DOWN: "BEARISH: Sellers are in control. Price is making lower highs and lower lows.",
  direction_BALANCED: "CHOPPY: Neither side has control. Price is moving sideways - best to wait.",

  confidence_STRUCTURAL_EDGE: "STRONG EDGE: Clear market structure with high probability setup. Good conditions to trade.",
  confidence_CONTEXT_ONLY: "WEAK EDGE: Some direction visible but not ideal. Trade with caution or reduced size.",
  confidence_NO_TRADE: "NO EDGE: Market is unclear or too risky. Stay on the sidelines.",

  acceptance: "ACCEPTANCE: Has price 'accepted' the current level? If buyers keep price above a level, they've accepted it as support.",
  acceptance_STRONG: "STRONG: Price is firmly holding this level with conviction. Good sign.",
  acceptance_MODERATE: "MODERATE: Price is holding but with some back-and-forth. Decent sign.",
  acceptance_WEAK: "WEAK: Price is barely holding. Could break at any moment.",

  range_TREND: "TRENDING: Price is moving in a clear direction. Great for momentum trades.",
  range_BALANCE: "RANGING: Price is stuck between two levels. Wait for a breakout or trade the range.",
  range_FAILED_EXPANSION: "FAILED MOVE: Price tried to break out but got rejected. Often signals reversal.",

  mtf: "MULTI-TIMEFRAME: Are different timeframes (5min, 15min, 1hr) agreeing on direction?",
  mtf_aligned: "ALIGNED: All timeframes agree - stronger signal.",
  mtf_conflict: "CONFLICT: Timeframes disagree - be careful, signal is weaker.",

  conviction_HIGH: "HIGH CONVICTION: Strong volume confirming the move. Institutions are likely involved.",
  conviction_MEDIUM: "MEDIUM CONVICTION: Decent volume but not exceptional. Proceed normally.",
  conviction_LOW: "LOW CONVICTION: Weak volume. The move might not have staying power.",

  effort_result: "EFFORT vs RESULT: Is the price movement matching the volume? Big volume should mean big moves.",

  failure: "FAILURE PATTERNS: Has price failed to do what it 'should' have done? Failed breakouts often reverse.",

  // Phase 2 signals
  health_HEALTHY: "HEALTHY: All conditions look good. Green light for trading.",
  health_DEGRADED: "DEGRADED: Some concerns but tradeable. Consider smaller position size.",
  health_UNSTABLE: "UNSTABLE: Too many warning signs. Best to wait for conditions to improve.",

  structural_integrity: "STRUCTURE: How clean is the price pattern? Higher = cleaner chart.",
  time_persistence: "TIME: Has the signal lasted long enough to be meaningful?",
  volatility_alignment: "VOLATILITY: Is current volatility normal? Extreme volatility can be risky.",
  participation_consistency: "PARTICIPATION: Is volume consistent with the move?",
  failure_risk: "FAILURE RISK: How likely is this setup to fail? Lower = safer.",

  // Phase 3 signals
  throttle_OPEN: "OPEN: Full trading allowed. Take signals as they come.",
  throttle_LIMITED: "LIMITED: Restrict to best setups only. Be selective.",
  throttle_BLOCKED: "BLOCKED: No new trades. Wait for reset.",

  density: "SIGNAL DENSITY: How many good setups have we seen recently? Too many = be more selective.",

  // Phase 4 signals
  allowed: "TRADE ALLOWED: All checks passed. You have permission to execute.",
  denied: "TRADE DENIED: One or more checks failed. Do not trade this setup.",

  bias_LONG: "LONG BIAS: Look for buying opportunities (calls). Price expected to go up.",
  bias_SHORT: "SHORT BIAS: Look for selling opportunities (puts). Price expected to go down.",
  bias_NEUTRAL: "NEUTRAL: No clear direction. Wait for clarity.",

  mode_TREND_CONTINUATION: "TREND FOLLOWING: Trade in the direction of the trend. Buy dips in uptrends.",
  mode_MEAN_REVERSION: "MEAN REVERSION: Price has moved too far, expect a pullback. Counter-trend trade.",
  mode_SCALP: "QUICK SCALP: Fast in-and-out trade. Take small profits quickly.",
  mode_NO_TRADE: "NO TRADE: Conditions don't support any strategy right now.",

  risk_NORMAL: "NORMAL RISK: Standard position size. Conditions are good.",
  risk_REDUCED: "REDUCED RISK: Use half your normal size. Some uncertainty present.",
  risk_DEFENSIVE: "DEFENSIVE: Use quarter size or skip. High uncertainty.",

  // Key Levels
  pivot: "PIVOT: The central price level. Price above = bullish, below = bearish.",
  pivot_r1: "RESISTANCE 1 (R1): First major ceiling. Price often struggles here.",
  pivot_s1: "SUPPORT 1 (S1): First major floor. Price often bounces here.",
  recent_high: "RECENT HIGH: The highest price in the last few days. Breaking above = bullish.",
  recent_low: "RECENT LOW: The lowest price in the last few days. Breaking below = bearish.",
}

// Tooltip component
function Tooltip({ text, children }: { text: string; children: React.ReactNode }) {
  return (
    <span className="relative group cursor-help">
      {children}
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-800 border border-gray-600 text-gray-200 text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-normal w-64 z-50 shadow-lg">
        {text}
        <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-800"></span>
      </span>
    </span>
  )
}

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
  current_time_et?: string
  current_hour?: number
  market_open?: boolean
  tickers: Record<string, TickerAnalysis | any>
  pipeline_version?: string
  session?: string
  analysis_type?: string
  summary?: {
    execution_allowed?: boolean
    allowed_tickers?: Array<{
      ticker: string
      bias: string
      mode: string
      risk: string
    }>
    recommendation?: string
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
          // Transform MTF response format to expected Northstar format
          // MTF returns: { tickers: { SPY: { northstar: {...}, swing: {...} } } }
          // We need: { tickers: { SPY: { phase1, phase2, phase3, phase4, ... } } }
          const transformed: NorthstarData = {
            generated_at: result.generated_at || new Date().toISOString(),
            current_time_et: result.current_time_et || new Date().toLocaleTimeString('en-US', { timeZone: 'America/New_York' }),
            tickers: {},
            pipeline_version: result.pipeline_version || '2.0',
            session: result.session,
          }

          // Process each ticker
          if (result.tickers) {
            for (const [symbol, tickerData] of Object.entries(result.tickers)) {
              const td = tickerData as any
              // If it has northstar property (from MTF), extract it
              if (td.northstar) {
                transformed.tickers[symbol] = {
                  ...td.northstar,
                  symbol,
                  current_price: td.current_price || td.northstar?.phase1?.key_levels?.current_price || 0,
                  today_change_pct: 0,
                  bars_analyzed: 0,
                }
              } else if (td.phase1) {
                // Already in correct format
                transformed.tickers[symbol] = td
              }
            }
          }

          // Build summary from ticker data
          const allowedTickers: Array<{ ticker: string; bias: string; mode: string; risk: string }> = []
          let anyAllowed = false
          for (const [sym, td] of Object.entries(transformed.tickers)) {
            const tickerInfo = td as any
            if (tickerInfo?.phase4?.allowed) {
              anyAllowed = true
              allowedTickers.push({
                ticker: sym,
                bias: tickerInfo.phase4.bias || 'NEUTRAL',
                mode: tickerInfo.phase4.execution_mode || 'STANDARD',
                risk: tickerInfo.phase4.risk_state || 'NORMAL',
              })
            }
          }

          transformed.summary = {
            execution_allowed: anyAllowed,
            allowed_tickers: allowedTickers,
            recommendation: anyAllowed
              ? `Trade signals available for ${allowedTickers.map(t => t.ticker).join(', ')}`
              : 'No clear setups - wait for better conditions',
          }

          setData(transformed)
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
    const isProduction = typeof window !== 'undefined' && !window.location.hostname.includes('localhost')
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="text-yellow-400 text-sm flex items-center gap-2">
          <span>⚠</span>
          <span>ML Server Not Connected</span>
        </div>
        <div className="text-gray-500 text-xs mt-2">
          {isProduction ? (
            <>
              <p>The ML server runs locally and isn't available in production.</p>
              <p className="mt-1">To see live signals, run the app locally with:</p>
              <code className="block mt-1 bg-gray-800 px-2 py-1 rounded text-gray-400">
                cd ml && python -m server.app
              </code>
            </>
          ) : (
            <>
              <p>Start ML server: <code className="bg-gray-800 px-1 rounded">cd ml && python -m server.app</code></p>
            </>
          )}
        </div>
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
              {data.pipeline_version && (
                <span className="px-2 py-0.5 rounded text-xs bg-purple-500/20 text-purple-400">
                  v{data.pipeline_version}
                </span>
              )}
            </h3>
            <p className="text-gray-400 text-xs mt-0.5">{data.current_time_et || data.session || 'Live'}</p>
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
      {data.summary && (
        <div className={`px-4 py-2 border-b border-gray-800 ${
          data.summary.execution_allowed ? 'bg-green-900/20' : 'bg-red-900/20'
        }`}>
          <div className="flex items-center justify-between">
            <span className={`text-sm font-medium ${
              data.summary.execution_allowed ? 'text-green-400' : 'text-red-400'
            }`}>
              {data.summary.execution_allowed ? 'EXECUTION ALLOWED' : 'STAND DOWN'}
            </span>
            <span className="text-gray-400 text-sm">{data.summary.recommendation || ''}</span>
          </div>
        </div>
      )}

      {tickerData && !tickerData.error && tickerData.phase1 && (
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
                  <Tooltip text={TOOLTIPS.phase1}>
                    <span className="text-white font-medium border-b border-dashed border-gray-600">TRUTH</span>
                  </Tooltip>
                  <Tooltip text={TOOLTIPS[`direction_${tickerData.phase1.direction}` as keyof typeof TOOLTIPS] || ''}>
                    <span className={`text-sm ${getDirectionColor(tickerData.phase1.direction)} border-b border-dashed border-gray-600`}>
                      {getDirectionIcon(tickerData.phase1.direction)} {tickerData.phase1.direction}
                    </span>
                  </Tooltip>
                </div>
                <Tooltip text={TOOLTIPS[`confidence_${tickerData.phase1.confidence_band}` as keyof typeof TOOLTIPS] || ''}>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium border-b border-dashed border-transparent hover:border-gray-500 ${
                    tickerData.phase1.confidence_band === 'STRUCTURAL_EDGE' ? 'bg-green-500/20 text-green-400' :
                    tickerData.phase1.confidence_band === 'CONTEXT_ONLY' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {tickerData.phase1.confidence_band.replace('_', ' ')}
                  </span>
                </Tooltip>
              </div>

              {expandedPhase === 1 && (
                <div className="mt-3 pt-3 border-t border-gray-700 space-y-2 text-xs">
                  {/* Acceptance Details */}
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${tickerData.phase1.acceptance.accepted ? 'bg-green-500' : 'bg-red-500'}`} />
                    <Tooltip text={TOOLTIPS.acceptance}>
                      <span className="text-gray-400 border-b border-dashed border-gray-600">Acceptance:</span>
                    </Tooltip>
                    <Tooltip text={TOOLTIPS[`acceptance_${tickerData.phase1.acceptance.acceptance_strength}` as keyof typeof TOOLTIPS] || ''}>
                      <span className={tickerData.phase1.acceptance.accepted ? 'text-green-400' : 'text-red-400'}>
                        {tickerData.phase1.acceptance.accepted
                          ? `YES - ${tickerData.phase1.acceptance.acceptance_strength}`
                          : 'NO'}
                      </span>
                    </Tooltip>
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
                    <Tooltip text={TOOLTIPS[`range_${tickerData.phase1.range.state}` as keyof typeof TOOLTIPS] || ''}>
                      <span className={`border-b border-dashed border-gray-600 ${
                        tickerData.phase1.range.state === 'TREND' ? 'text-green-400' :
                        tickerData.phase1.range.state === 'BALANCE' ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {tickerData.phase1.range.state}
                        {tickerData.phase1.range.expansion_quality !== 'NONE' && ` (${tickerData.phase1.range.expansion_quality})`}
                        {tickerData.phase1.range.rotation_complete && ' - Rotation complete'}
                      </span>
                    </Tooltip>
                  </div>

                  {/* MTF Alignment */}
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${tickerData.phase1.mtf.aligned ? 'bg-green-500' : 'bg-red-500'}`} />
                    <Tooltip text={TOOLTIPS.mtf}>
                      <span className="text-gray-400 border-b border-dashed border-gray-600">MTF:</span>
                    </Tooltip>
                    <Tooltip text={tickerData.phase1.mtf.aligned ? TOOLTIPS.mtf_aligned : TOOLTIPS.mtf_conflict}>
                      <span className={`border-b border-dashed border-gray-600 ${tickerData.phase1.mtf.aligned ? 'text-green-400' : 'text-red-400'}`}>
                        {tickerData.phase1.mtf.aligned
                          ? `ALIGNED - ${tickerData.phase1.mtf.dominant_tf} dominant`
                          : `CONFLICT - ${tickerData.phase1.mtf.conflict_tf || 'Timeframes disagree'}`}
                      </span>
                    </Tooltip>
                  </div>

                  {/* Participation */}
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${
                      tickerData.phase1.participation.conviction === 'HIGH' ? 'bg-green-500' :
                      tickerData.phase1.participation.conviction === 'MEDIUM' ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                    <span className="text-gray-400">Conviction:</span>
                    <Tooltip text={TOOLTIPS[`conviction_${tickerData.phase1.participation.conviction}` as keyof typeof TOOLTIPS] || ''}>
                      <span className={`border-b border-dashed border-gray-600 ${
                        tickerData.phase1.participation.conviction === 'HIGH' ? 'text-green-400' :
                        tickerData.phase1.participation.conviction === 'MEDIUM' ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {tickerData.phase1.participation.conviction}
                        {!tickerData.phase1.participation.effort_result_match && ' ⚠ Effort/result mismatch'}
                      </span>
                    </Tooltip>
                  </div>

                  {/* Failure Patterns */}
                  {tickerData.phase1.failure.present && (
                    <Tooltip text={TOOLTIPS.failure}>
                      <div className="text-red-400 flex items-center gap-2 cursor-help">
                        <span className="w-2 h-2 rounded-full bg-red-500" />
                        FAILURE: {tickerData.phase1.failure.failure_types.join(', ')}
                      </div>
                    </Tooltip>
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
                    <Tooltip text={TOOLTIPS.pivot_r1}>
                      <span><span className="text-red-400 border-b border-dashed border-red-400/50">R1:</span> <span className="text-white font-mono">${tickerData.phase1.key_levels.pivot_r1?.toFixed(2)}</span></span>
                    </Tooltip>
                    <Tooltip text={TOOLTIPS.pivot}>
                      <span><span className="text-yellow-400 border-b border-dashed border-yellow-400/50">P:</span> <span className="text-white font-mono">${tickerData.phase1.key_levels.pivot?.toFixed(2)}</span></span>
                    </Tooltip>
                    <Tooltip text={TOOLTIPS.pivot_s1}>
                      <span><span className="text-green-400 border-b border-dashed border-green-400/50">S1:</span> <span className="text-white font-mono">${tickerData.phase1.key_levels.pivot_s1?.toFixed(2)}</span></span>
                    </Tooltip>
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
                  <Tooltip text={TOOLTIPS.phase2}>
                    <span className="text-white font-medium border-b border-dashed border-gray-600">HEALTH GATE</span>
                  </Tooltip>
                  <span className="text-gray-400 text-sm">{tickerData.phase2.health_score}%</span>
                </div>
                <Tooltip text={TOOLTIPS[`health_${tickerData.phase2.tier}` as keyof typeof TOOLTIPS] || ''}>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium border-b border-dashed border-transparent hover:border-gray-500 ${
                    tickerData.phase2.tier === 'HEALTHY' ? 'bg-green-500/20 text-green-400' :
                    tickerData.phase2.tier === 'DEGRADED' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {tickerData.phase2.tier}
                  </span>
                </Tooltip>
              </div>

              {expandedPhase === 2 && (
                <div className="mt-3 pt-3 border-t border-gray-700 text-xs">
                  <div className="grid grid-cols-5 gap-2 mb-2">
                    {Object.entries(tickerData.phase2.dimensions).map(([key, value]) => (
                      <Tooltip key={key} text={TOOLTIPS[key as keyof typeof TOOLTIPS] || `${key.replace(/_/g, ' ')}: Score out of 100`}>
                        <div className="text-center cursor-help">
                          <div className="text-gray-500 text-[10px] border-b border-dashed border-gray-600">{key.replace('_', ' ').slice(0, 8)}</div>
                          <div className={`font-mono ${value >= 75 ? 'text-green-400' : value >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                            {value}
                          </div>
                        </div>
                      </Tooltip>
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
                  <Tooltip text={TOOLTIPS.phase3}>
                    <span className="text-white font-medium border-b border-dashed border-gray-600">DENSITY</span>
                  </Tooltip>
                  <Tooltip text={TOOLTIPS.density}>
                    <span className="text-gray-400 text-sm border-b border-dashed border-gray-600">{tickerData.phase3.density_score}%</span>
                  </Tooltip>
                </div>
                <Tooltip text={TOOLTIPS[`throttle_${tickerData.phase3.throttle}` as keyof typeof TOOLTIPS] || ''}>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium border-b border-dashed border-transparent hover:border-gray-500 ${
                    tickerData.phase3.throttle === 'OPEN' ? 'bg-green-500/20 text-green-400' :
                    tickerData.phase3.throttle === 'LIMITED' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {tickerData.phase3.throttle}
                  </span>
                </Tooltip>
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
                  <Tooltip text={TOOLTIPS.phase4}>
                    <span className="text-white font-medium border-b border-dashed border-gray-600">EXECUTION</span>
                  </Tooltip>
                  {tickerData.phase4.bias !== 'NEUTRAL' && (
                    <Tooltip text={TOOLTIPS[`bias_${tickerData.phase4.bias}` as keyof typeof TOOLTIPS] || ''}>
                      <span className={`text-sm font-bold border-b border-dashed border-gray-600 ${
                        tickerData.phase4.bias === 'LONG' ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {tickerData.phase4.bias}
                      </span>
                    </Tooltip>
                  )}
                </div>
                <Tooltip text={tickerData.phase4.allowed ? TOOLTIPS.allowed : TOOLTIPS.denied}>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium border-b border-dashed border-transparent hover:border-gray-500 ${
                    tickerData.phase4.allowed ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                  }`}>
                    {tickerData.phase4.allowed ? 'ALLOWED' : 'DENIED'}
                  </span>
                </Tooltip>
              </div>

              {expandedPhase === 4 && (
                <div className="mt-3 pt-3 border-t border-gray-700 grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <div className="text-gray-500">Mode</div>
                    <Tooltip text={TOOLTIPS[`mode_${tickerData.phase4.execution_mode}` as keyof typeof TOOLTIPS] || `${tickerData.phase4.execution_mode}: Trading mode based on current conditions`}>
                      <div className="text-white cursor-help border-b border-dashed border-gray-600 inline-block">{tickerData.phase4.execution_mode.replace('_', ' ')}</div>
                    </Tooltip>
                  </div>
                  <div>
                    <div className="text-gray-500">Risk State</div>
                    <Tooltip text={TOOLTIPS[`risk_${tickerData.phase4.risk_state}` as keyof typeof TOOLTIPS] || ''}>
                      <div className={`cursor-help border-b border-dashed border-gray-600 inline-block ${
                        tickerData.phase4.risk_state === 'NORMAL' ? 'text-green-400' :
                        tickerData.phase4.risk_state === 'REDUCED' ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {tickerData.phase4.risk_state}
                      </div>
                    </Tooltip>
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
