/**
 * Replay Mode API
 * GET /api/v2/replay?date=2025-12-20&time=14:30
 *
 * Time-travel testing - replay any historical day at any point in time.
 * Returns V6 signals + Northstar analysis as they would have appeared.
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const ML_SERVER_URL =
  process.env.ML_SERVER_URL ||
  process.env.NEXT_PUBLIC_ML_SERVER_URL ||
  'https://project-genesis-6roa.onrender.com'

// Determine action from probabilities using 25-75 neutral zone
function getActionFromProb(probA: number, probB: number | null, session: string): string {
  const prob = session === 'late' && probB !== null ? probB : probA
  if (prob >= 0.75) return 'LONG'
  if (prob <= 0.25) return 'SHORT'
  return 'NO_TRADE'
}

// Get reason string
function getReason(prob: number, session: string): string {
  if (prob >= 0.75) return `Strong bullish signal (${Math.round(prob * 100)}%)`
  if (prob <= 0.25) return `Strong bearish signal (${Math.round(prob * 100)}%)`
  return `Neutral zone (${Math.round(prob * 100)}%) - no edge`
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const date = searchParams.get('date')
    const time = searchParams.get('time')
    const ticker = searchParams.get('ticker')

    if (!date || !time) {
      return NextResponse.json(
        { error: 'Missing required parameters: date and time' },
        { status: 400 }
      )
    }

    // Parse time (HH:MM) to extract hour for ML server
    const [hourStr, minuteStr] = time.split(':')
    const hour = parseInt(hourStr, 10)
    const minute = parseInt(minuteStr || '0', 10)
    const session = hour < 11 ? 'early' : 'late'

    const url = new URL(`${ML_SERVER_URL}/replay`)
    url.searchParams.set('date', date)
    url.searchParams.set('hour', hour.toString())
    if (ticker) url.searchParams.set('tickers', ticker)

    const response = await fetch(url.toString(), {
      cache: 'no-store',
    })

    if (!response.ok) {
      const text = await response.text()
      return NextResponse.json(
        { error: 'ML server returned error', status: response.status, details: text },
        { status: 502 }
      )
    }

    const mlData = await response.json()

    // Transform ML server response to match frontend expectations
    const tickers: Record<string, any> = {}
    const v6Signals: Record<string, any> = {}
    const northstar: Record<string, any> = {}
    const actionableTickers: string[] = []
    const v6Actionable: string[] = []
    let bestTicker: string | null = null
    let bestProb = 0

    for (const [tickerName, tickerData] of Object.entries(mlData.tickers || {})) {
      const data = tickerData as any

      if (data.error) {
        tickers[tickerName] = {
          error: data.error,
          current_price: 0,
          today_open: 0,
          today_change_pct: 0,
          price_11am: null,
          bars_analyzed: 0,
          v6: {
            action: 'NO_TRADE',
            reason: data.error,
            probability_a: null,
            probability_b: null,
            session: session,
            price_11am: null
          },
          northstar: createDefaultNorthstar()
        }
        continue
      }

      const probA = data.target_a_prob || 0.5
      const probB = data.target_b_prob || null
      const action = getActionFromProb(probA, probB, session)
      const activeProb = session === 'late' && probB !== null ? probB : probA

      // Track best ticker
      const confidence = Math.abs(activeProb - 0.5)
      if (action !== 'NO_TRADE' && confidence > bestProb) {
        bestProb = confidence
        bestTicker = tickerName
      }

      if (action !== 'NO_TRADE') {
        v6Actionable.push(tickerName)
        actionableTickers.push(tickerName)
      }

      const v6Signal = {
        action,
        reason: getReason(activeProb, session),
        probability_a: probA,
        probability_b: probB,
        session: session,
        price_11am: data.price_11am
      }

      v6Signals[tickerName] = v6Signal

      // Create placeholder Northstar data (simplified since replay doesn't have full analysis)
      const nsData = createDefaultNorthstar()
      nsData.phase1.direction = action === 'LONG' ? 'UP' : action === 'SHORT' ? 'DOWN' : 'BALANCED'
      nsData.phase4.allowed = action !== 'NO_TRADE'
      nsData.phase4.bias = action === 'LONG' ? 'LONG' : action === 'SHORT' ? 'SHORT' : 'NEUTRAL'
      northstar[tickerName] = nsData

      tickers[tickerName] = {
        current_price: data.today_open || 0, // Use open as approximation
        today_open: data.today_open || 0,
        today_change_pct: 0,
        price_11am: data.price_11am,
        bars_analyzed: data.bars_analyzed || 0,
        v6: v6Signal,
        northstar: nsData
      }
    }

    // Build summary
    const recommendation = v6Actionable.length > 0
      ? `${v6Actionable.length} actionable signal${v6Actionable.length > 1 ? 's' : ''}: ${v6Actionable.join(', ')}`
      : 'No actionable signals - market in neutral zone'

    const result = {
      mode: 'replay',
      replay_date: date,
      replay_time: time,
      simulated_time_et: `${hour}:${minute.toString().padStart(2, '0')} ET`,
      simulated_hour: hour,
      simulated_minute: minute,
      market_open: hour >= 9 && hour < 16,
      session: session,
      tickers,
      v6_signals: v6Signals,
      northstar,
      summary: {
        best_ticker: bestTicker,
        allowed_tickers: actionableTickers,
        v6_actionable: v6Actionable,
        recommendation
      }
    }

    return NextResponse.json(result)
  } catch (error) {
    return NextResponse.json(
      {
        error: 'ML server unavailable',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 503 }
    )
  }
}

// Create default Northstar structure for replay mode
function createDefaultNorthstar() {
  return {
    phase1: {
      direction: 'BALANCED',
      confidence_band: 'CONTEXT_ONLY',
      dominant_timeframe: 'INTRADAY',
      acceptance: {
        accepted: false,
        acceptance_strength: 'WEAK',
        acceptance_reason: 'Replay mode - limited data',
        failed_levels: []
      },
      range: {
        state: 'BALANCE',
        rotation_complete: false,
        expansion_quality: 'NONE'
      },
      mtf: {
        aligned: true,
        dominant_tf: 'INTRADAY',
        conflict_tf: null
      },
      participation: {
        conviction: 'MEDIUM',
        effort_result_match: true
      },
      failure: {
        present: false,
        failure_types: []
      },
      key_levels: {
        recent_high: 0,
        recent_low: 0,
        mid_point: 0,
        pivot: 0,
        pivot_r1: 0,
        pivot_s1: 0,
        current_price: 0,
        today_open: 0,
        prev_close: 0,
        retest_high: 0,
        retest_low: 0
      },
      volatility_expansion: {
        probability: 0.5,
        signal: 'NEUTRAL',
        expansion_likely: false,
        reasons: []
      }
    },
    phase2: {
      health_score: 50,
      tier: 'DEGRADED',
      stand_down: false,
      reasons: ['Replay mode - using V6 signal only'],
      dimensions: {
        structural_integrity: 50,
        time_persistence: 50,
        volatility_alignment: 50,
        participation_consistency: 50,
        failure_risk: 50
      }
    },
    phase3: {
      throttle: 'OPEN',
      density_score: 50,
      allowed_signals: 999,
      reasons: []
    },
    phase4: {
      allowed: false,
      bias: 'NEUTRAL',
      execution_mode: 'NO_TRADE',
      risk_state: 'NORMAL',
      invalidation_context: []
    }
  }
}
