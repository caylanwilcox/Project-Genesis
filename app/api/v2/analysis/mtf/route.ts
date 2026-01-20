/**
 * Multi-Timeframe Analysis API
 * GET /api/v2/analysis/mtf?tickers=SPY,QQQ,IWM
 *
 * Combines INTRADAY (V6 + Northstar) and SWING (V6 SWING + RPE SWING)
 * for comprehensive market view across timeframes.
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const ML_SERVER_URL =
  process.env.ML_SERVER_URL ||
  process.env.NEXT_PUBLIC_ML_SERVER_URL ||
  'https://project-genesis-6roa.onrender.com'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const tickers = searchParams.get('tickers')

    const url = new URL(`${ML_SERVER_URL}/mtf`)
    if (tickers) url.searchParams.set('tickers', tickers)

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

    const data = await response.json()

    // Transform ML server response to match frontend expectations
    const transformed: Record<string, any> = {
      ...data,
      tickers: {}
    }

    if (data.tickers) {
      for (const [ticker, tickerData] of Object.entries(data.tickers)) {
        const td = tickerData as any

        // Transform intraday data to expected format
        const intraday = td.intraday ? {
          v6: td.intraday.v6 ? {
            target_a_prob: td.intraday.v6.target_a_prob ?? 0.5,
            target_b_prob: td.intraday.v6.target_b_prob ?? 0.5,
            session: td.intraday.v6.session || 'early',
            signal: td.intraday.v6.signal || 'NEUTRAL'
          } : null,
          phase4: td.intraday.phase4 || null
        } : null

        // Transform swing data to expected format
        const swing = td.swing ? {
          v6_swing: td.swing.v6_swing ? {
            prob_1d_up: td.swing.v6_swing.prob_1d_up ?? 0.5,
            prob_3d_up: td.swing.v6_swing.prob_3d_up ?? 0.5,
            prob_5d_up: td.swing.v6_swing.prob_5d_up ?? 0.5,
            prob_10d_up: td.swing.v6_swing.prob_10d_up ?? 0.5,
            signal_1d: td.swing.v6_swing.signal_1d || 'NEUTRAL',
            signal_3d: td.swing.v6_swing.signal_3d || 'NEUTRAL',
            signal_5d: td.swing.v6_swing.signal_5d || 'NEUTRAL',
            signal_10d: td.swing.v6_swing.signal_10d || 'NEUTRAL'
          } : null,
          phase4: td.swing.phase4 || null
        } : null

        transformed.tickers[ticker] = {
          current_price: td.current_price || 0,
          intraday,
          swing,
          alignment: td.alignment ? {
            status: td.alignment === 'bullish' ? 'ALIGNED' :
                    td.alignment === 'bearish' ? 'ALIGNED' : 'PARTIAL',
            direction: td.alignment,
            confidence: 'MEDIUM'
          } : null,
          error: td.error
        }
      }
    }

    return NextResponse.json(transformed)
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
