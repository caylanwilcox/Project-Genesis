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
          v6: {
            target_a_prob: td.intraday.probability_a || 0.5,
            target_b_prob: td.intraday.probability_b || 0.5,
            session: td.intraday.session || 'early',
            signal: td.intraday.action === 'LONG' ? 'BULLISH' :
                    td.intraday.action === 'SHORT' ? 'BEARISH' : 'NEUTRAL',
            confidence: td.intraday.confidence || 0
          }
        } : null

        // Transform swing data to expected format
        const swing = td.swing ? {
          v6_swing: {
            prob_1d_up: td.swing['1d']?.probability || 0.5,
            prob_3d_up: td.swing['3d']?.probability || 0.5,
            prob_5d_up: td.swing['5d']?.probability || 0.5,
            prob_10d_up: td.swing['10d']?.probability || 0.5,
            signal_1d: td.swing['1d']?.signal || 'NEUTRAL',
            signal_3d: td.swing['3d']?.signal || 'NEUTRAL',
            signal_5d: td.swing['5d']?.signal || 'NEUTRAL',
            signal_10d: td.swing['10d']?.signal || 'NEUTRAL'
          }
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
