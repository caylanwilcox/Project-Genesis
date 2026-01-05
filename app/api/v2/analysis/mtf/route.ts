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
  'http://localhost:5001'

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
    return NextResponse.json(data)
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
