/**
 * Northstar & Swing Data API
 * GET /api/v2/northstar
 *
 * Proxies to the Python ML server `/mtf` endpoint.
 * Returns both intraday (northstar) and swing analysis with AI recommendation data.
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const ML_SERVER_URL =
  process.env.ML_SERVER_URL ||
  process.env.NEXT_PUBLIC_ML_SERVER_URL ||
  'http://127.0.0.1:5001'

export async function GET(request: Request) {
  try {
    // Forward any query params (e.g., ?ticker=SPY)
    const { searchParams } = new URL(request.url)
    const ticker = searchParams.get('ticker') || 'SPY'

    // Use MTF endpoint which returns both intraday and swing data
    const url = new URL(`${ML_SERVER_URL}/mtf`)
    url.searchParams.set('tickers', ticker)

    const response = await fetch(url.toString(), {
      cache: 'no-store',
    })

    if (!response.ok) {
      const text = await response.text()
      return NextResponse.json(
        {
          error: 'ML server returned error',
          status: response.status,
          details: text,
        },
        { status: 502 }
      )
    }

    const data = await response.json()

    // Transform MTF response to match expected format
    // MTF returns: { tickers: { SPY: { intraday: {...}, swing: {...} } } }
    // We map intraday -> northstar for compatibility
    const transformed: Record<string, any> = {
      tickers: {},
      analysis_type: data.analysis_type,
      generated_at: data.generated_at,
      session: data.session,
    }

    if (data.tickers) {
      for (const [sym, tickerData] of Object.entries(data.tickers)) {
        const td = tickerData as any
        transformed.tickers[sym] = {
          northstar: td.intraday, // Map intraday phases to northstar
          swing: td.swing,        // Include swing data
          alignment: td.alignment,
          current_price: td.current_price,
        }
      }
    }

    return NextResponse.json(transformed)
  } catch (error) {
    return NextResponse.json(
      {
        error: 'ML server unavailable',
        message:
          'Start ML server with: cd ml && python -m server.app',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 503 }
    )
  }
}
