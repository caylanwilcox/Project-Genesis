/**
 * Northstar & Swing Data API
 * GET /api/v2/northstar
 *
 * Uses local ML server (localhost:5001) with full models + swing data
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const ML_SERVER_URL = process.env.ML_SERVER_URL || process.env.NEXT_PUBLIC_ML_SERVER_URL || 'https://project-genesis-6roa.onrender.com'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const ticker = searchParams.get('ticker') || 'SPY'

    const mtfUrl = new URL(`${ML_SERVER_URL}/mtf`)
    mtfUrl.searchParams.set('tickers', ticker)

    const response = await fetch(mtfUrl.toString(), { cache: 'no-store' })

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

    // Transform MTF response to consistent format
    const transformed: Record<string, any> = {
      tickers: {},
      analysis_type: data.analysis_type,
      generated_at: data.generated_at,
      session: data.session,
      pipeline_version: data.pipeline_version,
      summary: data.summary,
    }

    if (data.tickers) {
      // MTF format: { tickers: { SPY: { intraday: {...}, swing: {...} } } }
      for (const [sym, tickerData] of Object.entries(data.tickers)) {
        const td = tickerData as any
        transformed.tickers[sym] = {
          northstar: td.intraday,
          swing: td.swing,
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
        message: 'ML server connection failed',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 503 }
    )
  }
}
