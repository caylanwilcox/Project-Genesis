/**
 * Northstar & Swing Data API
 * GET /api/v2/northstar
 *
 * Priority order:
 * 1. Local server (localhost:5001) - has full models + swing data
 * 2. Railway server (fallback) - may have limited models
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const LOCAL_SERVER_URL = 'http://localhost:5001'
const RAILWAY_SERVER_URL =
  process.env.ML_SERVER_URL ||
  process.env.NEXT_PUBLIC_ML_SERVER_URL ||
  'https://genesis-production-c1e9.up.railway.app'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const ticker = searchParams.get('ticker') || 'SPY'

    let data: any = null
    let usedMtf = false
    let sourceServer = ''

    // Try LOCAL server first (has all models + swing data)
    try {
      const mtfUrl = new URL(`${LOCAL_SERVER_URL}/mtf`)
      mtfUrl.searchParams.set('tickers', ticker)
      const mtfResponse = await fetch(mtfUrl.toString(), {
        cache: 'no-store',
        signal: AbortSignal.timeout(5000) // 5s timeout for local
      })

      if (mtfResponse.ok) {
        data = await mtfResponse.json()
        usedMtf = true
        sourceServer = 'local'
      }
    } catch {
      // Local server not available, will try Railway
    }

    // Fall back to RAILWAY server
    if (!data) {
      try {
        // Try MTF on Railway first
        const mtfUrl = new URL(`${RAILWAY_SERVER_URL}/mtf`)
        mtfUrl.searchParams.set('tickers', ticker)
        const mtfResponse = await fetch(mtfUrl.toString(), { cache: 'no-store' })

        if (mtfResponse.ok) {
          data = await mtfResponse.json()
          usedMtf = true
          sourceServer = 'railway-mtf'
        }
      } catch {
        // MTF not available on Railway
      }
    }

    // Final fallback: /northstar on Railway (intraday only, no swing)
    if (!data) {
      const northstarUrl = new URL(`${RAILWAY_SERVER_URL}/northstar`)
      northstarUrl.searchParams.set('ticker', ticker)

      const response = await fetch(northstarUrl.toString(), { cache: 'no-store' })

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

      data = await response.json()
      sourceServer = 'railway-northstar'
    }

    // Transform response to consistent format
    const transformed: Record<string, any> = {
      tickers: {},
      analysis_type: data.analysis_type,
      generated_at: data.generated_at,
      session: data.session,
      pipeline_version: data.pipeline_version,
      summary: data.summary,
    }

    if (usedMtf && data.tickers) {
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
    } else if (data.tickers) {
      // Northstar format: { tickers: { SPY: { phase1, phase2, ... } } }
      for (const [sym, tickerData] of Object.entries(data.tickers)) {
        const td = tickerData as any
        transformed.tickers[sym] = {
          northstar: td, // Already in correct format
          swing: null,   // No swing data from /northstar endpoint
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
