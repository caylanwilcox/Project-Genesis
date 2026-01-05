/**
 * Northstar Phase Pipeline API
 * GET /api/v2/northstar
 *
 * Proxies to the Python ML server `/northstar` endpoint.
 * Returns 4-phase market structure analysis.
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const ML_SERVER_URL =
  process.env.ML_SERVER_URL ||
  process.env.NEXT_PUBLIC_ML_SERVER_URL ||
  'https://genesis-production-c1e9.up.railway.app'

export async function GET(request: Request) {
  try {
    // Forward any query params (e.g., ?ticker=SPY)
    const { searchParams } = new URL(request.url)
    const ticker = searchParams.get('ticker')

    const url = new URL(`${ML_SERVER_URL}/northstar`)
    if (ticker) {
      url.searchParams.set('ticker', ticker)
    }

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
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json(
      {
        error: 'ML server unavailable',
        message:
          'Start ML server with: cd ml && python3 predict_server.py',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 503 }
    )
  }
}
