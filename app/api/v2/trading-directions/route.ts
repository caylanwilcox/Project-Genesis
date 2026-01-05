/**
 * Trading Directions API
 * GET /api/v2/trading-directions
 *
 * Proxies to the Python ML server `/trading_directions` endpoint.
 * Returns EV-optimized trading signals using V6 time-split model.
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const ML_SERVER_URL =
  process.env.ML_SERVER_URL ||
  process.env.NEXT_PUBLIC_ML_SERVER_URL ||
  'https://genesis-production-c1e9.up.railway.app'

export async function GET() {
  try {
    const response = await fetch(`${ML_SERVER_URL}/trading_directions`, {
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
