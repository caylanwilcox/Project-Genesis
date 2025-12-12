/**
 * ML Daily Signals Proxy
 * GET /api/v2/ml/daily-signals
 *
 * Proxies to the Python ML server `/daily_signals` endpoint.
 * This avoids client-side CORS/env issues by calling from the server.
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const ML_SERVER_URL =
  process.env.ML_SERVER_URL ||
  process.env.NEXT_PUBLIC_ML_SERVER_URL ||
  'http://localhost:5001'

export async function GET() {
  try {
    const response = await fetch(`${ML_SERVER_URL}/daily_signals`, {
      // Ensure we don't cache ML signals server-side
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
          'Start ML server with: cd ml && python3 predict_server.py (default http://localhost:5001)',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 503 }
    )
  }
}

