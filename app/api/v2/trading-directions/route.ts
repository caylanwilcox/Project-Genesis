/**
 * Trading Directions API
 * GET /api/v2/trading-directions
 *
 * Priority order:
 * 1. Local server (localhost:5001) - has full models
 * 2. Railway server (fallback) - may have limited models
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const LOCAL_SERVER_URL = 'http://localhost:5001'
const RAILWAY_SERVER_URL =
  process.env.ML_SERVER_URL ||
  process.env.NEXT_PUBLIC_ML_SERVER_URL ||
  'https://project-genesis-6roa.onrender.com'

export async function GET() {
  try {
    // Try LOCAL server first (has all models)
    try {
      const response = await fetch(`${LOCAL_SERVER_URL}/trading_directions`, {
        cache: 'no-store',
        signal: AbortSignal.timeout(5000) // 5s timeout for local
      })

      if (response.ok) {
        const data = await response.json()
        return NextResponse.json({ ...data, source: 'local' })
      }
    } catch {
      // Local server not available, will try Railway
    }

    // Fall back to RAILWAY server
    const response = await fetch(`${RAILWAY_SERVER_URL}/trading_directions`, {
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
    return NextResponse.json({ ...data, source: 'railway' })
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
