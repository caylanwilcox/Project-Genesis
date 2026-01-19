/**
 * Trading Directions API
 * GET /api/v2/trading-directions
 *
 * Uses local ML server (localhost:5001) with full models
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const ML_SERVER_URL = process.env.ML_SERVER_URL || 'https://project-genesis-6roa.onrender.com'

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
        message: 'Start ML server with: cd ml && python -m server.app',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 503 }
    )
  }
}
