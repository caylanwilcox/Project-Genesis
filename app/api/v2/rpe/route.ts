/**
 * Reality Proof Engine (RPE) API
 * GET /api/v2/rpe?ticker=SPY
 *
 * 5-Phase market structure analysis with strict layering invariants.
 * Phase 1-4: No ML predictions (pure observation and rules)
 * Phase 5: ONLY phase with ML predictions (forecasts, entry/exit)
 */
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

const ML_SERVER_URL =
  process.env.ML_SERVER_URL ||
  process.env.NEXT_PUBLIC_ML_SERVER_URL ||
  'https://genesis-production-c1e9.up.railway.app'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const ticker = searchParams.get('ticker')

    const url = new URL(`${ML_SERVER_URL}/rpe`)
    if (ticker) url.searchParams.set('ticker', ticker)

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
