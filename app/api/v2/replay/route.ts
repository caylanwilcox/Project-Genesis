/**
 * Replay Mode API
 * GET /api/v2/replay?date=2025-12-20&time=14:30
 *
 * Time-travel testing - replay any historical day at any point in time.
 * Returns V6 signals + Northstar analysis as they would have appeared.
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
    const date = searchParams.get('date')
    const time = searchParams.get('time')
    const ticker = searchParams.get('ticker')

    if (!date || !time) {
      return NextResponse.json(
        { error: 'Missing required parameters: date and time' },
        { status: 400 }
      )
    }

    const url = new URL(`${ML_SERVER_URL}/replay`)
    url.searchParams.set('date', date)
    url.searchParams.set('time', time)
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
