'use client'

import { useState, useEffect } from 'react'

const ML_SERVER_URL = process.env.NEXT_PUBLIC_ML_SERVER_URL || 'https://project-genesis-6roa.onrender.com'

interface RangeData {
  high: number
  low: number
  color: string
  open?: number
  close?: number
  last?: number
  range?: number
}

interface TickerRanges {
  ticker: string
  timestamp: string
  aftermarket?: RangeData | null
  rolling_24h?: RangeData | null
  premarket?: RangeData | null
  first_30min?: RangeData | null
  current_session?: RangeData | null
  yesterday_session?: RangeData | null
  ml_refined?: {
    available: boolean
    capture_rate?: number
  }
  error?: string
}

interface PriceRangesProps {
  ticker: string
  currentPrice?: number
}

const RANGE_LABELS: Record<string, { label: string; shortLabel: string; description: string }> = {
  aftermarket: {
    label: 'After-Hours',
    shortLabel: 'AH',
    description: 'Previous day 4PM-8PM ET'
  },
  premarket: {
    label: 'Pre-Market',
    shortLabel: 'PM',
    description: '4AM-9:30AM ET'
  },
  first_30min: {
    label: 'First 30 Min',
    shortLabel: '30M',
    description: '9:30AM-10AM ET range'
  },
  current_session: {
    label: 'Session',
    shortLabel: 'RTH',
    description: 'Today\'s regular hours'
  },
  rolling_24h: {
    label: '24 Hour',
    shortLabel: '24H',
    description: 'Rolling 24-hour range'
  },
}

export function PriceRanges({ ticker, currentPrice }: PriceRangesProps) {
  const [ranges, setRanges] = useState<TickerRanges | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchRanges = async () => {
      try {
        setIsLoading(true)
        const response = await fetch(`${ML_SERVER_URL}/price_ranges`)
        const data = await response.json()

        if (data.tickers && data.tickers[ticker]) {
          setRanges(data.tickers[ticker])
          setError(null)
        } else {
          setError('No data available')
        }
      } catch (err) {
        setError('Failed to fetch price ranges')
        console.error(err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchRanges()
    const interval = setInterval(fetchRanges, 60000) // Refresh every minute

    return () => clearInterval(interval)
  }, [ticker])

  if (isLoading) {
    return (
      <div className="bg-gray-900/50 rounded-lg p-3 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-24 mb-2"></div>
        <div className="space-y-1">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="h-6 bg-gray-800 rounded"></div>
          ))}
        </div>
      </div>
    )
  }

  if (error || !ranges) {
    return (
      <div className="bg-gray-900/50 rounded-lg p-3 text-gray-500 text-xs">
        {error || 'No range data'}
      </div>
    )
  }

  // Calculate overall price range for positioning
  const allRanges = [
    ranges.aftermarket,
    ranges.premarket,
    ranges.first_30min,
    ranges.current_session,
    ranges.rolling_24h,
  ].filter(Boolean) as RangeData[]

  if (allRanges.length === 0) {
    return (
      <div className="bg-gray-900/50 rounded-lg p-3 text-gray-500 text-xs">
        No range data available
      </div>
    )
  }

  const overallHigh = Math.max(...allRanges.map(r => r.high))
  const overallLow = Math.min(...allRanges.map(r => r.low))
  const overallRange = overallHigh - overallLow

  // Calculate position as percentage
  const getPosition = (price: number) => {
    if (overallRange === 0) return 50
    return ((price - overallLow) / overallRange) * 100
  }

  const rangeKeys = ['aftermarket', 'premarket', 'first_30min', 'current_session', 'rolling_24h'] as const

  return (
    <div className="bg-gray-900/80 rounded-lg p-3 border border-gray-800">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Price Ranges
        </h3>
        {ranges.ml_refined?.available && (
          <span className="text-[10px] text-cyan-400 bg-cyan-500/10 px-1.5 py-0.5 rounded">
            ML: {(ranges.ml_refined.capture_rate! * 100).toFixed(0)}% capture
          </span>
        )}
      </div>

      {/* Visual Range Bars */}
      <div className="space-y-2 mb-3">
        {rangeKeys.map(key => {
          const range = ranges[key]
          if (!range) return null

          const label = RANGE_LABELS[key]
          const leftPos = getPosition(range.low)
          const rightPos = getPosition(range.high)
          const width = rightPos - leftPos

          return (
            <div key={key} className="relative">
              <div className="flex items-center justify-between text-[10px] text-gray-500 mb-0.5">
                <span title={label.description}>{label.shortLabel}</span>
                <span className="text-gray-600">
                  ${range.low.toFixed(2)} - ${range.high.toFixed(2)}
                </span>
              </div>
              <div className="h-4 bg-gray-800/50 rounded relative overflow-hidden">
                <div
                  className="absolute h-full rounded opacity-60"
                  style={{
                    left: `${leftPos}%`,
                    width: `${Math.max(width, 2)}%`,
                    backgroundColor: range.color,
                  }}
                />
                {/* Current price marker */}
                {currentPrice && currentPrice >= overallLow && currentPrice <= overallHigh && (
                  <div
                    className="absolute w-0.5 h-full bg-white"
                    style={{ left: `${getPosition(currentPrice)}%` }}
                  />
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-2 text-[10px]">
        {rangeKeys.map(key => {
          const range = ranges[key]
          if (!range) return null
          const label = RANGE_LABELS[key]

          return (
            <div key={key} className="flex items-center gap-1">
              <div
                className="w-2 h-2 rounded-sm"
                style={{ backgroundColor: range.color }}
              />
              <span className="text-gray-500">{label.label}</span>
            </div>
          )
        })}
      </div>

      {/* Current Price Position */}
      {currentPrice && ranges.current_session && (
        <div className="mt-3 pt-2 border-t border-gray-800">
          <div className="flex justify-between text-xs">
            <span className="text-gray-500">Position in Range:</span>
            <span className={
              currentPrice > (ranges.current_session.high + ranges.current_session.low) / 2
                ? 'text-green-400'
                : 'text-red-400'
            }>
              {((currentPrice - ranges.current_session.low) / (ranges.current_session.high - ranges.current_session.low) * 100).toFixed(0)}%
              {currentPrice > (ranges.current_session.high + ranges.current_session.low) / 2 ? ' (upper half)' : ' (lower half)'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

// Compact version for use in signal cards
export function PriceRangesCompact({ ticker, currentPrice }: PriceRangesProps) {
  const [ranges, setRanges] = useState<TickerRanges | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchRanges = async () => {
      try {
        const response = await fetch(`${ML_SERVER_URL}/price_ranges`)
        const data = await response.json()
        if (data.tickers && data.tickers[ticker]) {
          setRanges(data.tickers[ticker])
        }
      } catch (err) {
        console.error(err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchRanges()
  }, [ticker])

  if (isLoading || !ranges) return null

  const displayRanges = [
    { key: 'first_30min', data: ranges.first_30min, label: '30M' },
    { key: 'current_session', data: ranges.current_session, label: 'RTH' },
    { key: 'rolling_24h', data: ranges.rolling_24h, label: '24H' },
  ].filter(r => r.data)

  return (
    <div className="flex gap-2 text-[10px]">
      {displayRanges.map(({ key, data, label }) => (
        <div key={key} className="flex items-center gap-1">
          <div
            className="w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: data!.color }}
          />
          <span className="text-gray-500">{label}:</span>
          <span className="text-gray-300 font-mono">
            ${data!.low.toFixed(0)}-${data!.high.toFixed(0)}
          </span>
        </div>
      ))}
    </div>
  )
}
