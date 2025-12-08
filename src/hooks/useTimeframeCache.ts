'use client'

import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { polygonService } from '../services/polygonService'
import { NormalizedChartData, Timeframe } from '../types/polygon'
import {
  resolveDisplayToData,
  recommendedBarLimit,
  getNextCoarserTimeframe,
  getNextFinerTimeframe,
  TIMEFRAME_PROGRESSION
} from '../utils/timeframePolicy'

interface CachedTimeframeData {
  displayTimeframe: string
  dataTimeframe: Timeframe
  data: NormalizedChartData[]
  fetchedAt: number
  isStale: boolean
}

interface UseTimeframeCacheOptions {
  ticker: string
  initialDisplayTimeframe?: string
  prefetchAdjacent?: boolean  // Whether to prefetch neighboring timeframes
  cacheMaxAge?: number        // Max age in ms before data is considered stale (default 5 min)
}

interface UseTimeframeCacheResult {
  // Current timeframe data
  data: NormalizedChartData[]
  currentPrice: number | null
  isLoading: boolean
  error: Error | null

  // Current timeframe info
  displayTimeframe: string
  dataTimeframe: Timeframe

  // Timeframe switching - returns true if data was available from cache (instant)
  switchTimeframe: (newDisplayTimeframe: string) => boolean

  // Check if a timeframe is cached and ready
  isTimeframeCached: (displayTimeframe: string) => boolean

  // Force refresh current timeframe
  refetch: () => Promise<void>

  // Prefetch status
  prefetchingTimeframes: string[]
}

const CACHE_MAX_AGE_DEFAULT = 5 * 60 * 1000  // 5 minutes

export function useTimeframeCache({
  ticker,
  initialDisplayTimeframe = '1D',
  prefetchAdjacent = true,
  cacheMaxAge = CACHE_MAX_AGE_DEFAULT,
}: UseTimeframeCacheOptions): UseTimeframeCacheResult {
  // Cache storage - keyed by displayTimeframe
  const cacheRef = useRef<Map<string, CachedTimeframeData>>(new Map())

  // Current active timeframe
  const [displayTimeframe, setDisplayTimeframe] = useState(initialDisplayTimeframe)
  const [dataTimeframe, setDataTimeframe] = useState<Timeframe>(() => {
    return resolveDisplayToData(initialDisplayTimeframe).timeframe
  })

  // Current data state
  const [data, setData] = useState<NormalizedChartData[]>([])
  const [currentPrice, setCurrentPrice] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  // Track which timeframes are currently being prefetched
  const [prefetchingTimeframes, setPrefetchingTimeframes] = useState<string[]>([])

  // Pending prefetch abort controller
  const prefetchAbortRef = useRef<AbortController | null>(null)

  // Fetch data for a specific timeframe
  const fetchTimeframeData = useCallback(async (
    displayTf: string,
    signal?: AbortSignal
  ): Promise<CachedTimeframeData | null> => {
    if (!ticker) return null

    const { timeframe: dataTf, intervalLabel } = resolveDisplayToData(displayTf)
    const limit = recommendedBarLimit(dataTf, displayTf)

    try {
      console.log(`[TimeframeCache] Fetching ${displayTf} (${dataTf}) - ${limit} bars`)

      const aggregates = await polygonService.getAggregates(
        ticker.toUpperCase(),
        dataTf,
        limit,
        displayTf
      )

      if (signal?.aborted) return null

      const cached: CachedTimeframeData = {
        displayTimeframe: displayTf,
        dataTimeframe: dataTf,
        data: aggregates,
        fetchedAt: Date.now(),
        isStale: false,
      }

      // Store in cache
      cacheRef.current.set(displayTf, cached)

      console.log(`[TimeframeCache] Cached ${displayTf}: ${aggregates.length} bars`)
      return cached
    } catch (err) {
      if (signal?.aborted) return null
      console.error(`[TimeframeCache] Failed to fetch ${displayTf}:`, err)
      throw err
    }
  }, [ticker])

  // Prefetch adjacent timeframes in the background
  const prefetchAdjacentTimeframes = useCallback(async (currentDisplayTf: string) => {
    if (!prefetchAdjacent) return

    // Cancel any pending prefetch
    prefetchAbortRef.current?.abort()
    prefetchAbortRef.current = new AbortController()
    const signal = prefetchAbortRef.current.signal

    const timeframesToPrefetch: string[] = []

    // Get adjacent timeframes
    const coarser = getNextCoarserTimeframe(currentDisplayTf)
    const finer = getNextFinerTimeframe(currentDisplayTf)

    if (coarser && !cacheRef.current.has(coarser)) {
      timeframesToPrefetch.push(coarser)
    }
    if (finer && !cacheRef.current.has(finer)) {
      timeframesToPrefetch.push(finer)
    }

    if (timeframesToPrefetch.length === 0) return

    setPrefetchingTimeframes(timeframesToPrefetch)
    console.log(`[TimeframeCache] Prefetching adjacent: ${timeframesToPrefetch.join(', ')}`)

    // Fetch in parallel with small delay to not compete with main request
    await new Promise(resolve => setTimeout(resolve, 500))

    if (signal.aborted) return

    await Promise.allSettled(
      timeframesToPrefetch.map(tf => fetchTimeframeData(tf, signal))
    )

    setPrefetchingTimeframes([])
  }, [prefetchAdjacent, fetchTimeframeData])

  // Check if cached data is still valid
  const isCacheValid = useCallback((cached: CachedTimeframeData | undefined): boolean => {
    if (!cached) return false
    const age = Date.now() - cached.fetchedAt
    return age < cacheMaxAge && !cached.isStale
  }, [cacheMaxAge])

  // Check if a timeframe is cached
  const isTimeframeCached = useCallback((displayTf: string): boolean => {
    return isCacheValid(cacheRef.current.get(displayTf))
  }, [isCacheValid])

  // Switch to a different timeframe
  const switchTimeframe = useCallback((newDisplayTf: string): boolean => {
    if (newDisplayTf === displayTimeframe) return true

    const cached = cacheRef.current.get(newDisplayTf)
    const { timeframe: newDataTf } = resolveDisplayToData(newDisplayTf)

    // Update state
    setDisplayTimeframe(newDisplayTf)
    setDataTimeframe(newDataTf)

    if (isCacheValid(cached)) {
      // Instant switch from cache!
      console.log(`[TimeframeCache] Instant switch to ${newDisplayTf} from cache`)
      setData(cached!.data)
      setCurrentPrice(cached!.data[cached!.data.length - 1]?.close ?? null)
      setIsLoading(false)
      setError(null)

      // Prefetch new adjacent timeframes in background
      prefetchAdjacentTimeframes(newDisplayTf)

      return true
    } else {
      // Need to fetch - show loading
      console.log(`[TimeframeCache] Fetching ${newDisplayTf} (not cached)`)
      setIsLoading(true)

      // Fetch will be triggered by the effect below
      return false
    }
  }, [displayTimeframe, isCacheValid, prefetchAdjacentTimeframes])

  // Refetch current timeframe
  const refetch = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const cached = await fetchTimeframeData(displayTimeframe)
      if (cached) {
        setData(cached.data)
        setCurrentPrice(cached.data[cached.data.length - 1]?.close ?? null)
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch data'))
    } finally {
      setIsLoading(false)
    }
  }, [displayTimeframe, fetchTimeframeData])

  // Initial fetch and when timeframe changes
  useEffect(() => {
    const cached = cacheRef.current.get(displayTimeframe)

    if (isCacheValid(cached)) {
      // Use cached data
      setData(cached!.data)
      setCurrentPrice(cached!.data[cached!.data.length - 1]?.close ?? null)
      setIsLoading(false)
      setError(null)
      prefetchAdjacentTimeframes(displayTimeframe)
    } else {
      // Need to fetch
      setIsLoading(true)
      setError(null)

      fetchTimeframeData(displayTimeframe)
        .then(result => {
          if (result) {
            setData(result.data)
            setCurrentPrice(result.data[result.data.length - 1]?.close ?? null)
            prefetchAdjacentTimeframes(displayTimeframe)
          }
        })
        .catch(err => {
          setError(err instanceof Error ? err : new Error('Failed to fetch data'))
        })
        .finally(() => {
          setIsLoading(false)
        })
    }
  }, [displayTimeframe, ticker])  // Note: intentionally excluding other deps to avoid loops

  // Clear cache when ticker changes
  useEffect(() => {
    cacheRef.current.clear()
  }, [ticker])

  return {
    data,
    currentPrice,
    isLoading,
    error,
    displayTimeframe,
    dataTimeframe,
    switchTimeframe,
    isTimeframeCached,
    refetch,
    prefetchingTimeframes,
  }
}
