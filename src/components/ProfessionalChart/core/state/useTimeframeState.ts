import { useState, useMemo, useCallback, useRef } from 'react'
import {
  resolveDisplayToData,
  intervalLabelToTimeframe,
  getNextCoarserTimeframe,
  getNextFinerTimeframe,
  getZoomTransitionThresholds
} from '@/utils/timeframePolicy'
import type { Timeframe } from '@/types/polygon'

const DEFAULT_TIMEFRAME = '1D'

export interface TimeframeState {
  displayTimeframe: string
  dataTimeframe: Timeframe
  interval: string
  showIntervalDropdown: boolean
}

export interface TimeframeActions {
  handleTimeframeClick: (tf: string) => void
  handleIntervalChange: (newInterval: string) => void
  toggleIntervalDropdown: () => void
  setShowIntervalDropdown: (show: boolean) => void
  transitionToCoarserTimeframe: () => string | undefined
  transitionToFinerTimeframe: () => string | undefined
  checkZoomTransition: (timeScale: number) => string | undefined
}

interface UseTimeframeStateProps {
  onTimeframeChange?: (tf: string, displayTf: string, intervalLabel?: string) => void
  onResetScales?: () => void
  // Optional: check if a timeframe transition should be allowed (e.g., is data cached?)
  canTransitionTo?: (displayTf: string) => boolean
}

export function useTimeframeState({
  onTimeframeChange,
  onResetScales,
  canTransitionTo
}: UseTimeframeStateProps = {}) {
  const initialMapping = useMemo(() => resolveDisplayToData(DEFAULT_TIMEFRAME), [])

  const [displayTimeframe, setDisplayTimeframe] = useState(DEFAULT_TIMEFRAME)
  const [interval, setInterval] = useState(initialMapping.intervalLabel)
  const [dataTimeframe, setDataTimeframe] = useState<Timeframe>(initialMapping.timeframe)
  const [showIntervalDropdown, setShowIntervalDropdown] = useState(false)

  const handleTimeframeClick = useCallback(
    (tf: string) => {
      if (tf === displayTimeframe) return
      console.log(`[⏱️ TIMEFRAME] ${tf}`)
      const { timeframe, intervalLabel } = resolveDisplayToData(tf)
      setDisplayTimeframe(tf)
      setDataTimeframe(timeframe)
      setInterval(intervalLabel)
      setShowIntervalDropdown(false)
      onResetScales?.()
      onTimeframeChange?.(timeframe, tf, intervalLabel)
    },
    [displayTimeframe, onTimeframeChange, onResetScales]
  )

  const handleIntervalChange = useCallback(
    (newInterval: string) => {
      console.log(`[🔧 INTERVAL] ${newInterval}`)
      setInterval(newInterval)
      setShowIntervalDropdown(false)
      const mapped = intervalLabelToTimeframe(newInterval)
      if (mapped) {
        setDataTimeframe(mapped)
        setDisplayTimeframe('Custom')
        onResetScales?.()
        onTimeframeChange?.(mapped, 'Custom', newInterval)
      }
    },
    [onTimeframeChange, onResetScales]
  )

  const toggleIntervalDropdown = useCallback(() => {
    if (displayTimeframe !== 'Custom') {
      setDisplayTimeframe('Custom')
      setShowIntervalDropdown(true)
      onTimeframeChange?.(dataTimeframe, 'Custom', interval)
      return
    }
    setShowIntervalDropdown((prev) => !prev)
  }, [displayTimeframe, dataTimeframe, interval, onTimeframeChange])

  // Track last transition time to debounce rapid transitions
  const lastTransitionRef = useRef<number>(0)
  const TRANSITION_DEBOUNCE_MS = 500

  const transitionToCoarserTimeframe = useCallback(() => {
    const now = Date.now()
    if (now - lastTransitionRef.current < TRANSITION_DEBOUNCE_MS) return undefined

    const nextTf = getNextCoarserTimeframe(displayTimeframe)
    if (nextTf) {
      // Check if we're allowed to transition (e.g., is data cached?)
      if (canTransitionTo && !canTransitionTo(nextTf)) {
        console.log(`[⏱️ ZOOM TRANSITION] ${displayTimeframe} → ${nextTf} blocked (not cached)`)
        return undefined
      }

      console.log(`[⏱️ ZOOM TRANSITION] ${displayTimeframe} → ${nextTf} (coarser)`)
      lastTransitionRef.current = now
      const { timeframe, intervalLabel } = resolveDisplayToData(nextTf)
      setDisplayTimeframe(nextTf)
      setDataTimeframe(timeframe)
      setInterval(intervalLabel)
      // Don't reset scales - preserve zoom continuity
      onTimeframeChange?.(timeframe, nextTf, intervalLabel)
    }
    return nextTf
  }, [displayTimeframe, onTimeframeChange, canTransitionTo])

  const transitionToFinerTimeframe = useCallback(() => {
    const now = Date.now()
    if (now - lastTransitionRef.current < TRANSITION_DEBOUNCE_MS) return undefined

    const nextTf = getNextFinerTimeframe(displayTimeframe)
    if (nextTf) {
      // Check if we're allowed to transition (e.g., is data cached?)
      if (canTransitionTo && !canTransitionTo(nextTf)) {
        console.log(`[⏱️ ZOOM TRANSITION] ${displayTimeframe} → ${nextTf} blocked (not cached)`)
        return undefined
      }

      console.log(`[⏱️ ZOOM TRANSITION] ${displayTimeframe} → ${nextTf} (finer)`)
      lastTransitionRef.current = now
      const { timeframe, intervalLabel } = resolveDisplayToData(nextTf)
      setDisplayTimeframe(nextTf)
      setDataTimeframe(timeframe)
      setInterval(intervalLabel)
      // Don't reset scales - preserve zoom continuity
      onTimeframeChange?.(timeframe, nextTf, intervalLabel)
    }
    return nextTf
  }, [displayTimeframe, onTimeframeChange, canTransitionTo])

  const checkZoomTransition = useCallback((timeScale: number): string | undefined => {
    const thresholds = getZoomTransitionThresholds(displayTimeframe)

    if (timeScale <= thresholds.zoomOutThreshold) {
      return transitionToCoarserTimeframe()
    } else if (timeScale >= thresholds.zoomInThreshold) {
      return transitionToFinerTimeframe()
    }
    return undefined
  }, [displayTimeframe, transitionToCoarserTimeframe, transitionToFinerTimeframe])

  const state: TimeframeState = {
    displayTimeframe,
    dataTimeframe,
    interval,
    showIntervalDropdown,
  }

  const actions: TimeframeActions = {
    handleTimeframeClick,
    handleIntervalChange,
    toggleIntervalDropdown,
    setShowIntervalDropdown,
    transitionToCoarserTimeframe,
    transitionToFinerTimeframe,
    checkZoomTransition,
  }

  return { state, actions }
}
