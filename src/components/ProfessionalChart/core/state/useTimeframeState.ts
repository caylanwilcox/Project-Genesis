import { useState, useMemo, useCallback } from 'react'
import { resolveDisplayToData, intervalLabelToTimeframe } from '@/utils/timeframePolicy'
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
}

interface UseTimeframeStateProps {
  onTimeframeChange?: (tf: string, displayTf: string) => void
  onResetScales?: () => void
}

export function useTimeframeState({
  onTimeframeChange,
  onResetScales
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
  }

  return { state, actions }
}
