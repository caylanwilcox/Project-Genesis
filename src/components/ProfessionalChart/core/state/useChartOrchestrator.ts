import { useCallback, useMemo, useRef, useState } from 'react'
import type { CandleData, ProfessionalChartProps } from '../../types'
import type { PriceTag } from '../../priceLines'
import { useChartData, useCurrentTime, useFullscreen, useVisibleRange } from '../../hooks'
import { filterMarketHoursData } from '@/utils/marketHours'
import { useChartScaling } from '../../useChartScaling'
import { useChartInteraction } from '../../useChartInteraction'
import { useTimeframeState } from './useTimeframeState'
import { useChartViewport } from './useChartViewport'
import { generateMockData } from '../../utils/generators/mockDataGenerator'

/**
 * Master orchestrator hook that coordinates all chart state
 */
export function useChartOrchestrator(props: ProfessionalChartProps) {
  const {
    currentPrice,
    data: externalData,
    onDataUpdate,
    onTimeframeChange,
    onLoadMoreData,
    isLoadingMore,
    isTimeframeCached,
  } = props

  const containerRef = useRef<HTMLDivElement>(null)
  const chartAreaRef = useRef<HTMLDivElement>(null)
  const [overlayTags, setOverlayTags] = useState<PriceTag[]>([])

  // Data management
  const { data } = useChartData(externalData, currentPrice, onDataUpdate)

  // Track dataTimeframe separately for filtering before timeframe hook
  const [dataTimeframeForFilter, setDataTimeframeForFilter] = useState('1m')

  const chartData = useMemo(() => {
    let processedData = data.length > 0 ? data : generateMockData(currentPrice)

    // Filter out premarket/after-hours for intraday timeframes
    const isIntraday = ['1m', '5m', '15m', '30m', '1h'].includes(dataTimeframeForFilter)
    if (isIntraday) {
      processedData = filterMarketHoursData(processedData)
    }

    return processedData
  }, [data, currentPrice, dataTimeframeForFilter])

  // Viewport state
  const viewport = useChartViewport(chartData.length)

  // Scaling state
  const scaling = useChartScaling()

  // Reset scales helper
  const resetScales = useCallback(() => {
    viewport.actions.resetOffsets()
    scaling.setPriceScale(1)
    scaling.setTimeScale(1)
  }, [viewport.actions, scaling])

  // Timeframe state
  const timeframe = useTimeframeState({
    onTimeframeChange: (tf, displayTf, intervalLabel) => {
      setDataTimeframeForFilter(tf)
      onTimeframeChange?.(tf, displayTf, intervalLabel)
    },
    onResetScales: resetScales,
    // Only allow zoom transitions to timeframes that are already cached
    canTransitionTo: isTimeframeCached,
  })

  // Visible range calculation
  const visibleRange = useVisibleRange(
    chartData,
    viewport.state.panOffset,
    scaling.timeScale,
    timeframe.state.displayTimeframe,
    timeframe.state.dataTimeframe
  )

  // Interaction handlers
  const handleReachLeftEdge = useCallback(() => {
    if (!onLoadMoreData || isLoadingMore) return
    onLoadMoreData()
  }, [isLoadingMore, onLoadMoreData])

  const interaction = useChartInteraction(
    chartData,
    viewport.state.panOffset,
    viewport.actions.setPanOffset,
    scaling.timeScale,
    scaling.setTimeScale,
    viewport.state.priceOffset,
    viewport.actions.setPriceOffset,
    timeframe.state.displayTimeframe,
    timeframe.state.dataTimeframe,
    handleReachLeftEdge,
    timeframe.actions.checkZoomTransition
  )

  // Time and fullscreen
  const currentTime = useCurrentTime()
  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef)

  return {
    refs: { containerRef, chartAreaRef },
    data: { chartData, overlayTags, setOverlayTags },
    viewport,
    scaling,
    timeframe,
    interaction,
    visibleRange,
    currentTime,
    isFullscreen,
    toggleFullscreen,
  }
}
