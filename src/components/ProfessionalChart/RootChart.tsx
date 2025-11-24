'use client'

import React from 'react'
import styles from '../ProfessionalChart.module.css'
import type { ProfessionalChartProps } from './types'
import { ChartHeader } from './ChartHeader'
import { ChartFooter } from './ui/layout/ChartFooter'
import { ChartCanvas } from './ui/layout/ChartCanvas'
import { useChartOrchestrator } from './core/state/useChartOrchestrator'
import { useIntervalDropdown } from './interactions/gestures/useIntervalDropdown'

/**
 * ProfessionalChart - Root orchestrator component
 * Responsibility: Compose all chart sub-components and coordinate high-level state
 * SRP: Component composition only - delegates all logic to hooks and sub-components
 */
export const ProfessionalChart: React.FC<ProfessionalChartProps> = (props) => {
  const {
    symbol,
    stopLoss,
    targets,
    entryPoint,
    showFvg = false,
    onFvgCountChange,
    onVisibleBarCountChange,
    isLoadingMore = false,
  } = props

  const orchestrator = useChartOrchestrator(props)

  useIntervalDropdown({
    isOpen: orchestrator.timeframe.state.showIntervalDropdown,
    onClose: () => orchestrator.timeframe.actions.setShowIntervalDropdown(false),
  })

  const containerStyle = orchestrator.viewport.state.customHeight
    ? {
        minHeight: orchestrator.viewport.state.customHeight,
        height: orchestrator.viewport.state.customHeight,
      }
    : undefined

  const chartAreaClassName = `${styles.chartMainArea} ${
    orchestrator.interaction.isPanning ? styles.panning : styles.idle
  }`

  return (
    <div
      ref={orchestrator.refs.containerRef}
      className={styles.chartContainer}
      style={containerStyle}
    >
      <ChartHeader
        symbol={symbol}
        timeframe={orchestrator.timeframe.state.displayTimeframe}
        interval={orchestrator.timeframe.state.interval}
        showIntervalDropdown={orchestrator.timeframe.state.showIntervalDropdown}
        isFullscreen={orchestrator.isFullscreen}
        onTimeframeClick={orchestrator.timeframe.actions.handleTimeframeClick}
        onIntervalClick={orchestrator.timeframe.actions.toggleIntervalDropdown}
        onIntervalChange={orchestrator.timeframe.actions.handleIntervalChange}
        onFullscreenToggle={orchestrator.toggleFullscreen}
      />

      <ChartCanvas
        chartAreaRef={orchestrator.refs.chartAreaRef}
        containerRef={orchestrator.refs.containerRef}
        chartData={orchestrator.data.chartData}
        overlayTags={orchestrator.data.overlayTags}
        visibleRange={orchestrator.visibleRange}
        priceScale={orchestrator.scaling.priceScale}
        timeScale={orchestrator.scaling.timeScale}
        stopLoss={stopLoss}
        entryPoint={entryPoint}
        targets={targets}
        dataTimeframe={orchestrator.timeframe.state.dataTimeframe}
        displayTimeframe={orchestrator.timeframe.state.displayTimeframe}
        onOverlayTagsUpdate={orchestrator.data.setOverlayTags}
        mousePos={orchestrator.interaction.mousePos}
        isPanning={orchestrator.interaction.isPanning}
        showFvg={showFvg}
        onFvgCountChange={onFvgCountChange}
        onVisibleBarCountChange={onVisibleBarCountChange}
        priceOffset={orchestrator.viewport.state.priceOffset}
        isLoadingMore={isLoadingMore}
        customHeight={orchestrator.viewport.state.customHeight}
        onHeightChange={orchestrator.viewport.actions.setCustomHeight}
        className={chartAreaClassName}
        interaction={orchestrator.interaction}
        scaling={orchestrator.scaling}
      />

      <ChartFooter currentTime={orchestrator.currentTime} />
    </div>
  )
}

export default ProfessionalChart
