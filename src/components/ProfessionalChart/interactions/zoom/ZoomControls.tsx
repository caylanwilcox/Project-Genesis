import React from 'react'
import interactionStyles from '../../../ChartInteraction.module.css'

interface ZoomControlsProps {
  priceScale: number
  timeScale: number
  onPriceScaleMouseDown: (e: React.MouseEvent) => void
  onPriceScaleMouseMove: (e: React.MouseEvent) => void
  onPriceScaleDoubleClick: () => void
  onTimeScaleMouseDown: (e: React.MouseEvent) => void
  onTimeScaleMouseMove: (e: React.MouseEvent) => void
  onTimeScaleDoubleClick: () => void
}

/**
 * Interactive zoom controls for price and time scaling
 */
export const ZoomControls: React.FC<ZoomControlsProps> = ({
  priceScale,
  timeScale,
  onPriceScaleMouseDown,
  onPriceScaleMouseMove,
  onPriceScaleDoubleClick,
  onTimeScaleMouseDown,
  onTimeScaleMouseMove,
  onTimeScaleDoubleClick,
}) => {
  return (
    <>
      {/* Price Scale Drag Area */}
      <div
        className={interactionStyles.priceScaleDragArea}
        onMouseDown={onPriceScaleMouseDown}
        onMouseMove={onPriceScaleMouseMove}
        onDoubleClick={onPriceScaleDoubleClick}
        title="Drag to zoom price scale (double-click to reset)"
      >
        <div className={interactionStyles.dragIndicator}>
          <div className={interactionStyles.dragIndicatorLine} />
          <div className={interactionStyles.dragIndicatorLine} />
          <div className={interactionStyles.dragIndicatorLine} />
        </div>
        {priceScale !== 1.0 && (
          <div className={interactionStyles.zoomIndicator}>
            {priceScale.toFixed(1)}x
          </div>
        )}
      </div>

      {/* Time Scale Drag Area */}
      <div
        className={interactionStyles.timeScaleDragArea}
        onMouseDown={onTimeScaleMouseDown}
        onMouseMove={onTimeScaleMouseMove}
        onDoubleClick={onTimeScaleDoubleClick}
        title="Drag horizontally to zoom time scale (double-click to reset)"
      >
        <div className={interactionStyles.timeScaleDragIndicator}>
          <div className={interactionStyles.timeScaleDragIndicatorLine} />
          <div className={interactionStyles.timeScaleDragIndicatorLine} />
          <div className={interactionStyles.timeScaleDragIndicatorLine} />
        </div>
        {timeScale !== 1.0 && (
          <div className={interactionStyles.timeZoomIndicator}>
            {timeScale.toFixed(1)}x
          </div>
        )}
      </div>
    </>
  )
}
