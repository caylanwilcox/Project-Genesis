'use client'

import React from 'react'

interface PriceScaleControlProps {
  priceScale: number
  isScaling: boolean
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void
  onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void
  onDoubleClick: () => void
}

export function PriceScaleControl({
  priceScale,
  isScaling,
  onMouseDown,
  onMouseMove,
  onDoubleClick,
}: PriceScaleControlProps) {
  return (
    <div
      className="absolute right-0 top-0 w-20 hover:bg-blue-500/5 transition-colors"
      style={{
        height: 'calc(100% - 80px)',
        cursor: isScaling ? 'ns-resize' : 'ns-resize',
        pointerEvents: 'auto'
      }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onDoubleClick={onDoubleClick}
      title="Drag to zoom price scale (double-click to reset)"
    >
      {/* Visual indicator for drag area */}
      <div className="absolute right-1 top-1/2 -translate-y-1/2 flex flex-col gap-1 opacity-30 hover:opacity-70 transition-opacity">
        <div className="w-3 h-0.5 bg-gray-400"></div>
        <div className="w-3 h-0.5 bg-gray-400"></div>
        <div className="w-3 h-0.5 bg-gray-400"></div>
      </div>

      {/* Zoom level indicator */}
      {priceScale !== 1.0 && (
        <div className="absolute right-2 bottom-4 bg-gray-800/90 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300">
          {priceScale.toFixed(1)}x
        </div>
      )}
    </div>
  )
}

interface TimeScaleControlProps {
  timeScale: number
  isTimeScaling: boolean
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void
  onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void
  onDoubleClick: () => void
}

export function TimeScaleControl({
  timeScale,
  isTimeScaling,
  onMouseDown,
  onMouseMove,
  onDoubleClick,
}: TimeScaleControlProps) {
  return (
    <div
      className="absolute bottom-20 left-0 right-20 h-8 hover:bg-blue-500/5 transition-colors"
      style={{
        cursor: isTimeScaling ? 'ew-resize' : 'ew-resize',
        pointerEvents: 'auto'
      }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onDoubleClick={onDoubleClick}
      title="Drag to zoom time scale (double-click to reset)"
    >
      {/* Visual indicator for drag area */}
      <div className="absolute left-1/2 -translate-x-1/2 bottom-1 flex gap-1 opacity-30 hover:opacity-70 transition-opacity">
        <div className="w-0.5 h-3 bg-gray-400"></div>
        <div className="w-0.5 h-3 bg-gray-400"></div>
        <div className="w-0.5 h-3 bg-gray-400"></div>
      </div>

      {/* Zoom level indicator */}
      {timeScale !== 1.0 && (
        <div className="absolute left-4 bottom-1 bg-gray-800/90 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300">
          {timeScale.toFixed(1)}x
        </div>
      )}
    </div>
  )
}
