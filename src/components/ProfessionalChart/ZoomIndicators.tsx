'use client'

import React from 'react'

interface ZoomIndicatorsProps {
  timeScale: number
  onTimeZoomIn: () => void
  onTimeZoomOut: () => void
  onTimeReset: () => void
}

export const ZoomIndicators: React.FC<ZoomIndicatorsProps> = ({
  timeScale, onTimeZoomIn, onTimeZoomOut, onTimeReset
}) => {
  return (
    <>
      {/* Time Zoom Controls - Bottom center */}
      <div className="absolute bottom-24 left-1/2 -translate-x-1/2 flex gap-2 bg-gray-800/90 rounded-lg p-2 border border-gray-700 z-10">
        <button onClick={onTimeZoomOut} className="text-white hover:text-blue-400 transition-colors" title="Zoom out (X)">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M4 10H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
        <div className="text-xs text-gray-400 text-center font-mono min-w-[32px]">{timeScale.toFixed(1)}x</div>
        <button onClick={onTimeZoomIn} className="text-white hover:text-blue-400 transition-colors" title="Zoom in (X)">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M10 4V16M4 10H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
        {timeScale !== 1.0 && (
          <button onClick={onTimeReset} className="text-xs text-gray-400 hover:text-white transition-colors ml-2" title="Reset zoom">
            Reset
          </button>
        )}
      </div>
    </>
  )
}
