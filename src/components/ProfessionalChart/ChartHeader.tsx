'use client'

import React from 'react'
import { Activity, TrendingUp, Maximize2, Minimize2, Settings, ChevronDown } from 'lucide-react'
import { DISPLAY_TIMEFRAMES, INTERVAL_LABELS } from '@/utils/timeframePolicy'

interface ChartHeaderProps {
  symbol: string
  timeframe: string
  interval: string
  showIntervalDropdown: boolean
  isFullscreen: boolean
  onTimeframeClick: (tf: string) => void
  onIntervalClick: () => void
  onIntervalChange: (interval: string) => void
  onFullscreenToggle: () => void
}

export function ChartHeader({
  symbol, timeframe, interval, showIntervalDropdown, isFullscreen,
  onTimeframeClick, onIntervalClick, onIntervalChange, onFullscreenToggle,
}: ChartHeaderProps) {
  const isCustomMode = timeframe === 'Custom'

  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-blue-400" />
          <span className="text-sm font-semibold text-white">{symbol}</span>
        </div>
        <div className="flex items-center gap-2">
          <TrendingUp size={14} className="text-green-400" />
          <span className="text-xs text-gray-400">Live Chart</span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <div className="flex gap-1 items-center">
          {DISPLAY_TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              onClick={() => onTimeframeClick(tf)}
              className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                timeframe === tf ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {tf}
            </button>
          ))}
          <button
            onClick={onIntervalClick}
            className={`flex items-center gap-1 px-2 py-1 text-xs font-medium rounded transition-colors ${
              isCustomMode ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            {isCustomMode ? `Interval: ${interval}` : 'Custom Interval'}
            <ChevronDown size={12} />
          </button>
        </div>

        <div className="relative interval-dropdown-container">
          {isCustomMode && showIntervalDropdown && (
            <div className="absolute left-0 top-full mt-1 bg-gray-800 border border-gray-700 rounded shadow-lg z-50 min-w-[120px]">
              {INTERVAL_LABELS.map((int) => (
                <button
                  key={int}
                  onClick={() => onIntervalChange(int)}
                  className={`w-full text-left px-3 py-2 text-xs hover:bg-gray-700 transition-colors first:rounded-t last:rounded-b ${
                    interval === int ? 'text-blue-400 bg-gray-700/50' : 'text-gray-300'
                  }`}
                >
                  {int}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="w-px h-4 bg-gray-700 mx-1" />

        <button className="text-gray-400 hover:text-white p-1" title="Settings">
          <Settings size={14} />
        </button>
        <button onClick={onFullscreenToggle} className="text-gray-400 hover:text-white p-1" title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
          {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
        </button>
      </div>
    </div>
  )
}
