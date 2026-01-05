'use client'

import React from 'react'
import type { V6Prediction } from './types'

export interface MLOverlayProps {
  /** V6 model prediction for the current ticker */
  v6Prediction?: V6Prediction
  /** Show/hide the ML overlay */
  showMLOverlay?: boolean
  /** Show session divider lines (9:30 AM, 11 AM) */
  showSessionLines?: boolean
  /** 11 AM price for Target B reference line */
  price11am?: number
  /** Chart dimensions for positioning */
  chartWidth: number
  chartHeight: number
  padding: { left: number; top: number; right: number; bottom: number }
}

/**
 * MLOverlay - Displays V6 ML predictions on the chart
 *
 * Components:
 * 1. Direction Banner - Shows current prediction at top of chart
 * 2. Session Lines - Vertical lines at key times (future)
 * 3. Prediction Zones - Shaded regions (future)
 */
export const MLOverlay: React.FC<MLOverlayProps> = ({
  v6Prediction,
  showMLOverlay = true,
  showSessionLines = false,
  price11am,
  chartWidth,
  chartHeight,
  padding,
}) => {
  if (!showMLOverlay || !v6Prediction) {
    return null
  }

  const { direction, probability_a, probability_b, confidence, session, action } = v6Prediction

  // Determine which probability to show based on session
  const activeProb = session === 'early' ? probability_a : probability_b
  const activeProbPct = Math.round(activeProb * 100)
  const targetLabel = session === 'early' ? 'Target A: Close > Open' : 'Target B: Close > 11AM'

  // Color based on direction
  const directionColor = direction === 'BULLISH'
    ? '#22c55e'  // green
    : direction === 'BEARISH'
      ? '#ef4444'  // red
      : '#6b7280'  // gray for neutral

  // Confidence-based opacity (higher confidence = more visible)
  const bannerOpacity = Math.max(0.7, confidence / 100)

  // Arrow icon
  const arrow = direction === 'BULLISH' ? '▲' : direction === 'BEARISH' ? '▼' : '●'

  // Action badge color
  const actionColor = action === 'BUY_CALL'
    ? '#22c55e'
    : action === 'BUY_PUT'
      ? '#ef4444'
      : '#6b7280'

  return (
    <div
      style={{
        position: 'absolute',
        top: padding.top,
        left: padding.left,
        right: padding.right,
        pointerEvents: 'none',
        zIndex: 10,
      }}
    >
      {/* Direction Banner */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 12px',
          backgroundColor: `rgba(0, 0, 0, ${bannerOpacity * 0.8})`,
          borderRadius: '4px',
          border: `1px solid ${directionColor}`,
          width: 'fit-content',
          fontSize: '12px',
          fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif',
        }}
      >
        {/* Direction indicator */}
        <span style={{ color: directionColor, fontSize: '14px', fontWeight: 'bold' }}>
          {arrow}
        </span>

        {/* Direction text */}
        <span style={{ color: directionColor, fontWeight: 'bold' }}>
          {direction}
        </span>

        {/* Probability */}
        <span style={{ color: '#e5e7eb' }}>
          {activeProbPct}%
        </span>

        {/* Separator */}
        <span style={{ color: '#6b7280' }}>|</span>

        {/* Target label */}
        <span style={{ color: '#9ca3af', fontSize: '11px' }}>
          {targetLabel}
        </span>

        {/* Action badge */}
        <span
          style={{
            backgroundColor: actionColor,
            color: '#fff',
            padding: '2px 6px',
            borderRadius: '3px',
            fontSize: '10px',
            fontWeight: 'bold',
            marginLeft: '4px',
          }}
        >
          {action.replace('_', ' ')}
        </span>
      </div>

      {/* Confidence bar */}
      <div
        style={{
          marginTop: '4px',
          width: '200px',
          height: '3px',
          backgroundColor: 'rgba(107, 114, 128, 0.3)',
          borderRadius: '2px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${confidence}%`,
            height: '100%',
            backgroundColor: directionColor,
            transition: 'width 0.3s ease',
          }}
        />
      </div>

      {/* Session indicator */}
      <div
        style={{
          marginTop: '2px',
          fontSize: '9px',
          color: '#6b7280',
        }}
      >
        {session.toUpperCase()} SESSION • Confidence: {confidence}%
      </div>
    </div>
  )
}

export default MLOverlay
