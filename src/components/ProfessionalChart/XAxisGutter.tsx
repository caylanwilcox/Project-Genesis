'use client'

import React, { useState, useRef, useEffect } from 'react'

interface XAxisGutterProps {
  timeScale: number
  onTimeScaleChange: (scale: number) => void
}

export const XAxisGutter: React.FC<XAxisGutterProps> = ({ timeScale, onTimeScaleChange }) => {
  const [isDragging, setIsDragging] = useState(false)
  const [isHovered, setIsHovered] = useState(false)
  const gutterRef = useRef<HTMLDivElement>(null)
  const startXRef = useRef(0)
  const startScaleRef = useRef(1)

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
    startXRef.current = e.clientX
    startScaleRef.current = timeScale
  }

  const handleTouchStart = (e: React.TouchEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
    startXRef.current = e.touches[0].clientX
    startScaleRef.current = timeScale
  }

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    e.stopPropagation()

    // Use multiplicative zoom for smooth scaling
    const delta = -e.deltaX || -e.deltaY
    const sensitivity = 0.0015
    const zoomFactor = Math.exp(delta * sensitivity)
    const newScale = Math.max(0.2, Math.min(5, timeScale * zoomFactor))
    onTimeScaleChange(newScale)
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return

      const deltaX = e.clientX - startXRef.current
      const sensitivity = 0.003
      const zoomFactor = Math.exp(deltaX * sensitivity)
      const newScale = Math.max(0.2, Math.min(5, startScaleRef.current * zoomFactor))
      onTimeScaleChange(newScale)
    }

    const handleTouchMove = (e: TouchEvent) => {
      if (!isDragging) return

      const deltaX = e.touches[0].clientX - startXRef.current
      const sensitivity = 0.003
      const zoomFactor = Math.exp(deltaX * sensitivity)
      const newScale = Math.max(0.2, Math.min(5, startScaleRef.current * zoomFactor))
      onTimeScaleChange(newScale)
    }

    const handleEnd = () => {
      setIsDragging(false)
    }

    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('mouseup', handleEnd)
      window.addEventListener('touchmove', handleTouchMove)
      window.addEventListener('touchend', handleEnd)
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleEnd)
      window.removeEventListener('touchmove', handleTouchMove)
      window.removeEventListener('touchend', handleEnd)
    }
  }, [isDragging, onTimeScaleChange])

  return (
    <div
      ref={gutterRef}
      className={`absolute bottom-0 left-0 h-24 z-20 cursor-ew-resize transition-all duration-200 ${
        isDragging ? 'bg-blue-500/10' : isHovered ? 'bg-white/5' : 'bg-transparent'
      }`}
      style={{ right: 'var(--chart-y-axis-gutter)' }}
      onMouseDown={handleMouseDown}
      onTouchStart={handleTouchStart}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onWheel={handleWheel}
    >
      {/* Horizontal guide lines */}
      <div className={`absolute top-2 left-1/2 -translate-x-1/2 w-32 h-px transition-opacity duration-200 ${
        isHovered || isDragging ? 'opacity-30 bg-white' : 'opacity-0'
      }`} />
      <div className={`absolute top-4 left-1/2 -translate-x-1/2 w-24 h-px transition-opacity duration-200 ${
        isHovered || isDragging ? 'opacity-20 bg-white' : 'opacity-0'
      }`} />

      {/* Scale indicator */}
      <div className={`absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-200 ${
        isHovered || isDragging ? 'opacity-100 scale-100' : 'opacity-0 scale-90'
      }`}>
        <div className="bg-gray-900/90 backdrop-blur-sm border border-gray-700/50 rounded-lg px-3 py-2 shadow-lg">
          <div className="text-[10px] text-gray-400 font-mono mb-1 text-center">X-ZOOM</div>
          <div className="text-sm text-white font-mono font-semibold text-center">
            {timeScale.toFixed(2)}x
          </div>
          <div className="text-[9px] text-gray-500 mt-1 text-center">↔ drag or scroll</div>
        </div>
      </div>

      {/* Hover hint at edges */}
      {!isDragging && (
        <div className={`absolute top-0 left-1/2 -translate-x-1/2 h-1 w-16 rounded-b transition-opacity duration-200 ${
          isHovered ? 'opacity-40 bg-blue-400' : 'opacity-0'
        }`} />
      )}
    </div>
  )
}
