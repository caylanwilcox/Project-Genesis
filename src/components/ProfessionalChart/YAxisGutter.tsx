'use client'

import React, { useState, useRef, useEffect } from 'react'

interface YAxisGutterProps {
  priceScale: number
  onPriceScaleChange: (scale: number) => void
}

export const YAxisGutter: React.FC<YAxisGutterProps> = ({ priceScale, onPriceScaleChange }) => {
  const [isDragging, setIsDragging] = useState(false)
  const [isHovered, setIsHovered] = useState(false)
  const gutterRef = useRef<HTMLDivElement>(null)
  const startYRef = useRef(0)
  const startScaleRef = useRef(1)

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
    startYRef.current = e.clientY
    startScaleRef.current = priceScale
  }

  const handleTouchStart = (e: React.TouchEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
    startYRef.current = e.touches[0].clientY
    startScaleRef.current = priceScale
  }

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    e.stopPropagation()

    // Use multiplicative zoom for smooth scaling
    const sensitivity = 0.0015
    const zoomDelta = -e.deltaY * sensitivity
    const zoomFactor = Math.exp(zoomDelta)
    const newScale = Math.max(0.1, Math.min(10, priceScale * zoomFactor))
    onPriceScaleChange(newScale)
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return

      const deltaY = startYRef.current - e.clientY
      const sensitivity = 0.005
      const zoomFactor = Math.exp(deltaY * sensitivity)
      const newScale = Math.max(0.1, Math.min(10, startScaleRef.current * zoomFactor))
      onPriceScaleChange(newScale)
    }

    const handleTouchMove = (e: TouchEvent) => {
      if (!isDragging) return

      const deltaY = startYRef.current - e.touches[0].clientY
      const sensitivity = 0.005
      const zoomFactor = Math.exp(deltaY * sensitivity)
      const newScale = Math.max(0.1, Math.min(10, startScaleRef.current * zoomFactor))
      onPriceScaleChange(newScale)
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
  }, [isDragging, onPriceScaleChange])

  return (
    <div
      ref={gutterRef}
      className={`absolute right-0 top-0 bottom-20 w-20 z-20 cursor-ns-resize transition-all duration-200 ${
        isDragging ? 'bg-blue-500/10' : isHovered ? 'bg-white/5' : 'bg-transparent'
      }`}
      onMouseDown={handleMouseDown}
      onTouchStart={handleTouchStart}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onWheel={handleWheel}
    >
      {/* Vertical guide lines */}
      <div className={`absolute left-2 top-1/2 -translate-y-1/2 h-32 w-px transition-opacity duration-200 ${
        isHovered || isDragging ? 'opacity-30 bg-white' : 'opacity-0'
      }`} />
      <div className={`absolute left-4 top-1/2 -translate-y-1/2 h-24 w-px transition-opacity duration-200 ${
        isHovered || isDragging ? 'opacity-20 bg-white' : 'opacity-0'
      }`} />

      {/* Scale indicator */}
      <div className={`absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-200 ${
        isHovered || isDragging ? 'opacity-100 scale-100' : 'opacity-0 scale-90'
      }`}>
        <div className="bg-gray-900/90 backdrop-blur-sm border border-gray-700/50 rounded-lg px-3 py-2 shadow-lg">
          <div className="text-[10px] text-gray-400 font-mono mb-1 text-center">Y-ZOOM</div>
          <div className="text-sm text-white font-mono font-semibold text-center">
            {priceScale.toFixed(2)}x
          </div>
          <div className="text-[9px] text-gray-500 mt-1 text-center">↕ drag or scroll</div>
        </div>
      </div>

      {/* Hover hint at edges */}
      {!isDragging && (
        <div className={`absolute left-0 top-1/2 -translate-y-1/2 w-1 h-16 rounded-r transition-opacity duration-200 ${
          isHovered ? 'opacity-40 bg-blue-400' : 'opacity-0'
        }`} />
      )}
    </div>
  )
}
