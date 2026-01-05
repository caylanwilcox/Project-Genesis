import { useCallback, useEffect } from 'react'

interface InteractionHandlers {
  handleMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void
  handleMouseMove: (e: React.MouseEvent<HTMLDivElement>, rect: DOMRect) => void
  handleMouseUp: () => void
  handleMouseLeave: () => void
  handleTouchStart: (e: React.TouchEvent<HTMLDivElement>) => void
  handleTouchMove: (e: React.TouchEvent<HTMLDivElement>, rect: DOMRect) => void
  handleTouchEnd: () => void
  handleWheel: (e: React.WheelEvent<HTMLDivElement>, rect: DOMRect) => void
}

interface ChartInteractionHandlersProps {
  chartAreaRef: React.RefObject<HTMLDivElement>
  interaction: InteractionHandlers
}

/**
 * Hook for wrapping mouse/touch interaction handlers with rect calculations
 */
export function useChartInteractionHandlers({
  chartAreaRef,
  interaction,
}: ChartInteractionHandlersProps) {
  useEffect(() => {
    const element = chartAreaRef.current
    if (!element) return

    const handleNativeWheel = (event: WheelEvent) => {
      // Block default scrolling so the wheel gesture only affects the chart zoom
      event.preventDefault()
      event.stopPropagation()

      // Call the interaction handler directly with native event
      const rect = element.getBoundingClientRect()
      if (rect) {
        // Create a synthetic event-like object for the handler
        interaction.handleWheel(event as unknown as React.WheelEvent<HTMLDivElement>, rect)
      }
    }

    element.addEventListener('wheel', handleNativeWheel, { passive: false })

    return () => {
      element.removeEventListener('wheel', handleNativeWheel)
    }
  }, [chartAreaRef, interaction])

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      const rect = chartAreaRef.current?.getBoundingClientRect()
      if (!rect) return
      interaction.handleMouseMove(event, rect)
    },
    [chartAreaRef, interaction.handleMouseMove]
  )

  const handleTouchMove = useCallback(
    (event: React.TouchEvent<HTMLDivElement>) => {
      const rect = chartAreaRef.current?.getBoundingClientRect()
      if (!rect) return
      interaction.handleTouchMove(event, rect)
    },
    [chartAreaRef, interaction.handleTouchMove]
  )

  const handleWheel = useCallback(
    (event: React.WheelEvent<HTMLDivElement>) => {
      const rect = chartAreaRef.current?.getBoundingClientRect()
      if (!rect) {
        event.preventDefault()
        event.stopPropagation()
        return
      }
      interaction.handleWheel(event, rect)
    },
    [chartAreaRef, interaction.handleWheel]
  )

  return {
    handleMouseMove,
    handleTouchMove,
    handleMouseDown: interaction.handleMouseDown,
    handleMouseUp: interaction.handleMouseUp,
    handleMouseLeave: interaction.handleMouseLeave,
    handleTouchStart: interaction.handleTouchStart,
    handleTouchEnd: interaction.handleTouchEnd,
    handleWheel,
  }
}
