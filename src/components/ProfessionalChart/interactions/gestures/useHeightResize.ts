import { useCallback } from 'react'

interface UseHeightResizeProps {
  containerRef: React.RefObject<HTMLDivElement>
  onHeightChange: (height: number) => void
}

/**
 * Hook for handling chart height resize drag interaction
 */
export function useHeightResize({ containerRef, onHeightChange }: UseHeightResizeProps) {
  const handleResizeStart = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      event.preventDefault()
      const startY = event.clientY
      const startHeight = containerRef.current?.getBoundingClientRect().height ?? 0

      const handleMove = (moveEvent: MouseEvent) => {
        const delta = moveEvent.clientY - startY
        onHeightChange(Math.max(320, startHeight + delta))
      }

      const handleUp = () => {
        document.removeEventListener('mousemove', handleMove)
        document.removeEventListener('mouseup', handleUp)
      }

      document.addEventListener('mousemove', handleMove)
      document.addEventListener('mouseup', handleUp)
    },
    [containerRef, onHeightChange]
  )

  return { handleResizeStart }
}
