import { useEffect, useRef } from 'react'

interface UseIntervalDropdownProps {
  isOpen: boolean
  onClose: () => void
}

/**
 * Hook for managing interval dropdown click-outside behavior
 */
export function useIntervalDropdown({ isOpen, onClose }: UseIntervalDropdownProps) {
  const listenerAttached = useRef(false)

  useEffect(() => {
    if (!isOpen || listenerAttached.current) return

    const handleClick = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null
      if (!target?.closest('.interval-dropdown-container')) {
        onClose()
      }
    }

    listenerAttached.current = true
    document.addEventListener('mousedown', handleClick)

    return () => {
      document.removeEventListener('mousedown', handleClick)
      listenerAttached.current = false
    }
  }, [isOpen, onClose])
}
