import { useState, useCallback, useEffect } from 'react'

export interface ViewportState {
  panOffset: number
  priceOffset: number
  customHeight: number | null
}

export interface ViewportActions {
  setPanOffset: (offset: number | ((prev: number) => number)) => void
  setPriceOffset: (offset: number) => void
  setCustomHeight: (height: number | null) => void
  resetOffsets: () => void
}

export function useChartViewport(dataLength: number) {
  const [panOffset, setPanOffset] = useState(0)
  const [priceOffset, setPriceOffset] = useState(0)
  const [customHeight, setCustomHeight] = useState<number | null>(null)
  const [prevDataLength, setPrevDataLength] = useState(dataLength)

  // Reset pan offset when data length changes significantly
  useEffect(() => {
    if (prevDataLength !== dataLength) {
      const lengthChange = Math.abs(dataLength - prevDataLength)
      const percentChange = prevDataLength > 0 ? lengthChange / prevDataLength : 1

      // If data length changed by more than 50%, reset to show most recent data
      if (percentChange > 0.5) {
        setPanOffset(0)
      }

      setPrevDataLength(dataLength)
    }
  }, [dataLength, prevDataLength])

  const resetOffsets = useCallback(() => {
    setPanOffset(0)
    setPriceOffset(0)
  }, [])

  const state: ViewportState = {
    panOffset,
    priceOffset,
    customHeight,
  }

  const actions: ViewportActions = {
    setPanOffset,
    setPriceOffset,
    setCustomHeight,
    resetOffsets,
  }

  return { state, actions }
}
