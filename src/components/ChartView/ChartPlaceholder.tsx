import React from 'react'

interface ChartPlaceholderProps {
  message: string
}

export const ChartPlaceholder: React.FC<ChartPlaceholderProps> = ({ message }) => {
  return (
    <div className="chart-placeholder">
      <p>{message}</p>
    </div>
  )
}
