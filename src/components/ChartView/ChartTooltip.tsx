import React from 'react'

interface ChartTooltipProps {
  active?: boolean
  payload?: any[]
  label?: string
}

export const ChartTooltip: React.FC<ChartTooltipProps> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null

  const data = payload[0]?.payload
  if (!data) return null

  return (
    <div className="custom-tooltip">
      <p className="label">{`Date: ${label}`}</p>
      <p className="value">{`Open: $${data.open?.toFixed(2)}`}</p>
      <p className="value">{`High: $${data.high?.toFixed(2)}`}</p>
      <p className="value">{`Low: $${data.low?.toFixed(2)}`}</p>
      <p className="value">{`Close: $${data.close?.toFixed(2)}`}</p>
      <p className="value">{`Volume: ${data.volume?.toLocaleString()}`}</p>
      {data.signals > 0 && (
        <p className="signals">{`Signals: ${data.signals}`}</p>
      )}
    </div>
  )
}
