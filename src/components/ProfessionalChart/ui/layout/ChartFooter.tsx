import React from 'react'
import styles from '../../../ProfessionalChart.module.css'

interface ChartFooterProps {
  currentTime: Date
}

/**
 * Chart footer displaying current time in ET timezone
 */
export const ChartFooter: React.FC<ChartFooterProps> = ({ currentTime }) => {
  return (
    <div className={styles.bottomInfoBar}>
      <div className={styles.timeDisplay}>
        <span>
          {currentTime.toLocaleTimeString('en-US', {
            hour12: false,
            timeZone: 'America/New_York'
          })}{' '}
          (ET)
        </span>
        <span className={styles.timeSeparator}>|</span>
        <span>
          {currentTime.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            timeZone: 'America/New_York'
          })}
        </span>
      </div>
    </div>
  )
}
