import React from 'react'
import styles from '../../../ProfessionalChart.module.css'

interface LoadingOverlayProps {
  show: boolean
  message?: string
}

/**
 * Loading indicator overlay for data fetching
 */
export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  show,
  message = 'Loading more data…'
}) => {
  if (!show) return null

  return (
    <div className={styles.loadingIndicator}>
      <div className={styles.loadingSpinner} />
      <span>{message}</span>
    </div>
  )
}
