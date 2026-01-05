import React from 'react'
import tagStyles from '../../../ChartYAxisTags.module.css'
import type { PriceTag } from '../../priceLines'

interface PriceTagsOverlayProps {
  tags: PriceTag[]
}

/**
 * Renders price tag overlays on the Y-axis
 * Displays stop loss, entry, targets, current price, and range levels
 */
export const PriceTagsOverlay: React.FC<PriceTagsOverlayProps> = ({ tags }) => {
  // Filter out target tags to avoid duplication
  const filteredTags = tags.filter(tag => tag.kind !== 'target')

  const getTagClass = (kind: string) => {
    switch (kind) {
      case 'current':
        return tagStyles['yAxisTag--current']
      case 'entry':
        return tagStyles['yAxisTag--entry']
      case 'stop':
        return tagStyles['yAxisTag--stop']
      case 'rangeHigh':
      case 'rangeLow':
      case 'rangeMid':
        return tagStyles['yAxisTag--range'] || tagStyles['yAxisTag--target']
      case 'retestHigh':
      case 'retestLow':
        return tagStyles['yAxisTag--retest'] || tagStyles['yAxisTag--entry']
      default:
        return tagStyles['yAxisTag--target']
    }
  }

  return (
    <div className={tagStyles.yAxisTagsContainer} aria-hidden>
      {filteredTags.map((tag, idx) => (
        <div
          key={`${tag.label}-${idx}`}
          className={`${tagStyles.yAxisTag} ${getTagClass(tag.kind)}`}
          style={{ top: `${tag.y - 11}px` }}
        >
          <span className={tagStyles.yAxisTagLabel}>{tag.label}</span>
        </div>
      ))}
    </div>
  )
}
