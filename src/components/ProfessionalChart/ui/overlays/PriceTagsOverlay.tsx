import React from 'react'
import tagStyles from '../../../ChartYAxisTags.module.css'
import type { PriceTag } from '../../priceLines'

interface PriceTagsOverlayProps {
  tags: PriceTag[]
}

/**
 * Renders price tag overlays on the Y-axis
 * Displays stop loss, entry, targets, and current price
 */
export const PriceTagsOverlay: React.FC<PriceTagsOverlayProps> = ({ tags }) => {
  return (
    <div className={tagStyles.yAxisTagsContainer} aria-hidden>
      {tags.map((tag, idx) => (
        <div
          key={`${tag.label}-${idx}`}
          className={`${tagStyles.yAxisTag} ${
            tag.kind === 'current'
              ? tagStyles['yAxisTag--current']
              : tag.kind === 'entry'
              ? tagStyles['yAxisTag--entry']
              : tag.kind === 'stop'
              ? tagStyles['yAxisTag--stop']
              : tagStyles['yAxisTag--target']
          }`}
          style={{ top: `${tag.y - 11}px` }}
        >
          <span className={tagStyles.yAxisTagLabel}>{tag.label}</span>
        </div>
      ))}
    </div>
  )
}
