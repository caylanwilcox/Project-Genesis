import React from 'react';
import styles from './TradingGrid.module.css';

interface TradingGridProps {
  children: React.ReactNode;
  isMobile: boolean;
}

const TradingGrid: React.FC<TradingGridProps> = ({ children, isMobile }) => {
  const containerClass = isMobile ? styles.containerMobile : styles.containerDesktop;
  const gridClass = isMobile ? styles.mobileGrid : styles.desktopGrid;

  return (
    <div className={containerClass}>
      <div className={gridClass}>
        {React.Children.map(children, (child) => (
          <div className={styles.gridItem}>
            {child}
          </div>
        ))}
      </div>
    </div>
  );
};

export default TradingGrid;