import React from 'react';
import styles from './TradingMetrics.module.css';

interface TradingMetricsProps {
  symbol: string;
  action: 'BUY NOW' | 'WAIT' | 'SELL NOW' | 'HOLD' | 'EXIT';
  price: number;
  change: number;
  changePercent: number;
  urgency: 'IMMEDIATE' | 'SOON' | 'WATCH' | 'LOW';
  timeToAction: string;
  timeRemaining?: number;
  exitTarget: number;
  sellPrice: number;
  stopLoss: number;
  estimatedTimeInTrade: string;
  confidence: number;
  signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell';
  isMobile?: boolean;
}

const TradingMetrics: React.FC<TradingMetricsProps> = ({
  symbol,
  action,
  price,
  change,
  changePercent,
  urgency,
  timeToAction,
  timeRemaining,
  exitTarget,
  sellPrice,
  stopLoss,
  estimatedTimeInTrade,
  confidence,
  signal,
  isMobile = false
}) => {
  const formatTimer = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getActionClasses = () => [
    styles.actionBadge,
    isMobile ? styles.actionBadgeMobile : styles.actionBadgeDesktop,
    styles[`action${action.replace(' ', '')}`]
  ].filter(Boolean).join(' ');

  const getUrgencyClasses = () => [
    styles.urgency,
    isMobile ? styles.urgencyMobile : styles.urgencyDesktop,
    styles[`urgency${urgency.charAt(0).toUpperCase() + urgency.slice(1).toLowerCase()}`]
  ].filter(Boolean).join(' ');

  const getConfidenceFillClass = () =>
    styles[`confidenceFill${signal.charAt(0).toUpperCase() + signal.slice(1).replace('_', '')}`];

  return (
    <>
      {/* Header Section */}
      <div className={isMobile ? styles.headerMobile : styles.headerDesktop}>
        <div className={styles.symbolSection}>
          <div className={isMobile ? styles.symbolRowMobile : styles.symbolRowDesktop}>
            <h2 className={isMobile ? styles.symbolMobile : styles.symbolDesktop}>
              {symbol}
            </h2>
            <div className={getActionClasses()}>
              {action}
            </div>
          </div>
          <div className={isMobile ? styles.priceMobile : styles.priceDesktop}>
            ${price.toFixed(2)}
          </div>
          <div className={`${isMobile ? styles.priceChangeMobile : styles.priceChangeDesktop} ${
            change >= 0 ? styles.priceChangePositive : styles.priceChangeNegative
          }`}>
            {change >= 0 ? '↑' : '↓'} {Math.abs(change).toFixed(2)} ({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%)
          </div>
        </div>

        <div className={styles.urgencySection}>
          <div className={getUrgencyClasses()}>
            {urgency}
          </div>
          {timeRemaining && timeRemaining > 0 && (
            <div className={isMobile ? styles.timerMobile : styles.timerDesktop}>
              <div className={styles.timerLabel}>TIME LEFT</div>
              <div className={isMobile ? styles.timerValueMobile : styles.timerValueDesktop}>
                {formatTimer(timeRemaining)}
              </div>
            </div>
          )}
          <div className={isMobile ? styles.timeToActionMobile : styles.timeToActionDesktop}>
            {timeToAction}
          </div>
        </div>
      </div>

      {/* Key Levels */}
      <div className={isMobile ? styles.keyLevelsMobile : styles.keyLevelsDesktop}>
        <div className={styles.keyLevel}>
          <div className={styles.keyLevelLabel}>TARGET</div>
          <div className={`${isMobile ? styles.keyLevelValueMobile : styles.keyLevelValueDesktop} ${styles.keyLevelTarget}`}>
            ${exitTarget.toFixed(2)}
          </div>
        </div>
        <div className={styles.keyLevel}>
          <div className={styles.keyLevelLabel}>SELL</div>
          <div className={`${isMobile ? styles.keyLevelValueMobile : styles.keyLevelValueDesktop} ${styles.keyLevelSell}`}>
            ${sellPrice.toFixed(2)}
          </div>
        </div>
        <div className={styles.keyLevel}>
          <div className={styles.keyLevelLabel}>STOP</div>
          <div className={`${isMobile ? styles.keyLevelValueMobile : styles.keyLevelValueDesktop} ${styles.keyLevelStop}`}>
            ${stopLoss.toFixed(2)}
          </div>
        </div>
        <div className={styles.keyLevel}>
          <div className={styles.keyLevelLabel}>TIME</div>
          <div className={`${isMobile ? styles.keyLevelValueMobile : styles.keyLevelValueDesktop} ${styles.keyLevelTime}`}>
            {estimatedTimeInTrade}
          </div>
        </div>
      </div>

      {/* Confidence Section */}
      <div className={styles.confidence}>
        <div className={styles.confidenceHeader}>
          <span className={isMobile ? styles.confidenceLabelMobile : styles.confidenceLabelDesktop}>
            AI Confidence
          </span>
          <span className={isMobile ? styles.confidenceValueMobile : styles.confidenceValueDesktop}>
            {confidence}%
          </span>
        </div>
        <div className={isMobile ? styles.confidenceBarMobile : styles.confidenceBarDesktop}>
          <div
            className={getConfidenceFillClass()}
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>
    </>
  );
};

export default TradingMetrics;