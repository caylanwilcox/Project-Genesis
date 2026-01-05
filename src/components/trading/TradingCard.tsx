import React from 'react';
import { useRouter } from 'next/navigation';
import styles from './TradingCard.module.css';
import SignalIndicator from './signal/SignalIndicator';
import TradingMetrics from './metrics/TradingMetrics';

interface TickerData {
  symbol: string;
  signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell';
  action: 'BUY NOW' | 'WAIT' | 'SELL NOW' | 'HOLD' | 'EXIT';
  urgency: 'IMMEDIATE' | 'SOON' | 'WATCH' | 'LOW';
  timeToAction: string;
  confidence: number;
  recommendation: string;
  price: number;
  change: number;
  changePercent: number;
  exitTarget: number;
  stopLoss: number;
  sellPrice: number;
  estimatedTimeInTrade: string;
}

interface TradingCardProps {
  ticker: TickerData;
  timeRemaining?: number;
  isMobile?: boolean;
}

const TradingCard: React.FC<TradingCardProps> = ({
  ticker,
  timeRemaining,
  isMobile = false
}) => {
  const router = useRouter();

  const getBackgroundGradientClass = () => {
    const signalClass = ticker.signal.charAt(0).toUpperCase() +
                      ticker.signal.slice(1).replace('_', '');
    return styles[`backgroundGradient${signalClass}`];
  };

  const cardClasses = isMobile ? styles.tradingCardMobile : styles.tradingCardDesktop;
  const contentClasses = isMobile ? styles.cardContentMobile : styles.cardContentDesktop;

  return (
    <div
      onClick={() => router.push(`/ticker/${ticker.symbol}`)}
      className={cardClasses}
    >
      {/* Animated Background */}
      <div className={getBackgroundGradientClass()} />

      <div className={contentClasses}>
        <div className={styles.contentWrapper}>
          {/* Top Section - Metrics */}
          <div style={{ flexShrink: 0 }}>
            <TradingMetrics
              symbol={ticker.symbol}
              action={ticker.action}
              price={ticker.price}
              change={ticker.change}
              changePercent={ticker.changePercent}
              urgency={ticker.urgency}
              timeToAction={ticker.timeToAction}
              timeRemaining={timeRemaining}
              exitTarget={ticker.exitTarget}
              sellPrice={ticker.sellPrice}
              stopLoss={ticker.stopLoss}
              estimatedTimeInTrade={ticker.estimatedTimeInTrade}
              confidence={ticker.confidence}
              signal={ticker.signal}
              isMobile={isMobile}
            />
          </div>

          {/* Center Section - Signal Indicator */}
          <SignalIndicator
            signal={ticker.signal}
            recommendation={ticker.recommendation}
            isMobile={isMobile}
          />
        </div>

        {/* Hover Overlay for Desktop */}
        {!isMobile && (
          <div className={styles.hoverOverlay}>
            <div className={styles.hoverContent}>
              <div className={styles.hoverTitle}>
                View Full Analysis & Execute Trade →
              </div>
              <div className={styles.hoverSubtitle}>
                Detailed charts, order execution, and risk management
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradingCard;