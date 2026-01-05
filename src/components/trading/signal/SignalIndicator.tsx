import React from 'react';
import styles from './SignalIndicator.module.css';

interface SignalIndicatorProps {
  signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell';
  recommendation: string;
  isMobile?: boolean;
}

const SignalIndicator: React.FC<SignalIndicatorProps> = ({
  signal,
  recommendation,
  isMobile = false
}) => {
  const getSignalDescription = (signal: string) => {
    switch (signal) {
      case 'strong_buy': return '🔥 Maximum Opportunity';
      case 'buy': return '✓ Good Entry Point';
      case 'neutral': return '⏸ Wait for Confirmation';
      case 'sell': return '⚠️ Consider Exit';
      case 'strong_sell': return '🚨 Exit Immediately';
      default: return '';
    }
  };

  const badgeClasses = [
    styles.signalBadge,
    isMobile ? styles.signalBadgeMobile : styles.signalBadgeDesktop,
    styles[`signalBadge${signal.charAt(0).toUpperCase() + signal.slice(1).replace('_', '')}`]
  ].filter(Boolean).join(' ');

  const textClasses = [
    styles.signalText,
    isMobile ? styles.signalTextMobile : styles.signalTextDesktop,
    styles[`signalText${signal.charAt(0).toUpperCase() + signal.slice(1).replace('_', '')}`]
  ].filter(Boolean).join(' ');

  const descriptionClasses = isMobile
    ? styles.signalDescriptionMobile
    : styles.signalDescriptionDesktop;

  return (
    <div className={styles.signalContainer}>
      <div className={styles.signalWrapper}>
        <div className={badgeClasses}>
          <div className={textClasses}>
            {recommendation}
          </div>
          <div className={descriptionClasses}>
            {getSignalDescription(signal)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignalIndicator;