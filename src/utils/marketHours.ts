/**
 * Market Hours Utility
 * Defines regular trading hours as 9:30 AM - 4:00 PM ET (Eastern Time)
 * US stock market operates in Eastern Time
 */

export interface MarketHoursConfig {
  openHour: number   // 9 (9am ET)
  openMinute: number // 30
  closeHour: number  // 16 (4pm ET in 24-hour format)
  closeMinute: number // 0
}

export const DEFAULT_MARKET_HOURS: MarketHoursConfig = {
  openHour: 9,      // 9:30 AM ET
  openMinute: 30,
  closeHour: 16,    // 4:00 PM ET
  closeMinute: 0,
}

/**
 * Get hour and minute in Eastern Time from UTC timestamp
 * @param timestamp Unix timestamp in milliseconds
 * @returns Hour and minute in ET
 */
function getEasternHourMinute(timestamp: number): { hour: number; minute: number } {
  const date = new Date(timestamp)
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false
  }).formatToParts(date)

  let hour = 0
  let minute = 0
  for (const p of parts) {
    if (p.type === 'hour') hour = parseInt(p.value, 10)
    if (p.type === 'minute') minute = parseInt(p.value, 10)
  }
  return { hour, minute }
}

/**
 * Check if a timestamp falls within regular market hours (9:30am - 4:00pm ET)
 * @param timestamp Unix timestamp in milliseconds
 * @param config Market hours configuration
 * @returns true if timestamp is during market hours
 */
export function isMarketHours(
  timestamp: number,
  config: MarketHoursConfig = DEFAULT_MARKET_HOURS
): boolean {
  const { hour, minute } = getEasternHourMinute(timestamp)

  // Convert time to minutes since midnight for comparison
  const currentMins = hour * 60 + minute
  const openMins = config.openHour * 60 + (config.openMinute ?? 0)
  const closeMins = config.closeHour * 60 + (config.closeMinute ?? 0)

  // Market is open from 9:30am (inclusive) to 4:00pm (exclusive) ET
  return currentMins >= openMins && currentMins < closeMins
}

/**
 * Filter candle data to only include market hours
 * @param data Array of candle data
 * @param config Market hours configuration
 * @returns Filtered array containing only market hours data
 */
export function filterMarketHoursData<T extends { time: number }>(
  data: T[],
  config: MarketHoursConfig = DEFAULT_MARKET_HOURS
): T[] {
  return data.filter(candle => isMarketHours(candle.time, config))
}

/**
 * Get market hours segments from visible data for rendering
 * @param data Array of candle data with time field
 * @param config Market hours configuration
 * @returns Array of segments with start/end indices and market open status
 */
export function getMarketHoursSegments(
  data: { time: number }[],
  config: MarketHoursConfig = DEFAULT_MARKET_HOURS
): Array<{ startIndex: number; endIndex: number; isOpen: boolean }> {
  if (data.length === 0) return []

  const segments: Array<{ startIndex: number; endIndex: number; isOpen: boolean }> = []
  let currentSegmentStart = 0
  let currentIsOpen = isMarketHours(data[0].time, config)

  for (let i = 1; i < data.length; i++) {
    const thisIsOpen = isMarketHours(data[i].time, config)

    // If market status changed, close current segment and start new one
    if (thisIsOpen !== currentIsOpen) {
      segments.push({
        startIndex: currentSegmentStart,
        endIndex: i - 1,
        isOpen: currentIsOpen
      })
      currentSegmentStart = i
      currentIsOpen = thisIsOpen
    }
  }

  // Close the final segment
  segments.push({
    startIndex: currentSegmentStart,
    endIndex: data.length - 1,
    isOpen: currentIsOpen
  })

  return segments
}

/**
 * Duration-aware segmentation: classify a bar as "open" if ANY overlap
 * between [barStart, barEnd) and [open, close) in ET exists.
 * This fixes 5D (1h bars) where bars like 9:00–10:00 should count as open.
 */
export function getMarketHoursSegmentsWithDuration(
  data: { time: number }[],
  barDurationMs: number,
  config: MarketHoursConfig = DEFAULT_MARKET_HOURS
): Array<{ startIndex: number; endIndex: number; isOpen: boolean }> {
  if (data.length === 0) return []

  // Helper: ET Y-M-D and hh:mm for a timestamp
  const getEtDateParts = (timestamp: number) => {
    const date = new Date(timestamp)
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    }).formatToParts(date)
    let year = 0, month = 0, day = 0, hour = 0, minute = 0
    for (const p of parts) {
      if (p.type === 'year') year = parseInt(p.value, 10)
      if (p.type === 'month') month = parseInt(p.value, 10)
      if (p.type === 'day') day = parseInt(p.value, 10)
      if (p.type === 'hour') hour = parseInt(p.value, 10)
      if (p.type === 'minute') minute = parseInt(p.value, 10)
    }
    const ymd = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`
    return { ymd, hour, minute }
  }

  const openMins = (config.openHour * 60) + (config.openMinute ?? 0) // 9:30 => 570
  const closeMins = (config.closeHour * 60) + (config.closeMinute ?? 0) // 16:00 => 960

  const barIsOpen = (startMs: number, durationMs: number) => {
    const endMs = startMs + durationMs
    const s = getEtDateParts(startMs)
    const e = getEtDateParts(endMs)
    const sMins = s.hour * 60 + s.minute
    const eMins = e.hour * 60 + e.minute

    if (s.ymd === e.ymd) {
      // Same ET day: check interval overlap
      const intervalStart = sMins
      const intervalEnd = eMins
      const overlap = Math.max(0, Math.min(intervalEnd, closeMins) - Math.max(intervalStart, openMins))
      return overlap > 0
    }

    // Crosses ET midnight: check overlap on start day and end day
    const overlapStartDay = Math.max(0, Math.min(24 * 60, closeMins) - Math.max(sMins, openMins))
    const overlapEndDay = Math.max(0, Math.min(eMins, closeMins) - Math.max(0, openMins))
    return (overlapStartDay + overlapEndDay) > 0
  }

  const segments: Array<{ startIndex: number; endIndex: number; isOpen: boolean }> = []
  let currentSegmentStart = 0
  let currentIsOpen = barIsOpen(data[0].time, barDurationMs)

  for (let i = 1; i < data.length; i++) {
    const thisIsOpen = barIsOpen(data[i].time, barDurationMs)
    if (thisIsOpen !== currentIsOpen) {
      segments.push({
        startIndex: currentSegmentStart,
        endIndex: i - 1,
        isOpen: currentIsOpen
      })
      currentSegmentStart = i
      currentIsOpen = thisIsOpen
    }
  }

  segments.push({
    startIndex: currentSegmentStart,
    endIndex: data.length - 1,
    isOpen: currentIsOpen
  })

  return segments
}

/**
 * Format market hours for display
 */
export function formatMarketHours(config: MarketHoursConfig = DEFAULT_MARKET_HOURS): string {
  const fmt = (h: number, m: number) => {
    const suffix = h < 12 ? 'am' : 'pm'
    const hour12 = h === 0 ? 12 : (h > 12 ? h - 12 : h)
    const minuteStr = m.toString().padStart(2, '0')
    return `${hour12}:${minuteStr}${suffix}`
  }

  const openTime = fmt(config.openHour, config.openMinute ?? 0)
  const closeTime = fmt(config.closeHour, config.closeMinute ?? 0)

  return `${openTime} - ${closeTime} ET`
}
