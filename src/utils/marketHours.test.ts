import { isMarketHours, getMarketHoursSegments, formatMarketHours } from './marketHours'

describe('marketHours utilities (9:30am–4:00pm ET)', () => {
  // Use a winter date to avoid DST complexity (EST = UTC-5)
  // Jan 15, 2025 (Wednesday)
  const year = 2025
  const monthZeroBased = 0 // January
  const day = 15

  // Helper to create a UTC timestamp corresponding to an ET clock time.
  // For this specific date (January), ET is UTC-5.
  const etToUtc = (etHour: number, etMinute: number) =>
    Date.UTC(year, monthZeroBased, day, etHour + 5, etMinute, 0, 0)

  it('formatMarketHours returns 9:30am - 4:00pm ET', () => {
    expect(formatMarketHours()).toBe('9:30am - 4:00pm ET')
  })

  it('isMarketHours is false before 9:30 ET (e.g., 6:45 ET)', () => {
    const ts = etToUtc(6, 45) // 06:45 ET => 11:45 UTC
    expect(isMarketHours(ts)).toBe(false)
  })

  it('isMarketHours is false at 9:29 ET and true at 9:30 ET', () => {
    const preOpen = etToUtc(9, 29)
    const open = etToUtc(9, 30)
    expect(isMarketHours(preOpen)).toBe(false)
    expect(isMarketHours(open)).toBe(true)
  })

  it('isMarketHours is true during regular session and false at/after 16:00 ET', () => {
    const midday = etToUtc(12, 0)     // 12:00 ET
    const lastMinute = etToUtc(15, 59) // 15:59 ET
    const close = etToUtc(16, 0)      // 16:00 ET

    expect(isMarketHours(midday)).toBe(true)
    expect(isMarketHours(lastMinute)).toBe(true)
    expect(isMarketHours(close)).toBe(false) // exclusive end
  })

  it('getMarketHoursSegments splits pre-open, open, and post-open correctly for intraday minutes', () => {
    // Build 1-minute candles from 08:00 ET to 16:59 ET inclusive
    // 08:00 ET => 13:00 UTC, 16:59 ET => 21:59 UTC
    const startUtc = etToUtc(8, 0)
    const minutes = 9 * 60 // 9 hours of minutes (08:00..16:59) = 540

    const data = Array.from({ length: minutes }, (_, i) => ({
      time: startUtc + i * 60_000
    }))

    const segments = getMarketHoursSegments(data)

    // Expect three segments: pre-open (closed), regular (open), after-open (closed)
    expect(segments.length).toBe(3)
    expect(segments[0].isOpen).toBe(false)
    expect(segments[1].isOpen).toBe(true)
    expect(segments[2].isOpen).toBe(false)

    // Pre-open: 08:00..09:29 => 90 minutes → indices 0..89
    expect(segments[0].startIndex).toBe(0)
    expect(segments[0].endIndex).toBe(89)

    // Open: 09:30..15:59 => 390 minutes → indices 90..479
    expect(segments[1].startIndex).toBe(90)
    expect(segments[1].endIndex).toBe(479)

    // After-open: 16:00..16:59 => 60 minutes → indices 480..539
    expect(segments[2].startIndex).toBe(480)
    expect(segments[2].endIndex).toBe(539)
  })
})

