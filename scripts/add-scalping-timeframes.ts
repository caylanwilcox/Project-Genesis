/**
 * Add Scalping Timeframes (1m, 5m)
 *
 * Adds 1-minute and 5-minute data for scalping mode (Week 2)
 *
 * Data Plan:
 * - 1m: Last 7 days (~2,730 bars per ticker)
 * - 5m: Last 30 days (~2,340 bars per ticker)
 *
 * Total: ~18,680 bars across 4 tickers
 */

import { DataIngestionServiceV2 } from '@/services/dataIngestionService.v2'
import { marketDataRepo } from '@/repositories'

interface ScalpingConfig {
  tickers: string[]
  timeframes: Array<{
    tf: '1m' | '5m'
    days: number
    description: string
  }>
}

async function addScalpingTimeframes(config: ScalpingConfig) {
  console.log('======================================')
  console.log('Adding Scalping Timeframes (1m, 5m)')
  console.log('======================================\n')

  const ingestionService = new DataIngestionServiceV2()
  const results: any[] = []
  let totalBars = 0

  for (const ticker of config.tickers) {
    console.log(`\n📈 Processing ${ticker}`)
    console.log('-----------------------------------')

    for (const { tf, days, description } of config.timeframes) {
      console.log(`\n⏱️  ${description}`)

      try {
        const startTime = Date.now()
        const result = await ingestionService.ingestHistoricalData(ticker, tf, days)

        if (result.success) {
          console.log(`✅ SUCCESS`)
          console.log(`   Bars inserted: ${result.barsInserted}`)
          console.log(`   Bars skipped: ${result.barsSkipped}`)
          console.log(`   Duration: ${(result.durationMs / 1000).toFixed(1)}s`)

          totalBars += result.barsInserted

          results.push({
            ticker,
            timeframe: tf,
            success: true,
            barsInserted: result.barsInserted,
            barsSkipped: result.barsSkipped,
            durationMs: result.durationMs,
          })
        } else {
          console.log(`❌ FAILED: ${result.error}`)
          results.push({
            ticker,
            timeframe: tf,
            success: false,
            error: result.error,
          })
        }

        // Rate limiting (Polygon.io free tier: 5 calls/min)
        console.log(`   ⏳ Waiting 13s for rate limit...`)
        await new Promise(resolve => setTimeout(resolve, 13000))

      } catch (error: any) {
        console.log(`❌ ERROR: ${error.message}`)
        results.push({
          ticker,
          timeframe: tf,
          success: false,
          error: error.message,
        })
      }
    }
  }

  // Summary
  console.log('\n======================================')
  console.log('Summary')
  console.log('======================================\n')

  const successful = results.filter(r => r.success).length
  const failed = results.filter(r => !r.success).length

  console.log(`Total jobs: ${results.length}`)
  console.log(`✅ Successful: ${successful}`)
  console.log(`❌ Failed: ${failed}`)
  console.log(`📊 Total bars inserted: ${totalBars}\n`)

  // Detailed results
  console.log('Detailed Results:')
  console.log('-----------------------------------')
  results.forEach(r => {
    const status = r.success ? '✅' : '❌'
    const bars = r.success ? `${r.barsInserted} bars` : r.error
    console.log(`${status} ${r.ticker} ${r.timeframe}: ${bars}`)
  })

  // Database verification
  console.log('\n======================================')
  console.log('Database Verification')
  console.log('======================================\n')

  for (const ticker of config.tickers) {
    for (const { tf } of config.timeframes) {
      const summary = await marketDataRepo.getSummary(ticker, tf)
      if (summary) {
        console.log(`${ticker} ${tf}: ${summary.bars} bars (${summary.earliest} → ${summary.latest})`)
      }
    }
  }

  console.log('\n======================================')
  console.log('Scalping Timeframes Added!')
  console.log('======================================')

  return results
}

// Configuration
const config: ScalpingConfig = {
  tickers: ['SPY', 'QQQ', 'IWM', 'UVXY'],
  timeframes: [
    {
      tf: '1m',
      days: 7,
      description: '1-minute candles (7 days)',
    },
    {
      tf: '5m',
      days: 30,
      description: '5-minute candles (30 days)',
    },
  ],
}

// Run the script
addScalpingTimeframes(config)
  .then(() => {
    console.log('\n✅ Script completed successfully!')
    process.exit(0)
  })
  .catch(error => {
    console.error('\n❌ Script failed:', error.message)
    process.exit(1)
  })
