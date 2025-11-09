/**
 * Historical Data Backfill Script
 *
 * Fetches historical market data from Polygon.io and stores in database
 * Splits data into TRAINING and TESTING sets for ML model validation
 *
 * TRAINING SET: First 70% of historical data (for model training)
 * TESTING SET:  Last 30% of historical data (for model validation - never seen during training)
 *
 * Usage:
 *   npx ts-node scripts/backfill-historical-data.ts
 */

import { dataIngestionServiceV2 } from '@/services/dataIngestionService.v2'
import { marketDataRepo, ingestionLogRepo } from '@/repositories'
import { Timeframe } from '@/types/polygon'

interface BackfillConfig {
  tickers: string[]
  timeframes: Timeframe[]
  yearsBack: number
  trainTestSplit: number  // 0.7 = 70% training, 30% testing
}

interface BackfillStats {
  ticker: string
  timeframe: string
  totalBars: number
  trainingBars: number
  testingBars: number
  trainingDateRange: { start: Date; end: Date }
  testingDateRange: { start: Date; end: Date }
  durationMs: number
  success: boolean
  error?: string
}

/**
 * Main backfill function with train/test split
 */
async function backfillHistoricalData(config: BackfillConfig): Promise<BackfillStats[]> {
  const { tickers, timeframes, yearsBack, trainTestSplit } = config
  const stats: BackfillStats[] = []

  console.log('\n╔════════════════════════════════════════════════════════════════╗')
  console.log('║          HISTORICAL DATA BACKFILL WITH TRAIN/TEST SPLIT        ║')
  console.log('╚════════════════════════════════════════════════════════════════╝\n')
  console.log(`📅 Date Range: ${yearsBack} years back from today`)
  console.log(`🎯 Tickers: ${tickers.join(', ')}`)
  console.log(`⏱️  Timeframes: ${timeframes.join(', ')}`)
  console.log(`📊 Train/Test Split: ${trainTestSplit * 100}% / ${(1 - trainTestSplit) * 100}%`)
  console.log(`\n${'='.repeat(70)}\n`)

  for (const ticker of tickers) {
    for (const timeframe of timeframes) {
      const startTime = Date.now()

      try {
        console.log(`\n📈 [${ticker} ${timeframe}] Starting backfill...`)

        // Calculate date range
        const endDate = new Date()
        const startDate = new Date()
        startDate.setFullYear(startDate.getFullYear() - yearsBack)

        console.log(`   📅 Range: ${startDate.toISOString().split('T')[0]} → ${endDate.toISOString().split('T')[0]}`)

        // Calculate split point (70% for training, 30% for testing)
        const totalDaysSpan = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24))
        const trainDaysSpan = Math.floor(totalDaysSpan * trainTestSplit)

        const trainEndDate = new Date(startDate)
        trainEndDate.setDate(trainEndDate.getDate() + trainDaysSpan)

        const testStartDate = new Date(trainEndDate)
        testStartDate.setDate(testStartDate.getDate() + 1) // Next day after training

        console.log(`\n   🎓 TRAINING SET:`)
        console.log(`      ${startDate.toISOString().split('T')[0]} → ${trainEndDate.toISOString().split('T')[0]}`)
        console.log(`   🧪 TESTING SET:`)
        console.log(`      ${testStartDate.toISOString().split('T')[0]} → ${endDate.toISOString().split('T')[0]}`)

        // Fetch data from Polygon.io
        const daysToFetch = Math.ceil(totalDaysSpan)
        console.log(`\n   🔄 Fetching data...`)

        const result = await dataIngestionServiceV2.ingestHistoricalData(
          ticker,
          timeframe,
          daysToFetch
        )

        if (!result.success) {
          throw new Error(result.error || 'Unknown error')
        }

        // Get all data we just ingested
        const allData = await marketDataRepo.findMany(
          {
            ticker,
            timeframe,
            startDate,
            endDate
          },
          100000 // Large limit to get all data
        )

        // Sort by timestamp
        allData.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())

        // Split into training and testing sets
        const trainingData = allData.filter(bar => bar.timestamp <= trainEndDate)
        const testingData = allData.filter(bar => bar.timestamp >= testStartDate)

        const durationMs = Date.now() - startTime

        // Log statistics
        console.log(`\n   ✅ SUCCESS!`)
        console.log(`   📊 Total bars: ${allData.length}`)
        console.log(`   🎓 Training bars: ${trainingData.length} (${((trainingData.length / allData.length) * 100).toFixed(1)}%)`)
        console.log(`   🧪 Testing bars: ${testingData.length} (${((testingData.length / allData.length) * 100).toFixed(1)}%)`)
        console.log(`   ⏱️  Duration: ${(durationMs / 1000).toFixed(1)}s`)

        stats.push({
          ticker,
          timeframe,
          totalBars: allData.length,
          trainingBars: trainingData.length,
          testingBars: testingData.length,
          trainingDateRange: {
            start: trainingData[0]?.timestamp || startDate,
            end: trainingData[trainingData.length - 1]?.timestamp || trainEndDate,
          },
          testingDateRange: {
            start: testingData[0]?.timestamp || testStartDate,
            end: testingData[testingData.length - 1]?.timestamp || endDate,
          },
          durationMs,
          success: true,
        })

        // Wait between requests to respect rate limits
        if (tickers.indexOf(ticker) < tickers.length - 1 || timeframes.indexOf(timeframe) < timeframes.length - 1) {
          console.log(`\n   ⏳ Waiting 13s for rate limit...`)
          await new Promise(resolve => setTimeout(resolve, 13000))
        }

      } catch (error: any) {
        const durationMs = Date.now() - startTime
        console.error(`\n   ❌ FAILED: ${error.message}`)

        stats.push({
          ticker,
          timeframe,
          totalBars: 0,
          trainingBars: 0,
          testingBars: 0,
          trainingDateRange: { start: new Date(), end: new Date() },
          testingDateRange: { start: new Date(), end: new Date() },
          durationMs,
          success: false,
          error: error.message,
        })
      }
    }
  }

  return stats
}

/**
 * Print summary report
 */
function printSummary(stats: BackfillStats[]) {
  console.log('\n\n╔════════════════════════════════════════════════════════════════╗')
  console.log('║                    BACKFILL SUMMARY REPORT                     ║')
  console.log('╚════════════════════════════════════════════════════════════════╝\n')

  const successful = stats.filter(s => s.success)
  const failed = stats.filter(s => !s.success)

  console.log(`📊 Total Jobs: ${stats.length}`)
  console.log(`✅ Successful: ${successful.length}`)
  console.log(`❌ Failed: ${failed.length}\n`)

  if (successful.length > 0) {
    console.log('─'.repeat(70))
    console.log('SUCCESSFUL INGESTIONS:\n')

    successful.forEach(stat => {
      console.log(`📈 ${stat.ticker} ${stat.timeframe}`)
      console.log(`   Total Bars: ${stat.totalBars.toLocaleString()}`)
      console.log(`   🎓 Training: ${stat.trainingBars.toLocaleString()} bars`)
      console.log(`      ${stat.trainingDateRange.start.toISOString().split('T')[0]} → ${stat.trainingDateRange.end.toISOString().split('T')[0]}`)
      console.log(`   🧪 Testing:  ${stat.testingBars.toLocaleString()} bars`)
      console.log(`      ${stat.testingDateRange.start.toISOString().split('T')[0]} → ${stat.testingDateRange.end.toISOString().split('T')[0]}`)
      console.log(`   ⏱️  Time: ${(stat.durationMs / 1000).toFixed(1)}s\n`)
    })
  }

  if (failed.length > 0) {
    console.log('─'.repeat(70))
    console.log('FAILED INGESTIONS:\n')

    failed.forEach(stat => {
      console.log(`❌ ${stat.ticker} ${stat.timeframe}`)
      console.log(`   Error: ${stat.error}\n`)
    })
  }

  // Grand totals
  const totalBars = successful.reduce((sum, s) => sum + s.totalBars, 0)
  const totalTraining = successful.reduce((sum, s) => sum + s.trainingBars, 0)
  const totalTesting = successful.reduce((sum, s) => sum + s.testingBars, 0)
  const totalDuration = stats.reduce((sum, s) => sum + s.durationMs, 0)

  console.log('─'.repeat(70))
  console.log('GRAND TOTALS:\n')
  console.log(`📊 Total Bars Ingested: ${totalBars.toLocaleString()}`)
  console.log(`🎓 Training Set: ${totalTraining.toLocaleString()} bars (${((totalTraining / totalBars) * 100).toFixed(1)}%)`)
  console.log(`🧪 Testing Set:  ${totalTesting.toLocaleString()} bars (${((totalTesting / totalBars) * 100).toFixed(1)}%)`)
  console.log(`⏱️  Total Time: ${(totalDuration / 1000 / 60).toFixed(1)} minutes`)
  console.log(`\n${'='.repeat(70)}\n`)

  // Data split validation
  console.log('✅ DATA SPLIT VALIDATION:\n')
  console.log('   Training set = PAST data (for model learning)')
  console.log('   Testing set = RECENT data (model has never seen this)')
  console.log('   This ensures realistic backtesting and prevents data leakage!\n')
}

/**
 * Save summary to file
 */
async function saveSummary(stats: BackfillStats[]) {
  const summary = {
    timestamp: new Date().toISOString(),
    stats,
    totals: {
      totalBars: stats.reduce((sum, s) => sum + s.totalBars, 0),
      trainingBars: stats.reduce((sum, s) => sum + s.trainingBars, 0),
      testingBars: stats.reduce((sum, s) => sum + s.testingBars, 0),
      successful: stats.filter(s => s.success).length,
      failed: stats.filter(s => !s.success).length,
    }
  }

  const fs = require('fs')
  const path = require('path')

  const outputPath = path.join(process.cwd(), 'backfill-summary.json')
  fs.writeFileSync(outputPath, JSON.stringify(summary, null, 2))

  console.log(`💾 Summary saved to: ${outputPath}\n`)
}

// ============================================================================
// RUN BACKFILL
// ============================================================================

async function main() {
  try {
    const config: BackfillConfig = {
      tickers: ['SPY', 'QQQ', 'IWM', 'UVXY'],
      timeframes: ['1h', '1d'],  // Start with these, can add more later: '5m', '15m', '4h'
      yearsBack: 2,  // 2 years of historical data
      trainTestSplit: 0.7,  // 70% training, 30% testing
    }

    const stats = await backfillHistoricalData(config)
    printSummary(stats)
    await saveSummary(stats)

    console.log('✅ Backfill complete!\n')
    process.exit(0)
  } catch (error: any) {
    console.error('\n❌ Fatal error:', error.message)
    console.error(error.stack)
    process.exit(1)
  }
}

// Run if called directly
if (require.main === module) {
  main()
}

export { backfillHistoricalData }
export type { BackfillConfig, BackfillStats }
