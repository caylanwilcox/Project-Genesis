/**
 * Mark Train/Test Split in Database
 *
 * Adds metadata to track which data should be used for training vs testing
 * This prevents data leakage and ensures proper ML model validation
 *
 * TRAINING SET: First 70% of data (oldest)
 * TESTING SET:  Last 30% of data (most recent)
 *
 * Usage:
 *   npx ts-node scripts/mark-train-test-split.ts
 */

import { prisma } from '@/lib/prisma'

interface SplitResult {
  ticker: string
  timeframe: string
  totalBars: number
  trainingBars: number
  testingBars: number
  splitDate: Date
}

async function markTrainTestSplit(
  trainSplitRatio: number = 0.7
): Promise<SplitResult[]> {

  console.log('\n╔════════════════════════════════════════════════════════════════╗')
  console.log('║             MARKING TRAIN/TEST SPLIT IN DATABASE              ║')
  console.log('╚════════════════════════════════════════════════════════════════╝\n')
  console.log(`📊 Split Ratio: ${trainSplitRatio * 100}% training / ${(1 - trainSplitRatio) * 100}% testing\n`)

  const results: SplitResult[] = []

  // Get all ticker/timeframe combinations
  const combinations = await prisma.$queryRaw<
    Array<{ ticker: string; timeframe: string }>
  >`
    SELECT DISTINCT ticker, timeframe
    FROM market_data
    ORDER BY ticker, timeframe
  `

  for (const { ticker, timeframe } of combinations) {
    console.log(`\n📈 Processing ${ticker} ${timeframe}...`)

    // Get date range for this ticker/timeframe
    const dateRange = await prisma.$queryRaw<
      Array<{
        earliest: Date
        latest: Date
        total_count: bigint
      }>
    >`
      SELECT
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        COUNT(*) as total_count
      FROM market_data
      WHERE ticker = ${ticker}
        AND timeframe = ${timeframe}
    `

    if (!dateRange[0] || dateRange[0].total_count === 0n) {
      console.log(`   ⚠️  No data found, skipping...`)
      continue
    }

    const { earliest, latest, total_count } = dateRange[0]
    const totalBars = Number(total_count)

    // Calculate split point
    const timeSpan = latest.getTime() - earliest.getTime()
    const trainTimeSpan = timeSpan * trainSplitRatio
    const splitDate = new Date(earliest.getTime() + trainTimeSpan)

    console.log(`   📅 Date Range: ${earliest.toISOString().split('T')[0]} → ${latest.toISOString().split('T')[0]}`)
    console.log(`   🎯 Split Date: ${splitDate.toISOString().split('T')[0]}`)

    // Count training and testing bars
    const trainingCount = await prisma.marketData.count({
      where: {
        ticker,
        timeframe,
        timestamp: { lte: splitDate }
      }
    })

    const testingCount = await prisma.marketData.count({
      where: {
        ticker,
        timeframe,
        timestamp: { gt: splitDate }
      }
    })

    console.log(`   🎓 Training: ${trainingCount.toLocaleString()} bars (${((trainingCount / totalBars) * 100).toFixed(1)}%)`)
    console.log(`   🧪 Testing:  ${testingCount.toLocaleString()} bars (${((testingCount / totalBars) * 100).toFixed(1)}%)`)

    results.push({
      ticker,
      timeframe,
      totalBars,
      trainingBars: trainingCount,
      testingBars: testingCount,
      splitDate
    })
  }

  return results
}

/**
 * Create views for easy querying of training vs testing data
 */
async function createTrainTestViews() {
  console.log('\n📊 Creating database views for train/test splits...\n')

  // This will be useful for querying training vs testing data
  // We'll store the split metadata in a configuration table

  await prisma.$executeRaw`
    CREATE TABLE IF NOT EXISTS train_test_config (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      ticker VARCHAR(10) NOT NULL,
      timeframe VARCHAR(10) NOT NULL,
      split_date TIMESTAMPTZ NOT NULL,
      train_ratio DECIMAL(3,2) NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE(ticker, timeframe)
    )
  `

  console.log('   ✅ Created train_test_config table')
}

/**
 * Save split configuration
 */
async function saveSplitConfig(results: SplitResult[], trainRatio: number) {
  console.log('\n💾 Saving split configuration to database...\n')

  for (const result of results) {
    await prisma.$executeRaw`
      INSERT INTO train_test_config (ticker, timeframe, split_date, train_ratio)
      VALUES (
        ${result.ticker},
        ${result.timeframe},
        ${result.splitDate},
        ${trainRatio}
      )
      ON CONFLICT (ticker, timeframe)
      DO UPDATE SET
        split_date = EXCLUDED.split_date,
        train_ratio = EXCLUDED.train_ratio,
        created_at = NOW()
    `

    console.log(`   ✅ Saved: ${result.ticker} ${result.timeframe}`)
  }
}

/**
 * Helper functions to query training vs testing data
 */
async function printHelperFunctions() {
  console.log('\n╔════════════════════════════════════════════════════════════════╗')
  console.log('║                   HOW TO USE TRAIN/TEST SPLIT                  ║')
  console.log('╚════════════════════════════════════════════════════════════════╝\n')

  console.log('Use these queries in your ML training code:\n')

  console.log('🎓 GET TRAINING DATA:')
  console.log(`
  const trainingData = await prisma.$queryRaw\`
    SELECT md.*
    FROM market_data md
    JOIN train_test_config ttc
      ON md.ticker = ttc.ticker
      AND md.timeframe = ttc.timeframe
    WHERE md.timestamp <= ttc.split_date
      AND md.ticker = 'SPY'
      AND md.timeframe = '1h'
    ORDER BY md.timestamp ASC
  \`
  `)

  console.log('🧪 GET TESTING DATA:')
  console.log(`
  const testingData = await prisma.$queryRaw\`
    SELECT md.*
    FROM market_data md
    JOIN train_test_config ttc
      ON md.ticker = ttc.ticker
      AND md.timeframe = ttc.timeframe
    WHERE md.timestamp > ttc.split_date
      AND md.ticker = 'SPY'
      AND md.timeframe = '1h'
    ORDER BY md.timestamp ASC
  \`
  `)

  console.log('📊 GET SPLIT INFO:')
  console.log(`
  const splitInfo = await prisma.$queryRaw\`
    SELECT
      ticker,
      timeframe,
      split_date,
      train_ratio,
      (SELECT COUNT(*) FROM market_data md
       WHERE md.ticker = ttc.ticker
         AND md.timeframe = ttc.timeframe
         AND md.timestamp <= ttc.split_date) as training_bars,
      (SELECT COUNT(*) FROM market_data md
       WHERE md.ticker = ttc.ticker
         AND md.timeframe = ttc.timeframe
         AND md.timestamp > ttc.split_date) as testing_bars
    FROM train_test_config ttc
    WHERE ticker = 'SPY'
  \`
  `)

  console.log('\n' + '='.repeat(70) + '\n')
}

async function main() {
  try {
    // Create config table
    await createTrainTestViews()

    // Mark the split
    const results = await markTrainTestSplit(0.7)

    // Save to database
    await saveSplitConfig(results, 0.7)

    // Print summary
    console.log('\n╔════════════════════════════════════════════════════════════════╗')
    console.log('║                          SUMMARY                               ║')
    console.log('╚════════════════════════════════════════════════════════════════╝\n')

    const totalBars = results.reduce((sum, r) => sum + r.totalBars, 0)
    const totalTraining = results.reduce((sum, r) => sum + r.trainingBars, 0)
    const totalTesting = results.reduce((sum, r) => sum + r.testingBars, 0)

    console.log(`📊 Total Datasets: ${results.length}`)
    console.log(`📈 Total Bars: ${totalBars.toLocaleString()}`)
    console.log(`🎓 Training Bars: ${totalTraining.toLocaleString()} (${((totalTraining / totalBars) * 100).toFixed(1)}%)`)
    console.log(`🧪 Testing Bars: ${totalTesting.toLocaleString()} (${((totalTesting / totalBars) * 100).toFixed(1)}%)`)

    console.log('\n' + '─'.repeat(70) + '\n')

    results.forEach(r => {
      console.log(`${r.ticker} ${r.timeframe}:`)
      console.log(`  Split: ${r.splitDate.toISOString().split('T')[0]}`)
      console.log(`  Train: ${r.trainingBars.toLocaleString()} | Test: ${r.testingBars.toLocaleString()}`)
    })

    console.log()

    // Print helper info
    await printHelperFunctions()

    console.log('✅ Train/Test split marked successfully!\n')
    await prisma.$disconnect()
    process.exit(0)

  } catch (error: any) {
    console.error('\n❌ Error:', error.message)
    console.error(error.stack)
    await prisma.$disconnect()
    process.exit(1)
  }
}

if (require.main === module) {
  main()
}

export { markTrainTestSplit, createTrainTestViews }
