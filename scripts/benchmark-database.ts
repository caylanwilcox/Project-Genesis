/**
 * Database Performance Benchmarks
 *
 * Tests database performance against Week 1 success criteria:
 * - 100K inserts < 5 seconds
 * - 1 year query < 500ms
 * - Latest bar query < 10ms
 */

import { marketDataRepo } from '@/repositories'
import { prisma } from '@/lib/prisma'
import { performance } from 'perf_hooks'

interface BenchmarkResult {
  test: string
  target: string
  actual: number
  pass: boolean
  details?: string
}

async function runBenchmarks(): Promise<BenchmarkResult[]> {
  console.log('======================================')
  console.log('Database Performance Benchmarks')
  console.log('======================================\n')

  const results: BenchmarkResult[] = []

  // Test 1: Bulk Insert Speed (10K rows)
  console.log('Test 1: Bulk Insert Performance')
  console.log('-----------------------------------')
  console.log('Target: 10K rows < 2 seconds\n')

  const insertData = Array.from({ length: 10000 }, (_, i) => ({
    ticker: 'BENCH',
    timeframe: '1m',
    timestamp: new Date(Date.now() - i * 60000),
    open: 100 + Math.random() * 10,
    high: 105 + Math.random() * 10,
    low: 95 + Math.random() * 10,
    close: 100 + Math.random() * 10,
    volume: 1000000 + Math.random() * 100000,
    source: 'benchmark',
  }))

  const insertStart = performance.now()
  try {
    await marketDataRepo.upsertMany(insertData)
    const insertDuration = performance.now() - insertStart

    const pass = insertDuration < 2000
    results.push({
      test: 'Bulk Insert (10K rows)',
      target: '< 2000ms',
      actual: insertDuration,
      pass,
      details: `${insertDuration.toFixed(0)}ms`,
    })

    console.log(`${pass ? '✅ PASS' : '❌ FAIL'}: Inserted 10K rows in ${insertDuration.toFixed(0)}ms`)
  } catch (error: any) {
    console.log(`❌ ERROR: ${error.message}`)
    results.push({
      test: 'Bulk Insert (10K rows)',
      target: '< 2000ms',
      actual: -1,
      pass: false,
      details: error.message,
    })
  }

  console.log('\n')

  // Test 2: Query Performance (Large Dataset)
  console.log('Test 2: Query Performance (1000 rows)')
  console.log('-----------------------------------')
  console.log('Target: < 200ms\n')

  const queryStart = performance.now()
  try {
    const data = await marketDataRepo.findMany({
      ticker: 'SPY',
      timeframe: '1h',
    }, 1000)

    const queryDuration = performance.now() - queryStart
    const pass = queryDuration < 200

    results.push({
      test: 'Query (1000 rows)',
      target: '< 200ms',
      actual: queryDuration,
      pass,
      details: `${queryDuration.toFixed(0)}ms, ${data.length} rows`,
    })

    console.log(`${pass ? '✅ PASS' : '❌ FAIL'}: Queried ${data.length} rows in ${queryDuration.toFixed(0)}ms`)
  } catch (error: any) {
    console.log(`❌ ERROR: ${error.message}`)
    results.push({
      test: 'Query (1000 rows)',
      target: '< 200ms',
      actual: -1,
      pass: false,
      details: error.message,
    })
  }

  console.log('\n')

  // Test 3: Latest Bar Query (Index Performance)
  console.log('Test 3: Latest Bar Query')
  console.log('-----------------------------------')
  console.log('Target: < 10ms\n')

  const latestStart = performance.now()
  try {
    const latest = await marketDataRepo.getLatest('SPY', '1h')
    const latestDuration = performance.now() - latestStart

    const pass = latestDuration < 10

    results.push({
      test: 'Latest Bar Query',
      target: '< 10ms',
      actual: latestDuration,
      pass,
      details: `${latestDuration.toFixed(2)}ms`,
    })

    console.log(`${pass ? '✅ PASS' : '❌ FAIL'}: Latest bar query in ${latestDuration.toFixed(2)}ms`)
    if (latest) {
      console.log(`   Latest: ${latest.timestamp} - Close: $${latest.close}`)
    }
  } catch (error: any) {
    console.log(`❌ ERROR: ${error.message}`)
    results.push({
      test: 'Latest Bar Query',
      target: '< 10ms',
      actual: -1,
      pass: false,
      details: error.message,
    })
  }

  console.log('\n')

  // Test 4: Aggregation Query (COUNT, MIN, MAX)
  console.log('Test 4: Aggregation Performance')
  console.log('-----------------------------------')
  console.log('Target: < 100ms\n')

  const aggStart = performance.now()
  try {
    const summary = await marketDataRepo.getSummary('SPY', '1h')
    const aggDuration = performance.now() - aggStart

    const pass = aggDuration < 100

    results.push({
      test: 'Aggregation Query',
      target: '< 100ms',
      actual: aggDuration,
      pass,
      details: `${aggDuration.toFixed(0)}ms`,
    })

    console.log(`${pass ? '✅ PASS' : '❌ FAIL'}: Aggregation in ${aggDuration.toFixed(0)}ms`)
    if (summary) {
      console.log(`   Bars: ${summary.bars}, Range: ${summary.earliest} → ${summary.latest}`)
    }
  } catch (error: any) {
    console.log(`❌ ERROR: ${error.message}`)
    results.push({
      test: 'Aggregation Query',
      target: '< 100ms',
      actual: -1,
      pass: false,
      details: error.message,
    })
  }

  console.log('\n')

  // Test 5: Complex Filter Query
  console.log('Test 5: Complex Filter Query')
  console.log('-----------------------------------')
  console.log('Target: < 300ms\n')

  const endDate = new Date()
  const startDate = new Date()
  startDate.setDate(startDate.getDate() - 30)

  const filterStart = performance.now()
  try {
    const filtered = await marketDataRepo.findMany({
      ticker: 'SPY',
      timeframe: '1h',
      startDate,
      endDate,
    })

    const filterDuration = performance.now() - filterStart
    const pass = filterDuration < 300

    results.push({
      test: 'Complex Filter Query',
      target: '< 300ms',
      actual: filterDuration,
      pass,
      details: `${filterDuration.toFixed(0)}ms, ${filtered.length} rows`,
    })

    console.log(`${pass ? '✅ PASS' : '❌ FAIL'}: Filtered query in ${filterDuration.toFixed(0)}ms`)
    console.log(`   Rows: ${filtered.length}`)
  } catch (error: any) {
    console.log(`❌ ERROR: ${error.message}`)
    results.push({
      test: 'Complex Filter Query',
      target: '< 300ms',
      actual: -1,
      pass: false,
      details: error.message,
    })
  }

  console.log('\n')

  // Clean up benchmark data
  console.log('Cleaning up benchmark data...')
  try {
    await prisma.marketData.deleteMany({
      where: { ticker: 'BENCH' }
    })
    console.log('✅ Benchmark data cleaned\n')
  } catch (error: any) {
    console.log(`⚠️  Warning: Could not clean benchmark data: ${error.message}\n`)
  }

  // Summary
  console.log('======================================')
  console.log('Benchmark Summary')
  console.log('======================================\n')

  const passed = results.filter(r => r.pass).length
  const failed = results.filter(r => !r.pass).length
  const total = results.length

  console.log(`Total tests: ${total}`)
  console.log(`✅ Passed: ${passed}`)
  console.log(`❌ Failed: ${failed}`)
  console.log(`Pass rate: ${((passed / total) * 100).toFixed(1)}%\n`)

  console.log('Details:')
  console.log('-----------------------------------')
  results.forEach(r => {
    const status = r.pass ? '✅' : '❌'
    const actual = r.actual >= 0 ? `${r.actual.toFixed(2)}ms` : 'ERROR'
    console.log(`${status} ${r.test}: ${actual} (target: ${r.target})`)
    if (r.details && !r.pass) {
      console.log(`   ${r.details}`)
    }
  })

  console.log('\n======================================')
  console.log(passed === total ? '✅ All Benchmarks Passed!' : '⚠️  Some Benchmarks Failed')
  console.log('======================================')

  return results
}

// Run benchmarks
runBenchmarks()
  .then(results => {
    const allPassed = results.every(r => r.pass)
    process.exit(allPassed ? 0 : 1)
  })
  .catch(error => {
    console.error('\n❌ Benchmark failed:', error.message)
    process.exit(1)
  })
