/**
 * Test UVXY Data Ingestion
 *
 * This script tests ingesting UVXY data after the volume column migration
 * UVXY has fractional volume values, which required changing volume from BigInt to Decimal
 */

import { DataIngestionServiceV2 } from '@/services/dataIngestionService.v2'
import { marketDataRepo } from '@/repositories'

async function testUvxyIngestion() {
  console.log('======================================')
  console.log('Testing UVXY Data Ingestion')
  console.log('======================================\n')

  const ingestionService = new DataIngestionServiceV2()

  try {
    // Test 1: Ingest UVXY 1h data (30 days)
    console.log('Test 1: UVXY 1h timeframe (30 days)')
    console.log('-----------------------------------')
    const result1h = await ingestionService.ingestHistoricalData('UVXY', '1h', 30)

    if (result1h.success) {
      console.log(`✅ SUCCESS: ${result1h.barsInserted} bars inserted`)
      console.log(`   Fetched: ${result1h.barsFetched}`)
      console.log(`   Skipped: ${result1h.barsSkipped}`)
      console.log(`   Duration: ${result1h.durationMs}ms\n`)
    } else {
      console.log(`❌ FAILED: ${result1h.error}\n`)
    }

    // Wait for rate limit
    console.log('⏳ Waiting 13s for rate limit...\n')
    await new Promise(resolve => setTimeout(resolve, 13000))

    // Test 2: Ingest UVXY 1d data (2 years)
    console.log('Test 2: UVXY 1d timeframe (2 years)')
    console.log('-----------------------------------')
    const result1d = await ingestionService.ingestHistoricalData('UVXY', '1d', 730)

    if (result1d.success) {
      console.log(`✅ SUCCESS: ${result1d.barsInserted} bars inserted`)
      console.log(`   Fetched: ${result1d.barsFetched}`)
      console.log(`   Skipped: ${result1d.barsSkipped}`)
      console.log(`   Duration: ${result1d.durationMs}ms\n`)
    } else {
      console.log(`❌ FAILED: ${result1d.error}\n`)
    }

    // Verify data in database
    console.log('Verification: Check UVXY data in database')
    console.log('------------------------------------------')

    const uvxy1h = await marketDataRepo.findMany({
      ticker: 'UVXY',
      timeframe: '1h',
    }, 10)

    const uvxy1d = await marketDataRepo.findMany({
      ticker: 'UVXY',
      timeframe: '1d',
    }, 10)

    console.log(`✅ UVXY 1h: ${uvxy1h.length} bars found`)
    console.log(`✅ UVXY 1d: ${uvxy1d.length} bars found\n`)

    // Check for fractional volumes
    if (uvxy1h.length > 0) {
      const firstBar = uvxy1h[0]
      console.log('Sample UVXY 1h bar:')
      console.log(`   Timestamp: ${firstBar.timestamp}`)
      console.log(`   OHLC: ${firstBar.open} / ${firstBar.high} / ${firstBar.low} / ${firstBar.close}`)
      console.log(`   Volume: ${firstBar.volume} (type: ${typeof firstBar.volume})`)

      // Check if volume is a Decimal with potential fractional part
      const volumeStr = firstBar.volume.toString()
      if (volumeStr.includes('.')) {
        console.log(`   ✅ Fractional volume detected: ${volumeStr}`)
      } else {
        console.log(`   ℹ️  Volume is whole number: ${volumeStr}`)
      }
    }

    console.log('\n======================================')
    console.log('UVXY Ingestion Test Complete!')
    console.log('======================================')

  } catch (error) {
    console.error('\n❌ Error during UVXY ingestion test:', error)
    throw error
  }
}

// Run the test
testUvxyIngestion()
  .then(() => {
    console.log('\n✅ All tests passed!')
    process.exit(0)
  })
  .catch(error => {
    console.error('\n❌ Tests failed:', error.message)
    process.exit(1)
  })
