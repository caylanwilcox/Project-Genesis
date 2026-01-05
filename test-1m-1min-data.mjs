/**
 * Diagnostic script to test 1M/1m (1-minute) data
 * Check why 1-minute data is showing almost a month behind
 */

import { restClient } from '@polygon.io/client-js';
import { config } from 'dotenv';

config({ path: '.env.local' });

const apiKey = process.env.NEXT_PUBLIC_POLYGON_API_KEY;

if (!apiKey) {
  console.error('ERROR: NEXT_PUBLIC_POLYGON_API_KEY not found in .env.local');
  process.exit(1);
}

const client = restClient(apiKey);

function formatDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function formatDateTime(timestamp) {
  const date = new Date(timestamp);
  return date.toLocaleString('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false
  });
}

function getLastTradingDate(date) {
  const adjusted = new Date(date.getTime());
  let safety = 7;
  while (safety > 0) {
    const day = adjusted.getDay();
    if (day !== 0 && day !== 6) {
      return adjusted;
    }
    adjusted.setDate(adjusted.getDate() - 1);
    safety -= 1;
  }
  return adjusted;
}

async function test1MinuteData() {
  const ticker = 'SPY';
  const now = Date.now();
  const toDate = getLastTradingDate(new Date(now));

  console.log('='.repeat(80));
  console.log('POLYGON.IO 1M/1m (1-MINUTE) DATA DIAGNOSTIC');
  console.log('='.repeat(80));
  console.log(`Ticker: ${ticker}`);
  console.log(`Current Time: ${new Date(now).toISOString()}`);
  console.log(`To Date (Last Trading Day): ${toDate.toISOString()}`);
  console.log('='.repeat(80));
  console.log('');

  // Test 1: Request 1-minute bars for LAST 1 TRADING DAY (what should work)
  console.log('TEST 1: Last 1 Trading Day (1-minute bars)');
  console.log('-'.repeat(80));
  const from1Day = new Date(toDate.getTime());
  from1Day.setDate(from1Day.getDate() - 1);

  try {
    const response = await client.getStocksAggregates(
      ticker,
      1,
      'minute',
      formatDate(from1Day),
      formatDate(toDate),
      true,
      'asc',
      50000
    );

    console.log(`Request: ${formatDate(from1Day)} to ${formatDate(toDate)}`);
    console.log(`Status: ${response.status}`);
    console.log(`Total Bars: ${response.results?.length || 0}`);

    if (response.results && response.results.length > 0) {
      const first = response.results[0];
      const last = response.results[response.results.length - 1];

      console.log(`First: ${formatDateTime(first.t)} Close: $${first.c.toFixed(2)}`);
      console.log(`Last: ${formatDateTime(last.t)} Close: $${last.c.toFixed(2)}`);
      console.log(`Staleness: ${((now - last.t) / (1000 * 60 * 60)).toFixed(2)} hours`);
    }
  } catch (error) {
    console.error('ERROR:', error.message);
  }

  console.log('');
  console.log('='.repeat(80));
  console.log('');

  // Test 2: Request 1-minute bars for LAST 5 DAYS
  console.log('TEST 2: Last 5 Days (1-minute bars)');
  console.log('-'.repeat(80));
  const from5Days = new Date(toDate.getTime());
  from5Days.setDate(from5Days.getDate() - 5);

  try {
    const response = await client.getStocksAggregates(
      ticker,
      1,
      'minute',
      formatDate(from5Days),
      formatDate(toDate),
      true,
      'asc',
      50000
    );

    console.log(`Request: ${formatDate(from5Days)} to ${formatDate(toDate)}`);
    console.log(`Status: ${response.status}`);
    console.log(`Total Bars: ${response.results?.length || 0}`);

    if (response.results && response.results.length > 0) {
      const first = response.results[0];
      const last = response.results[response.results.length - 1];

      console.log(`First: ${formatDateTime(first.t)} Close: $${first.c.toFixed(2)}`);
      console.log(`Last: ${formatDateTime(last.t)} Close: $${last.c.toFixed(2)}`);
      console.log(`Staleness: ${((now - last.t) / (1000 * 60 * 60)).toFixed(2)} hours`);
    }
  } catch (error) {
    console.error('ERROR:', error.message);
  }

  console.log('');
  console.log('='.repeat(80));
  console.log('');

  // Test 3: Request 1-minute bars for LAST 30 DAYS (simulating 1M view)
  console.log('TEST 3: Last 30 Days (1-minute bars) - Simulating 1M View');
  console.log('-'.repeat(80));
  const from30Days = new Date(toDate.getTime());
  from30Days.setDate(from30Days.getDate() - 30);

  try {
    const response = await client.getStocksAggregates(
      ticker,
      1,
      'minute',
      formatDate(from30Days),
      formatDate(toDate),
      true,
      'asc',
      50000
    );

    console.log(`Request: ${formatDate(from30Days)} to ${formatDate(toDate)}`);
    console.log(`Status: ${response.status}`);
    console.log(`Total Bars: ${response.results?.length || 0}`);

    if (response.results && response.results.length > 0) {
      const first = response.results[0];
      const last = response.results[response.results.length - 1];

      console.log(`First: ${formatDateTime(first.t)} Close: $${first.c.toFixed(2)}`);
      console.log(`Last: ${formatDateTime(last.t)} Close: $${last.c.toFixed(2)}`);

      const stalenessHours = (now - last.t) / (1000 * 60 * 60);
      const stalenessDays = stalenessHours / 24;
      console.log(`Staleness: ${stalenessHours.toFixed(2)} hours (${stalenessDays.toFixed(2)} days)`);

      console.log('');
      console.log('Last 5 Bars:');
      const last5 = response.results.slice(-5);
      last5.forEach((bar, idx) => {
        console.log(`  ${idx + 1}. ${formatDateTime(bar.t)} - Close: $${bar.c.toFixed(2)}`);
      });
    }
  } catch (error) {
    console.error('ERROR:', error.message);
  }

  console.log('');
  console.log('='.repeat(80));
  console.log('');

  // Test 4: Request 1-hour bars for comparison
  console.log('TEST 4: Last 30 Days (1-hour bars) - For Comparison');
  console.log('-'.repeat(80));

  try {
    const response = await client.getStocksAggregates(
      ticker,
      1,
      'hour',
      formatDate(from30Days),
      formatDate(toDate),
      true,
      'asc',
      50000
    );

    console.log(`Request: ${formatDate(from30Days)} to ${formatDate(toDate)}`);
    console.log(`Status: ${response.status}`);
    console.log(`Total Bars: ${response.results?.length || 0}`);

    if (response.results && response.results.length > 0) {
      const first = response.results[0];
      const last = response.results[response.results.length - 1];

      console.log(`First: ${formatDateTime(first.t)} Close: $${first.c.toFixed(2)}`);
      console.log(`Last: ${formatDateTime(last.t)} Close: $${last.c.toFixed(2)}`);

      const stalenessHours = (now - last.t) / (1000 * 60 * 60);
      const stalenessDays = stalenessHours / 24;
      console.log(`Staleness: ${stalenessHours.toFixed(2)} hours (${stalenessDays.toFixed(2)} days)`);
    }
  } catch (error) {
    console.error('ERROR:', error.message);
  }

  console.log('');
  console.log('='.repeat(80));
  console.log('DIAGNOSIS COMPLETE');
  console.log('='.repeat(80));
  console.log('');
  console.log('KEY FINDINGS:');
  console.log('- Does 1-minute data work for recent days (1-5 days)?');
  console.log('- Does 1-minute data break when requesting 30 days?');
  console.log('- Is there a limit on how far back 1-minute data goes on free tier?');
  console.log('- Compare staleness: 1m vs 1h for same date range');
  console.log('');
}

// Run the diagnostic
test1MinuteData().catch(console.error);
