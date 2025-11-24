/**
 * Diagnostic script to test 1M timeframe data feed
 * Compares 1h vs 1d intervals to see what data is being returned
 */

const { restClient } = require('@polygon.io/client-js');
require('dotenv').config({ path: '.env.local' });

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

async function test1MData() {
  const ticker = 'SPY';
  const now = Date.now();
  const toDate = getLastTradingDate(new Date(now));

  // Calculate from date for 1M (go back ~90 days to be safe)
  const fromDate = new Date(toDate.getTime());
  fromDate.setDate(fromDate.getDate() - 90);

  const toDateStr = formatDate(toDate);
  const fromDateStr = formatDate(fromDate);

  console.log('='.repeat(80));
  console.log('POLYGON.IO 1M TIMEFRAME DATA DIAGNOSTIC');
  console.log('='.repeat(80));
  console.log(`Ticker: ${ticker}`);
  console.log(`Current Time: ${new Date(now).toISOString()}`);
  console.log(`To Date (Last Trading Day): ${toDate.toISOString()}`);
  console.log(`From Date: ${fromDate.toISOString()}`);
  console.log(`Date Range: ${fromDateStr} to ${toDateStr}`);
  console.log('='.repeat(80));
  console.log('');

  // Test 1: 1-hour bars (what 1M display uses)
  console.log('TEST 1: Fetching 1-hour bars (1M display timeframe)');
  console.log('-'.repeat(80));
  try {
    const response1h = await client.stocks.aggregates(ticker, 1, 'hour', fromDateStr, toDateStr, {
      adjusted: true,
      sort: 'asc',
      limit: 50000
    });

    console.log(`Status: ${response1h.status}`);
    console.log(`Results Count: ${response1h.resultsCount}`);
    console.log(`Query Count: ${response1h.queryCount}`);
    console.log(`Total Bars: ${response1h.results?.length || 0}`);

    if (response1h.results && response1h.results.length > 0) {
      const first = response1h.results[0];
      const last = response1h.results[response1h.results.length - 1];

      console.log('');
      console.log('First Bar:');
      console.log(`  Time: ${formatDateTime(first.t)} (${new Date(first.t).toISOString()})`);
      console.log(`  OHLC: O=$${first.o} H=$${first.h} L=$${first.l} C=$${first.c}`);
      console.log(`  Volume: ${first.v.toLocaleString()}`);

      console.log('');
      console.log('Last Bar:');
      console.log(`  Time: ${formatDateTime(last.t)} (${new Date(last.t).toISOString()})`);
      console.log(`  OHLC: O=$${last.o} H=$${last.h} L=$${last.l} C=$${last.c}`);
      console.log(`  Volume: ${last.v.toLocaleString()}`);

      // Show last 10 bars for inspection
      console.log('');
      console.log('Last 10 Bars:');
      const last10 = response1h.results.slice(-10);
      last10.forEach((bar, idx) => {
        console.log(`  ${idx + 1}. ${formatDateTime(bar.t)} - Close: $${bar.c}`);
      });

      // Check staleness
      const nowMs = Date.now();
      const lastBarMs = last.t;
      const stalenessHours = (nowMs - lastBarMs) / (1000 * 60 * 60);
      console.log('');
      console.log(`Data Staleness: ${stalenessHours.toFixed(2)} hours old`);
      console.log(`Last bar is from: ${formatDateTime(lastBarMs)}`);
      console.log(`Current time is: ${formatDateTime(nowMs)}`);
    } else {
      console.log('NO DATA RETURNED');
    }
  } catch (error) {
    console.error('ERROR:', error.message);
  }

  console.log('');
  console.log('='.repeat(80));
  console.log('');

  // Test 2: Daily bars (what 1M display could use)
  console.log('TEST 2: Fetching daily bars (for comparison)');
  console.log('-'.repeat(80));
  try {
    const response1d = await client.stocks.aggregates(ticker, 1, 'day', fromDateStr, toDateStr, {
      adjusted: true,
      sort: 'asc',
      limit: 50000
    });

    console.log(`Status: ${response1d.status}`);
    console.log(`Results Count: ${response1d.resultsCount}`);
    console.log(`Query Count: ${response1d.queryCount}`);
    console.log(`Total Bars: ${response1d.results?.length || 0}`);

    if (response1d.results && response1d.results.length > 0) {
      const first = response1d.results[0];
      const last = response1d.results[response1d.results.length - 1];

      console.log('');
      console.log('First Bar:');
      console.log(`  Date: ${formatDateTime(first.t)} (${new Date(first.t).toISOString()})`);
      console.log(`  OHLC: O=$${first.o} H=$${first.h} L=$${first.l} C=$${first.c}`);
      console.log(`  Volume: ${first.v.toLocaleString()}`);

      console.log('');
      console.log('Last Bar:');
      console.log(`  Date: ${formatDateTime(last.t)} (${new Date(last.t).toISOString()})`);
      console.log(`  OHLC: O=$${last.o} H=$${last.h} L=$${last.l} C=$${last.c}`);
      console.log(`  Volume: ${last.v.toLocaleString()}`);

      // Show last 10 bars for inspection
      console.log('');
      console.log('Last 10 Bars:');
      const last10 = response1d.results.slice(-10);
      last10.forEach((bar, idx) => {
        console.log(`  ${idx + 1}. ${formatDateTime(bar.t)} - Close: $${bar.c}`);
      });

      // Check staleness
      const nowMs = Date.now();
      const lastBarMs = last.t;
      const stalenessDays = (nowMs - lastBarMs) / (1000 * 60 * 60 * 24);
      console.log('');
      console.log(`Data Staleness: ${stalenessDays.toFixed(2)} days old`);
      console.log(`Last bar is from: ${formatDateTime(lastBarMs)}`);
      console.log(`Current time is: ${formatDateTime(nowMs)}`);
    } else {
      console.log('NO DATA RETURNED');
    }
  } catch (error) {
    console.error('ERROR:', error.message);
  }

  console.log('');
  console.log('='.repeat(80));
  console.log('DIAGNOSIS COMPLETE');
  console.log('='.repeat(80));
  console.log('');
  console.log('KEY QUESTIONS TO ANSWER:');
  console.log('1. Does 1h data show bars from today (or last trading day)?');
  console.log('2. Is the last 1h bar close to the market close time (4:00 PM ET)?');
  console.log('3. Does 1d data show more recent date than 1h data?');
  console.log('4. What is the staleness (hours/days) of each dataset?');
  console.log('');
}

// Run the diagnostic
test1MData().catch(console.error);
