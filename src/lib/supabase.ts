import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables')
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Database types for TypeScript
export interface MarketData {
  id: string
  ticker: string
  timeframe: string
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  source: string
  created_at: string
}

export interface Feature {
  id: string
  ticker: string
  timeframe: string
  timestamp: string
  feature_name: string
  feature_value: number
  created_at: string
}

export interface Prediction {
  id: string
  ticker: string
  timeframe: string
  timestamp: string
  model_name: string
  predicted_direction: 'up' | 'down' | 'neutral'
  predicted_change: number
  confidence: number
  actual_direction?: 'up' | 'down' | 'neutral'
  actual_change?: number
  accuracy?: number
  created_at: string
}
