import { createClient, SupabaseClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

// Lazy initialization - only create client when accessed and env vars are available
let _supabase: SupabaseClient | null = null

function getSupabaseClient() {
  if (_supabase) return _supabase

  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error('Missing Supabase environment variables')
  }

  _supabase = createClient(supabaseUrl, supabaseAnonKey)
  return _supabase
}

// Export a getter instead of direct client
export const supabase = new Proxy({} as SupabaseClient, {
  get: (target, prop) => {
    const client = getSupabaseClient()
    return (client as any)[prop]
  }
})

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
