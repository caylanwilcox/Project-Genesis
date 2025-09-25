'use client'

import dynamic from 'next/dynamic'

const TradingView = dynamic(() => import('@/components/TradingView').then(m => m.TradingView), {
  ssr: false,
})

export default function Home() {
  return <TradingView />
}