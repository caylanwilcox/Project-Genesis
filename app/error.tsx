'use client'

import { useEffect } from 'react'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error('App error:', error)
  }, [error])

  return (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="text-center max-w-md px-4">
        <div className="text-red-400 text-6xl mb-4">⚠️</div>
        <h2 className="text-white text-2xl font-bold mb-2">Something went wrong!</h2>
        <p className="text-gray-400 text-sm mb-4">
          {error.message || 'An unexpected error occurred'}
        </p>
        <button
          onClick={reset}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors font-medium"
        >
          Try again
        </button>
      </div>
    </div>
  )
}
