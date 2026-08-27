import { useState } from 'react'

interface SearchHit {
  timecode: string
  start: number
  end: number
  modality: string
  text: string
  score: number
  reason: string
}

export function VideoSearch({ videoId }: { videoId: string }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchHit[]>([])
  const [loading, setLoading] = useState(false)
  const [total, setTotal] = useState(0)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    try {
      const response = await fetch(`/api/v1/videos/${videoId}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 10 }),
      })
      const data = await response.json()
      setResults(data.hits || [])
      setTotal(data.total || 0)
    } catch (error) {
      console.error('Search failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const modalityColors = {
    transcript: 'bg-blue-500/20 border-blue-500',
    ocr: 'bg-green-500/20 border-green-500',
    vision: 'bg-purple-500/20 border-purple-500',
    events: 'bg-orange-500/20 border-orange-500',
  }

  return (
    <div className="space-y-4">
      <form onSubmit={handleSearch} className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search the video... (e.g., 'pricing', 'revenue', 'error')"
          className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {results.length > 0 && (
        <p className="text-sm text-gray-400">
          Found {total} result{total !== 1 ? 's' : ''} for "{query}"
        </p>
      )}

      <div className="space-y-3">
        {results.length === 0 && !loading && query ? (
          <p className="text-gray-500 text-center py-8">No matches found</p>
        ) : (
          results.map((hit, index) => (
            <div
              key={index}
              className={`p-4 rounded-lg border ${modalityColors[hit.modality as keyof typeof modalityColors] || 'bg-gray-700/50 border-gray-600'}`}
            >
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-24 font-mono text-xs text-gray-400">
                  {hit.timecode}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="px-2 py-0.5 text-xs font-medium rounded bg-gray-700 text-gray-200">
                      {hit.modality}
                    </span>
                    <span className="px-2 py-0.5 text-xs font-medium rounded bg-blue-500/20 text-blue-400">
                      Score: {hit.score.toFixed(3)}
                    </span>
                  </div>
                  <p className="text-gray-100 whitespace-pre-wrap break-words mb-2">{hit.text}</p>
                  <p className="text-xs text-gray-500">{hit.reason}</p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}