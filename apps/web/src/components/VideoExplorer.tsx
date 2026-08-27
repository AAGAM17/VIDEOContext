import { useState, useEffect } from 'react'

interface TimelineEntry {
  start: number
  end: number
  modality: string
  text: string
}

export function VideoExplorer({ videoId }: { videoId: string }) {
  const [timeline, setTimeline] = useState<TimelineEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<string>('all')

  useEffect(() => {
    fetch(`/api/v1/videos/${videoId}/timeline`)
      .then(res => res.json())
      .then(data => {
        setTimeline(data.timeline || [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [videoId])

  const filteredTimeline = timeline.filter(entry => 
    filter === 'all' || entry.modality === filter
  )

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 1000)
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`
  }

  const modalityColors = {
    transcript: 'bg-blue-500/20 border-blue-500',
    ocr: 'bg-green-500/20 border-green-500',
    vision: 'bg-purple-500/20 border-purple-500',
    events: 'bg-orange-500/20 border-orange-500',
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {['all', 'transcript', 'ocr', 'vision', 'events'].map(mod => (
          <button
            key={mod}
            onClick={() => setFilter(mod)}
            className={`px-3 py-1 text-sm rounded-full transition-colors ${
              filter === mod
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            {mod.charAt(0).toUpperCase() + mod.slice(1)}
          </button>
        ))}
      </div>

      <div className="max-h-96 overflow-y-auto space-y-2">
        {filteredTimeline.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No timeline entries found</p>
        ) : (
          filteredTimeline.map((entry, index) => (
            <div
              key={index}
              className={`p-4 rounded-lg border ${modalityColors[entry.modality as keyof typeof modalityColors] || 'bg-gray-700/50 border-gray-600'}`}
            >
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-24 font-mono text-xs text-gray-400">
                  {formatTime(entry.start)} → {formatTime(entry.end)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="px-2 py-0.5 text-xs font-medium rounded bg-gray-700 text-gray-200">
                      {entry.modality}
                    </span>
                  </div>
                  <p className="text-gray-100 whitespace-pre-wrap break-words">{entry.text}</p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}