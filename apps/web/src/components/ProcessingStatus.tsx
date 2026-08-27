interface ProcessingStatusProps {
  video: {
    status: 'uploaded' | 'processing' | 'completed' | 'failed'
    progress: number
    filename: string
    error?: string
  }
  onProcess: () => void
}

export function ProcessingStatus({ video, onProcess }: ProcessingStatusProps) {
  const statusColors = {
    uploaded: 'bg-gray-600',
    processing: 'bg-blue-500',
    completed: 'bg-green-500',
    failed: 'bg-red-500',
  }

  const statusLabels = {
    uploaded: 'Ready to process',
    processing: 'Processing...',
    completed: 'Processing complete',
    failed: 'Processing failed',
  }

  return (
    <div className="bg-gray-800 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-white">{video.filename}</h2>
          <p className="text-sm text-gray-400">{statusLabels[video.status]}</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="w-48 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`${statusColors[video.status]} h-full transition-all duration-300`}
              style={{ width: `${video.progress * 100}%` }}
            />
          </div>
          <span className="text-sm font-mono text-gray-300 w-10 text-right">
            {Math.round(video.progress * 100)}%
          </span>
        </div>
      </div>

      {video.status === 'uploaded' && (
        <button
          onClick={onProcess}
          className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
        >
          Start Processing
        </button>
      )}

      {video.status === 'completed' && (
        <div className="text-green-400 text-sm">
          ✓ Video processed successfully. Switch tabs to explore, search, or ask questions.
        </div>
      )}

      {video.status === 'failed' && (
        <div className="text-red-400 text-sm">
          ✗ Error: {video.error || 'Unknown error'}
        </div>
      )}
    </div>
  )
}