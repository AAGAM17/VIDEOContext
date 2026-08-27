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

interface AskResponse {
  question: string
  answer: string
  confidence: number
  evidence: SearchHit[]
}

export function VideoAsk({ videoId }: { videoId: string }) {
  const [question, setQuestion] = useState('')
  const [response, setResponse] = useState<AskResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim()) return

    setLoading(true)
    try {
      const res = await fetch(`/api/v1/videos/${videoId}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 5 }),
      })
      const data = await res.json()
      setResponse(data)
    } catch (error) {
      console.error('Ask failed:', error)
    } finally {
      setLoading(false)
    }
  }

  if (!response) {
    return (
      <div className="space-y-4">
        <form onSubmit={handleAsk} className="flex gap-2">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about the video... (e.g., 'What was the revenue?', 'When was the pricing slide shown?')"
            className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
          >
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </form>
        <p className="text-sm text-gray-500">
          Try questions like: "What was the revenue?", "When was the pricing slide shown?", "What command was typed?"
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleAsk} className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask another question..."
          className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
        >
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </form>

      <div className="bg-gray-800 rounded-xl p-6">
        <div className="mb-4">
          <p className="text-sm font-medium text-gray-400">Question</p>
          <p className="text-white mt-1">{response.question}</p>
        </div>

        <div className="mb-4">
          <p className="text-sm font-medium text-gray-400">Answer</p>
          <p className="text-white mt-1 whitespace-pre-wrap">{response.answer}</p>
        </div>

        <div className="mb-4 flex items-center gap-4">
          <div className="px-3 py-1 bg-gray-700 rounded-lg text-sm">
            Confidence: <span className="font-mono text-white">{Math.round(response.confidence * 100)}%</span>
          </div>
        </div>

        {response.evidence.length > 0 && (
          <div>
            <p className="text-sm font-medium text-gray-400 mb-3">Evidence ({response.evidence.length} spans)</p>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {response.evidence.map((hit, index) => (
                <div
                  key={index}
                  className="p-3 rounded-lg bg-gray-700/50 border border-gray-600"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-mono text-xs text-gray-400">{hit.timecode}</span>
                    <span className="px-2 py-0.5 text-xs font-medium rounded bg-gray-700 text-gray-200">
                      {hit.modality}
                    </span>
                  </div>
                  <p className="text-sm text-gray-200">{hit.text}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}