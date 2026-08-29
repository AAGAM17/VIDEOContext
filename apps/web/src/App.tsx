import { useState } from 'react'
import { VideoUpload } from './components/VideoUpload'
import { VideoExplorer } from './components/VideoExplorer'
import { VideoSearch } from './components/VideoSearch'
import { VideoAsk } from './components/VideoAsk'
import { VideoIntelligence } from './components/VideoIntelligence'
import { ProcessingStatus } from './components/ProcessingStatus'

interface VideoData {
  videoId: string
  filename: string
  status: 'uploaded' | 'processing' | 'completed' | 'failed'
  progress: number
  error?: string
  vctxPath?: string
}

function App() {
  const [video, setVideo] = useState<VideoData | null>(null)
  const [activeTab, setActiveTab] = useState<'explorer' | 'search' | 'ask' | 'intelligence'>('explorer')

  const handleUpload = async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/v1/videos', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setVideo({
        videoId: data.video_id,
        filename: data.filename,
        status: 'uploaded',
        progress: 0,
      })
    } catch (error) {
      console.error('Upload failed:', error)
      alert('Failed to upload video')
    }
  }

  const handleProcess = async () => {
    if (!video) return

    setVideo({ ...video, status: 'processing', progress: 0 })

    try {
      const response = await fetch(`/api/v1/videos/${video.videoId}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: {} }),
      })
      await response.json()

      // Poll for status
      const pollStatus = async () => {
        while (true) {
          await new Promise(resolve => setTimeout(resolve, 1000))
          const statusRes = await fetch(`/api/v1/videos/${video.videoId}/status`)
          const statusData = await statusRes.json()
          setVideo(prev => prev ? { ...prev, progress: statusData.progress, status: statusData.status, error: statusData.error } : null)
          if (statusData.status === 'completed' || statusData.status === 'failed') break
        }
      }
      await pollStatus()
    } catch (error) {
      console.error('Processing failed:', error)
      setVideo(prev => prev ? { ...prev, status: 'failed', error: 'Processing failed' } : null)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header className="border-b border-gray-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white">VideoContext</h1>
          <p className="text-sm text-gray-400">The open-source semantic layer for video</p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {video ? (
          <>
            <ProcessingStatus video={video} onProcess={handleProcess} />
            
            <div className="mt-8 border-b border-gray-700">
              <nav className="flex gap-4" aria-label="Main tabs">
                <button
                  onClick={() => setActiveTab('explorer')}
                  className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                    activeTab === 'explorer'
                      ? 'bg-gray-800 text-white'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  Explorer
                </button>
                <button
                  onClick={() => setActiveTab('search')}
                  className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                    activeTab === 'search'
                      ? 'bg-gray-800 text-white'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  Search
                </button>
                <button
                  onClick={() => setActiveTab('ask')}
                  className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                    activeTab === 'ask'
                      ? 'bg-gray-800 text-white'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  Ask AI
                </button>
                <button
                  onClick={() => setActiveTab('intelligence')}
                  className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                    activeTab === 'intelligence'
                      ? 'bg-gray-800 text-white'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  Intelligence
                </button>
              </nav>
            </div>

            <div className="mt-6">
              {activeTab === 'explorer' && <VideoExplorer videoId={video.videoId} />}
              {activeTab === 'search' && <VideoSearch videoId={video.videoId} />}
              {activeTab === 'ask' && <VideoAsk videoId={video.videoId} />}
              {activeTab === 'intelligence' && <VideoIntelligence videoId={video.videoId} />}
            </div>
          </>
        ) : (
          <VideoUpload onUpload={handleUpload} />
        )}
      </main>
    </div>
  )
}

export default App