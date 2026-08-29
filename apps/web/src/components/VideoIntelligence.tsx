import { useState, useEffect } from 'react'

interface ContextResponse {
  task: string
  task_type: string
  confidence: number
  context: string
  profiles: Record<string, any>
  evidence: any[]
  frames: any[]
  global_context: any
  summaries: Record<string, string>
  token_estimate: number
}

export function VideoIntelligence({ videoId }: { videoId: string }) {
  const [activeView, setActiveView] = useState<'overview' | 'profiles' | 'context' | 'context_builder'>('overview')
  const [overview, setOverview] = useState<any>(null)
  const [profiles, setProfiles] = useState<Record<string, any>>({})
  const [loadingProfiles, setLoadingProfiles] = useState(false)

  // Context builder state
  const [taskInput, setTaskInput] = useState('')
  const [contextResult, setContextResult] = useState<ContextResponse | null>(null)
  const [loadingContext, setLoadingContext] = useState(false)
  const [maxTokens, setMaxTokens] = useState(4000)

  // Profile detail state
  const [selectedProfile, setSelectedProfile] = useState<string | null>(null)
  const [profileData, setProfileData] = useState<any>(null)

  useEffect(() => {
    loadOverview()
    loadProfiles()
  }, [videoId])

  const loadOverview = async () => {
    try {
      const response = await fetch(`/api/v1/videos/${videoId}/profiles`)
      const data = await response.json()
      setOverview(data)
    } catch (error) {
      console.error('Failed to load overview:', error)
    }
  }

  const loadProfiles = async () => {
    setLoadingProfiles(true)
    try {
      const response = await fetch(`/api/v1/videos/${videoId}/profiles`)
      const data = await response.json()
      setProfiles(data.profiles || {})
    } catch (error) {
      console.error('Failed to load profiles:', error)
    } finally {
      setLoadingProfiles(false)
    }
  }

  const loadProfileDetail = async (name: string) => {
    setSelectedProfile(name)
    try {
      const response = await fetch(`/api/v1/videos/${videoId}/profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_name: name }),
      })
      const data = await response.json()
      if (data.profile) {
        setProfileData(data.profile)
      }
    } catch (error) {
      console.error('Failed to load profile:', error)
    }
  }

  const generateContext = async () => {
    if (!taskInput.trim()) return
    setLoadingContext(true)
    try {
      const response = await fetch(`/api/v1/videos/${videoId}/context`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task: taskInput,
          max_tokens: maxTokens,
        }),
      })
      const data = await response.json()
      setContextResult(data)
      setActiveView('context')
    } catch (error) {
      console.error('Failed to generate context:', error)
      alert('Failed to generate context')
    } finally {
      setLoadingContext(false)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    alert('Copied to clipboard!')
  }

  const formatJson = (obj: any) => JSON.stringify(obj, null, 2)

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2 mb-4">
        <button
          onClick={() => setActiveView('overview')}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            activeView === 'overview'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          Overview
        </button>
        <button
          onClick={() => { setActiveView('profiles'); loadProfiles(); }}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            activeView === 'profiles'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          Semantic Profiles
        </button>
        <button
          onClick={() => setActiveView('context_builder')}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            activeView === 'context_builder'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          AI Context Builder
        </button>
        {contextResult && (
          <button
            onClick={() => setActiveView('context')}
            className="px-4 py-2 text-sm font-medium rounded-lg transition-colors bg-green-600 text-white"
          >
            View Context
          </button>
        )}
      </div>

      {activeView === 'overview' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-gray-400 mb-2">Video Info</h3>
              <p className="text-white font-mono text-sm">
                {overview?.video?.filename || 'Unknown'}
              </p>
              <p className="text-gray-400 text-sm mt-1">
                Duration: {overview?.video?.duration ? Math.round(overview.video.duration) + 's' : 'Unknown'}
              </p>
            </div>
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-gray-400 mb-2">Available Profiles</h3>
              <p className="text-white font-mono text-2xl">
                {Object.keys(profiles).length || 0}
              </p>
              <p className="text-gray-400 text-sm mt-1">
                {Object.keys(profiles).join(', ') || 'None'}
              </p>
            </div>
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-gray-400 mb-2">Quick Actions</h3>
              <div className="space-y-2">
                <button
                  onClick={() => setActiveView('context_builder')}
                  className="w-full text-left px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-2"
                >
                  <span className="text-green-400">▸</span>
                  Build AI Context
                </button>
                <button
                  onClick={() => { setActiveView('profiles'); loadProfiles(); }}
                  className="w-full text-left px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-2"
                >
                  <span className="text-blue-400">▸</span>
                  View Semantic Profiles
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeView === 'profiles' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold">Semantic Profiles</h2>
            <button
              onClick={loadProfiles}
              disabled={loadingProfiles}
              className="px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 disabled:opacity-50"
            >
              {loadingProfiles ? 'Loading...' : 'Refresh'}
            </button>
          </div>

          {loadingProfiles ? (
            <div className="flex justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
            </div>
          ) : Object.keys(profiles).length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <p>No semantic profiles generated yet.</p>
              <p className="text-sm mt-2">Profiles are generated on-demand when you request them.</p>
              <button
                onClick={() => { loadProfileDetail('ui_design'); setActiveView('profiles'); }}
                className="mt-4 px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700"
              >
                Generate UI Design Profile
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(profiles).map(([name, profile]) => (
                <div key={name} className="bg-gray-800 rounded-xl p-6 hover:bg-gray-700 transition-colors cursor-pointer"
                     onClick={() => loadProfileDetail(name)}>
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white capitalize">{name.replace('_', ' ')}</h3>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      profile.available ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-400'
                    }`}>
                      {profile.available ? 'Available' : 'Not applicable'}
                    </span>
                  </div>
                  <p className="text-gray-400 text-sm line-clamp-2">
                    {typeof profile === 'object' && profile.visual_style?.overall
                      ? `Visual: ${profile.visual_style.overall.join(', ')}`
                      : typeof profile === 'object' && profile.overview
                      ? profile.overview.substring(0, 100) + '...'
                      : 'Click to view details'}
                  </p>
                  <div className="mt-3 flex gap-2">
                    <span className="px-2 py-1 text-xs bg-gray-700 rounded">
                      {typeof profile === 'object' && profile.visual_style?.overall?.length > 0
                        ? profile.visual_style.overall.length + ' styles'
                        : 'Profile data'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {selectedProfile && profileData && (
            <div className="bg-gray-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold capitalize">{selectedProfile.replace('_', ' ')} Profile</h3>
                <button
                  onClick={() => { setSelectedProfile(null); setProfileData(null); }}
                  className="text-gray-400 hover:text-white"
                >
                  × Close
                </button>
              </div>
              <pre className="bg-gray-900 rounded-lg p-4 overflow-x-auto text-sm text-gray-300 max-h-96 overflow-y-auto">
                {formatJson(profileData)}
              </pre>
            </div>
          )}
        </div>
      )}

      {activeView === 'context_builder' && (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold">AI Context Builder</h2>
          <p className="text-gray-400">
            Describe what you want to do with this video. VideoContext will select the optimal
            context (profiles, evidence, frames) for an LLM to complete your task.
          </p>

          <div className="bg-gray-800 rounded-xl p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                What do you want to do?
              </label>
              <textarea
                value={taskInput}
                onChange={(e) => setTaskInput(e.target.value)}
                rows={4}
                className="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                placeholder="Examples:
• Recreate the website design language
• What was the revenue mentioned?
• How does this application work?
• Describe the animations and transitions
• Summarize the product demo"
              />
            </div>

            <div className="flex items-center gap-4">
              <div className="flex-1">
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Max Tokens: {maxTokens}
                </label>
                <input
                  type="range"
                  min="1000"
                  max="8000"
                  step="500"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>
            </div>

            <button
              onClick={generateContext}
              disabled={loadingContext || !taskInput.trim()}
              className="w-full py-3 px-6 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
            >
              {loadingContext ? 'Generating Context...' : 'Generate Optimized Context'}
            </button>

            <div className="grid grid-cols-2 gap-4 text-sm text-gray-400">
              <div>
                <span className="font-medium text-white">Task Examples:</span>
                <ul className="mt-2 space-y-1 text-gray-500">
                  <li>• "Recreate the website design"</li>
                  <li>• "What was the revenue?"</li>
                  <li>• "How does this app work?"</li>
                  <li>• "Describe the animations"</li>
                </ul>
              </div>
              <div>
                <span className="font-medium text-white">Token Budget:</span>
                <ul className="mt-2 space-y-1 text-gray-500">
                  <li>• 1000-2000: Quick answers</li>
                  <li>• 3000-4000: Standard tasks</li>
                  <li>• 5000+: Complex analysis</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeView === 'context' && contextResult && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold">Generated Context</h2>
            <div className="flex gap-2">
              <button
                onClick={() => copyToClipboard(contextResult.context)}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg"
              >
                Copy Context
              </button>
              <button
                onClick={() => setActiveView('context_builder')}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg"
              >
                Back
              </button>
            </div>
          </div>

          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center gap-4 mb-4 flex-wrap">
              <span className="px-3 py-1 bg-blue-600 rounded-full text-sm font-medium">
                {contextResult.task_type}
              </span>
              <span className="px-3 py-1 bg-gray-700 rounded-full text-sm">
                Confidence: {Math.round(contextResult.confidence * 100)}%
              </span>
              <span className="px-3 py-1 bg-gray-700 rounded-full text-sm">
                Tokens: ~{contextResult.token_estimate}
              </span>
              {Object.keys(contextResult.profiles).length > 0 && (
                <span className="px-3 py-1 bg-purple-600 rounded-full text-sm">
                  Profiles: {Object.keys(contextResult.profiles).join(', ')}
                </span>
              )}
            </div>

            <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto whitespace-pre-wrap text-sm text-gray-300 max-h-[60vh] overflow-y-auto">
              {contextResult.context}
            </div>

            {contextResult.evidence.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-3">Evidence ({contextResult.evidence.length})</h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {contextResult.evidence.map((span, i) => (
                    <div key={i} className="bg-gray-800 rounded-lg p-3 border border-gray-700">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="px-2 py-0.5 text-xs font-medium rounded bg-gray-700 text-gray-200">
                          {span.modality}
                        </span>
                        <span className="font-mono text-xs text-gray-400">{span.timecode}</span>
                        <span className="px-2 py-0.5 text-xs font-medium rounded bg-blue-500/20 text-blue-400">
                          Score: {span.score.toFixed(3)}
                        </span>
                      </div>
                      <p className="text-sm text-gray-200 line-clamp-2">{span.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {contextResult.frames.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-3">Representative Frames ({contextResult.frames.length})</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
                  {contextResult.frames.map((frame: any) => (
                    <div key={frame.id} className="bg-gray-800 rounded-lg p-3 text-center">
                      <p className="font-mono text-xs text-gray-400">{frame.ts?.toFixed(1)}s</p>
                      <p className="text-xs text-gray-500 capitalize">{frame.reason}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

