import { useState } from 'react'

interface VideoUploadProps {
  onUpload: (file: File) => void
}

export function VideoUpload({ onUpload }: VideoUploadProps) {
  const [dragActive, setDragActive] = useState(false)

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    if (e.dataTransfer.files.length > 0) {
      onUpload(e.dataTransfer.files[0])
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUpload(e.target.files[0])
    }
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh]">
      <div
        className={`relative w-full max-w-2xl p-12 border-2 border-dashed rounded-xl transition-colors ${
          dragActive ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-gray-400'
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={(e) => { e.preventDefault(); setDragActive(false) }}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          id="video-upload"
        />
        <label htmlFor="video-upload" className="cursor-pointer">
          <div className="text-center">
            <svg className="mx-auto h-16 w-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <p className="mt-4 text-lg text-gray-200">Drag & drop a video file, or click to select</p>
            <p className="mt-2 text-sm text-gray-500">Supports MP4, MOV, WebM, MKV, AVI</p>
          </div>
        </label>
      </div>
    </div>
  )
}