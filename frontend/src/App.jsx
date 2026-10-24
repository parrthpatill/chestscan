import { useEffect, useRef, useState } from 'react'
import './App.css'

const MAX_FILE_SIZE = 5 * 1024 * 1024
const ACCEPTED_TYPES = ['image/jpeg', 'image/png']
const API_URL = (
  import.meta.env.VITE_API_URL || 'http://localhost:8000'
).replace(/\/$/, '')

const DISCLAIMER = 'Screening aid only. Not a medical diagnosis.'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)

  // Release preview URLs when they are replaced or the component unmounts.
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  function selectFile(selectedFile) {
    setError('')
    setResult(null)

    if (!selectedFile) return

    if (!ACCEPTED_TYPES.includes(selectedFile.type)) {
      setFile(null)
      setPreview('')
      setError('Please select a JPG or PNG image.')
      if (inputRef.current) inputRef.current.value = ''
      return
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setFile(null)
      setPreview('')
      setError('The image must be 5 MB or smaller.')
      if (inputRef.current) inputRef.current.value = ''
      return
    }

    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
  }

  function clearFile() {
    setFile(null)
    setPreview('')
    setError('')
    setResult(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  function handleDrop(event) {
    event.preventDefault()
    setIsDragging(false)
    if (!isLoading) selectFile(event.dataTransfer.files?.[0])
  }

  async function handlePredict() {
    if (!file || isLoading) return

    setIsLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json().catch(() => ({}))

      if (!response.ok) {
        const message =
          typeof data.detail === 'string'
            ? data.detail
            : 'The screening request failed. Please try again.'
        throw new Error(message)
      }

      if (
        typeof data.prediction !== 'string' ||
        typeof data.confidence_band !== 'string' ||
        typeof data.confidence !== 'number' ||
        typeof data.probability_pneumonia !== 'number' ||
        !Array.isArray(data.uncertain_band) ||
        data.uncertain_band.length !== 2
      ) {
        throw new Error('The server returned an unexpected response.')
      }

      setResult(data)
    } catch (err) {
      setError(
        err instanceof TypeError
          ? 'Cannot connect to the screening API. Check that the backend is running and the API URL is correct.'
          : err.message || 'Something went wrong. Please try again.',
      )
    } finally {
      setIsLoading(false)
    }
  }

  const isUncertain = result?.confidence_band === 'Uncertain'

  return (
    <main className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="ChestScan home">
          <span className="brand-mark" aria-hidden="true">C</span>
          <span>ChestScan</span>
        </a>
        <span className="header-label">Chest X-ray screening</span>
      </header>

      <section className="main-content">
        <div className="intro">
          <p className="eyebrow">AI-ASSISTED SCREENING</p>
          <h1>Chest X-ray screening</h1>
          <p className="intro-copy">
            Upload a chest X-ray image for a model-generated screening result.
          </p>
        </div>

        <section
          className={`upload-card ${isDragging ? 'is-dragging' : ''}`}
          onDragOver={(event) => {
            event.preventDefault()
            if (!isLoading) setIsDragging(true)
          }}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget)) {
              setIsDragging(false)
            }
          }}
          onDrop={handleDrop}
          aria-label="Chest X-ray upload"
        >
          {!file ? (
            <>
              <div className="upload-icon" aria-hidden="true">
                <svg viewBox="0 0 48 48" fill="none">
                  <path
                    d="M24 31V9m0 0-8 8m8-8 8 8M10 29v9a3 3 0 0 0 3 3h22a3 3 0 0 0 3-3v-9"
                    stroke="currentColor"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>

              <h2>Upload your X-ray</h2>
              <p className="upload-description">
                Drag and drop your image here, or choose a file from your device.
              </p>

              <button
                type="button"
                className="primary-button"
                onClick={() => inputRef.current?.click()}
                disabled={isLoading}
              >
                Choose image
              </button>

              <p className="file-hint">JPG or PNG · Maximum 5 MB</p>
            </>
          ) : (
            <div className="preview-panel">
              <div className="preview-heading">
                <div>
                  <p className="eyebrow">IMAGE SELECTED</p>
                  <h2>Review your image</h2>
                </div>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={clearFile}
                  disabled={isLoading}
                >
                  Remove image
                </button>
              </div>

              <div className="image-preview">
                <img src={preview} alt="Preview of selected chest X-ray" />
              </div>

              <p className="file-name">{file.name}</p>
              <p className="file-hint">
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </p>

              <button
                type="button"
                className="primary-button"
                onClick={handlePredict}
                disabled={isLoading}
              >
                {isLoading ? 'Analysing image…' : 'Run screening'}
              </button>

              {isLoading && (
                <p role="status" aria-live="polite">
                  Processing your image. Please wait.
                </p>
              )}
            </div>
          )}

          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept="image/jpeg,image/png"
            aria-label="Choose a chest X-ray image"
            onChange={(event) => selectFile(event.target.files?.[0])}
          />

          {error && (
            <p className="error-message" role="alert">
              {error}
            </p>
          )}
        </section>

        {result && (
          <section
            className={`result-card ${isUncertain ? 'result-uncertain' : ''}`}
            aria-live="polite"
            aria-labelledby="result-heading"
          >
            <p className="eyebrow">SCREENING RESULT</p>
            <h2 id="result-heading">
              {isUncertain
                ? 'Result unclear — recommend manual review'
                : result.prediction === 'PNEUMONIA'
                  ? 'Pneumonia pattern flagged'
                  : 'Normal pattern predicted'}
            </h2>

            <p>
              Triage status: <strong>{result.confidence_band}</strong>
            </p>

            <div className="result-metric">
              <span>Model confidence</span>
              <strong>
                {(result.confidence * 100).toFixed(1)}%
              </strong>
            </div>

            <div
              className="confidence-track"
              role="progressbar"
              aria-label="Model confidence"
              aria-valuenow={Math.round(result.confidence * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className="confidence-fill"
                style={{
                  width: `${Math.min(100, Math.max(0, result.confidence * 100))}%`,
                }}
              />
            </div>

            <p>
              Pneumonia probability:{' '}
              {(result.probability_pneumonia * 100).toFixed(1)}%
            </p>
            <p>
              Saved decision threshold:{' '}
              {result.threshold_used.toFixed(6)}
            </p>
            <p>
              Uncertain band: {result.uncertain_band[0].toFixed(6)}
              {' – '}
              {result.uncertain_band[1].toFixed(6)}
            </p>

            {typeof result.inference_ms === 'number' && (
              <p>Inference time: {result.inference_ms.toFixed(2)} ms</p>
            )}
          </section>
        )}

        <p className="disclaimer">{DISCLAIMER}</p>
      </section>

      <footer className="site-footer">
        <span>ChestScan</span>
        <span>For screening support only</span>
      </footer>
    </main>
  )
}

export default App
