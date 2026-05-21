import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// Apply persisted theme before first render to avoid flash
const stored = JSON.parse(localStorage.getItem('energy-predictor-store') || '{}')
const theme = stored?.state?.theme || 'dark'
document.documentElement.setAttribute('data-theme', theme)

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
