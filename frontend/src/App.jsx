import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Datasets from './pages/Datasets'
import Configure from './pages/Configure'
import BatchRun from './pages/BatchRun'
import Ensemble from './pages/Ensemble'
import Training from './pages/Training'
import Results from './pages/Results'
import Compare from './pages/Compare'

export default function App() {
  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          style: { background: '#141920', border: '1px solid rgba(255,255,255,0.08)', color: '#fff' },
          success: { iconTheme: { primary: '#22c55e', secondary: '#0f1318' } },
          error: { iconTheme: { primary: '#ef4444', secondary: '#0f1318' } },
        }}
      />
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="datasets" element={<Datasets />} />
          <Route path="configure" element={<Configure />} />
          <Route path="configure/:runId" element={<Configure />} />
          <Route path="training/:runId" element={<Training />} />
          <Route path="results/:runId" element={<Results />} />
          <Route path="batch" element={<BatchRun />} />
        <Route path="ensemble" element={<Ensemble />} />
          <Route path="compare" element={<Compare />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
