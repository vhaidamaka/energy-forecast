import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

// ── Datasets ──────────────────────────────────────────────────────────────────
export const datasetsApi = {
  upload: (file, name, description = '') => {
    const form = new FormData()
    form.append('file', file)
    form.append('name', name)
    form.append('description', description)
    return api.post('/datasets/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  list: () => api.get('/datasets/'),
  get: (id) => api.get(`/datasets/${id}`),
  preview: (id, rows = 200) => api.get(`/datasets/${id}/preview`, { params: { rows } }),
  delete: (id) => api.delete(`/datasets/${id}`),
  update: (id, data) => api.put(`/datasets/${id}`, null, { params: data }),
}

// ── Runs ──────────────────────────────────────────────────────────────────────
export const runsApi = {
  create: (payload) => api.post('/runs/', payload),
  start: (id) => api.post(`/runs/${id}/start`),
  list: () => api.get('/runs/'),
  get: (id) => api.get(`/runs/${id}`),
  delete: (id) => api.delete(`/runs/${id}`),
  compare: (ids) => api.get('/runs/compare/metrics', { params: { run_ids: ids.join(',') } }),
}

// ── Export ────────────────────────────────────────────────────────────────────
export const batchApi = {
  start: (dataset_id, models, horizon_days, name_prefix) =>
    api.post('/runs/batch', { dataset_id, models, horizon_days, name_prefix }),
}

export const exportApi = {
  csvUrl: (id) => `/api/export/${id}/csv`,
  excelUrl: (id) => `/api/export/${id}/excel`,
}

// ── Models ────────────────────────────────────────────────────────────────────
export const modelsApi = {
  defaults: () => api.get('/models/defaults'),
}

export default api
