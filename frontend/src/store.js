import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useStore = create(
  persist(
    (set, get) => ({
      // Datasets
      datasets: [],
      datasetsLoaded: false,
      setDatasets: (datasets) => set({ datasets, datasetsLoaded: true }),
      addDataset: (ds) => set((s) => ({ datasets: [ds, ...s.datasets] })),
      removeDataset: (id) => set((s) => ({ datasets: s.datasets.filter((d) => d.id !== id) })),

      // Runs
      runs: [],
      runsLoaded: false,
      setRuns: (runs) => set({ runs, runsLoaded: true }),
      addRun: (run) => set((s) => ({ runs: [run, ...s.runs] })),
      updateRun: (id, patch) =>
        set((s) => ({ runs: s.runs.map((r) => (r.id === id ? { ...r, ...patch } : r)) })),
      removeRun: (id) => set((s) => ({ runs: s.runs.filter((r) => r.id !== id) })),

      // Compare selection
      compareIds: [],
      toggleCompare: (id) =>
        set((s) => ({
          compareIds: s.compareIds.includes(id)
            ? s.compareIds.filter((i) => i !== id)
            : [...s.compareIds, id],
        })),
      clearCompare: () => set({ compareIds: [] }),

      // Theme
      theme: 'dark',
      toggleTheme: () =>
        set((s) => {
          const next = s.theme === 'dark' ? 'light' : 'dark'
          document.documentElement.setAttribute('data-theme', next)
          return { theme: next }
        }),
      initTheme: () => {
        const theme = get().theme || 'dark'
        document.documentElement.setAttribute('data-theme', theme)
      },
    }),
    {
      name: 'energy-predictor-store',
      partialize: (s) => ({ theme: s.theme }), // only persist theme
    }
  )
)
