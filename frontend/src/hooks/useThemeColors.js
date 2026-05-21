import { useStore } from '../store'

export function useThemeColors() {
  const theme = useStore((s) => s.theme)
  const isDark = theme !== 'light'

  return {
    // Chart
    chartBg:       isDark ? '#0f1318'    : '#ffffff',
    gridStroke:    isDark ? 'rgba(255,255,255,0.07)' : '#e2e5ea',
    axisText:      isDark ? 'rgba(255,255,255,0.45)' : '#6b7280',
    tooltipBg:     isDark ? '#141920'    : '#ffffff',
    tooltipBorder: isDark ? 'rgba(255,255,255,0.13)' : '#b0b7c3',
    tooltipText:   isDark ? '#f0f4f8'    : '#0d1117',
    tooltipLabel:  isDark ? 'rgba(255,255,255,0.5)'  : '#3d4550',
    legendText:    isDark ? 'rgba(255,255,255,0.5)'  : '#3d4550',
    refLine:       isDark ? 'rgba(255,255,255,0.15)' : '#b0b7c3',
    // Series colors
    accentLine:    isDark ? '#00e5ff'    : '#0070b8',
    energyLine:    '#f97316',
    warnLine:      '#eab308',
    // UI colors
    iconBg:        isDark ? 'rgba(0,229,255,0.1)'  : 'rgba(0,112,184,0.1)',
    iconBorder:    isDark ? 'rgba(0,229,255,0.3)'  : 'rgba(0,112,184,0.4)',
    iconColor:     isDark ? '#00e5ff'    : '#0070b8',
    fg:            isDark ? '#f0f4f8'    : '#0d1117',
    fgMuted:       isDark ? 'rgba(240,244,248,0.55)' : '#3d4550',
    fgSubtle:      isDark ? 'rgba(240,244,248,0.3)'  : '#6b7280',
    border:        isDark ? 'rgba(255,255,255,0.07)' : '#d1d5db',
    borderMid:     isDark ? 'rgba(255,255,255,0.13)' : '#b0b7c3',
    bg0:           isDark ? '#0a0d12'    : '#e8eaed',
    bg1:           isDark ? '#0f1318'    : '#ffffff',
    bg2:           isDark ? '#141920'    : '#f1f3f6',
    bg3:           isDark ? '#1a2130'    : '#e2e5ea',
    accentBg:      isDark ? 'rgba(0,229,255,0.09)'  : 'rgba(0,112,184,0.08)',
    shadow:        isDark ? '0 1px 3px rgba(0,0,0,0.4)' : '0 1px 3px rgba(0,0,0,0.12)',
  }
}
