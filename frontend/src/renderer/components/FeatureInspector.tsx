import React from 'react'
import { usePipelineStore } from '../store/pipelineStore'

/* ── colour maps ── */

const typeColors: Record<string, string> = {
  plane: '#3b82f6',
  cylinder: '#22c55e',
  cone: '#f59e0b',
  unknown: '#555',
}

const typeLabels: Record<string, string> = {
  plane: 'Plane',
  cylinder: 'Cylinder',
  cone: 'Cone',
  unknown: 'Unknown',
}

const confidenceBadge: Record<string, { bg: string; fg: string }> = {
  HIGH:     { bg: 'rgba(34,197,94,0.15)',  fg: '#22c55e' },
  MEDIUM:   { bg: 'rgba(245,158,11,0.15)', fg: '#f59e0b' },
  LOW:      { bg: 'rgba(239,68,68,0.15)',  fg: '#ef4444' },
  REJECTED: { bg: 'rgba(239,68,68,0.2)',   fg: '#ef4444' },
}

/* ── inline styles ── */

const s = {
  panel: {
    position: 'absolute' as const,
    right: 16,
    top: 16,
    width: 280,
    maxHeight: 'calc(100vh - 140px)',
    overflowY: 'auto' as const,
    background: 'rgba(15,15,15,0.92)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
    border: '1px solid #2a2a2a',
    borderRadius: 12,
    boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
    padding: 16,
    zIndex: 100,
    color: '#ccc',
  },
  header: {
    display: 'flex' as const,
    justifyContent: 'space-between' as const,
    alignItems: 'center' as const,
    marginBottom: 14,
  },
  headerTitle: {
    fontSize: 12,
    fontWeight: 600,
    color: '#f0f0f0',
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    color: '#666',
    fontSize: 18,
    cursor: 'pointer',
    padding: 0,
    lineHeight: 1,
  },
  sectionTitle: {
    fontSize: 10,
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: 1.5,
    color: '#666',
    marginBottom: 10,
  },
  statRow: {
    display: 'flex' as const,
    justifyContent: 'space-between' as const,
    alignItems: 'center' as const,
    fontSize: 12,
    padding: '2px 0',
    color: '#ccc',
  },
  label: { color: '#888' },
  mono: { fontFamily: 'monospace', color: '#999' },
  monoValue: { fontFamily: 'monospace', color: '#e0e0e0' },
} as const

/* ── component ── */

export function FeatureInspector({ onClose }: { onClose: () => void }) {
  const { intentSummary, intentRegions } = usePipelineStore()

  const hasData = intentSummary || (intentRegions && intentRegions.length > 0)

  return (
    <div style={s.panel} className="custom-scrollbar">
      {/* ── Header ── */}
      <div style={s.header}>
        <span style={s.headerTitle}>Region Inspector</span>
        <button
          style={s.closeBtn}
          onClick={onClose}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#f0f0f0' }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#666' }}
          aria-label="Close"
        >
          &times;
        </button>
      </div>

      {/* ── No data ── */}
      {!hasData && (
        <div style={{ fontSize: 12, color: '#555', textAlign: 'center', padding: '32px 8px' }}>
          Run intent segmentation to inspect detected regions.
        </div>
      )}

      {/* ── Summary ── */}
      {intentSummary && (
        <div style={{ marginBottom: 16 }}>
          <div style={s.sectionTitle}>Summary</div>

          {/* Total regions */}
          <div style={s.statRow}>
            <span style={s.label}>Total Regions</span>
            <span style={s.monoValue}>{intentSummary.n_regions ?? '\u2014'}</span>
          </div>

          {/* Type breakdown */}
          <div style={{ margin: '8px 0' }}>
            {(['plane', 'cylinder', 'cone', 'unknown'] as const).map((type) => {
              const familyKey =
                type === 'plane' ? 'n_plane'
                : type === 'cylinder' ? 'n_cyl'
                : type === 'cone' ? 'n_cone'
                : null
              const highKey =
                type === 'plane' ? 'high_planes'
                : type === 'cylinder' ? 'high_cylinders'
                : type === 'cone' ? 'high_cones'
                : 'unknowns'
              const count =
                familyKey && intentSummary.families
                  ? intentSummary.families[familyKey] ?? 0
                  : intentSummary[highKey] ?? 0
              if (count === 0 && type === 'unknown' && !intentSummary.unknowns) return null
              return (
                <div
                  key={type}
                  style={{ display: 'flex', alignItems: 'center', fontSize: 12, padding: '2px 0', color: '#ccc' }}
                >
                  <span
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: '50%',
                      background: typeColors[type],
                      marginRight: 8,
                      flexShrink: 0,
                    }}
                  />
                  <span style={{ flex: 1 }}>{typeLabels[type]}</span>
                  <span style={s.monoValue}>{count}</span>
                </div>
              )
            })}
          </div>

          {/* Area explained */}
          {intentSummary.area_explained_pct != null && (
            <div style={s.statRow}>
              <span style={s.label}>Area Explained</span>
              <span style={{ fontFamily: 'monospace', color: '#22c55e', fontWeight: 600 }}>
                {intentSummary.area_explained_pct.toFixed(1)}%
              </span>
            </div>
          )}

          {/* RMSE per type */}
          {intentSummary.rmse_plane != null && (
            <div style={s.statRow}>
              <span style={s.label}>RMSE Plane</span>
              <span style={s.mono}>{intentSummary.rmse_plane.toFixed(4)}</span>
            </div>
          )}
          {intentSummary.rmse_cyl != null && (
            <div style={s.statRow}>
              <span style={s.label}>RMSE Cylinder</span>
              <span style={s.mono}>{intentSummary.rmse_cyl.toFixed(4)}</span>
            </div>
          )}
          {intentSummary.rmse_cone != null && (
            <div style={s.statRow}>
              <span style={s.label}>RMSE Cone</span>
              <span style={s.mono}>{intentSummary.rmse_cone.toFixed(4)}</span>
            </div>
          )}

          {/* Elapsed time */}
          {intentSummary.elapsed_s != null && (
            <div style={s.statRow}>
              <span style={s.label}>Elapsed</span>
              <span style={s.mono}>{intentSummary.elapsed_s.toFixed(2)}s</span>
            </div>
          )}
        </div>
      )}

      {/* ── Region List ── */}
      {intentRegions && intentRegions.length > 0 && (
        <div>
          <div style={s.sectionTitle}>Regions ({intentRegions.length})</div>

          {intentRegions.map((region) => {
            const pType = region.primitive_type || 'unknown'
            const color = typeColors[pType] || typeColors.unknown
            const badge = confidenceBadge[region.confidence_class] || { bg: 'rgba(100,100,100,0.15)', fg: '#888' }

            return (
              <div
                key={region.id}
                style={{
                  padding: '8px 10px',
                  borderRadius: 6,
                  marginBottom: 4,
                  background: 'rgba(255,255,255,0.03)',
                  transition: 'background 0.15s',
                  cursor: 'default',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.06)' }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.03)' }}
              >
                {/* Top row: dot + ID | type | confidence pill */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                  {/* Colored dot */}
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      background: color,
                      flexShrink: 0,
                    }}
                  />
                  {/* Region ID */}
                  <span style={{ fontFamily: 'monospace', color: '#666', minWidth: 20 }}>
                    {region.id}
                  </span>
                  {/* Type label */}
                  <span style={{ color, flex: 1 }}>
                    {typeLabels[pType] || pType}
                  </span>
                  {/* Confidence pill */}
                  <span
                    style={{
                      fontSize: 9,
                      fontWeight: 700,
                      padding: '2px 6px',
                      borderRadius: 4,
                      background: badge.bg,
                      color: badge.fg,
                      textTransform: 'uppercase',
                      letterSpacing: 0.5,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {region.confidence_class || '?'}
                  </span>
                </div>

                {/* RMSE below */}
                {region.rmse != null && (
                  <div style={{ fontFamily: 'monospace', color: '#666', fontSize: 10, marginTop: 4, paddingLeft: 14 }}>
                    RMSE {region.rmse.toFixed(4)}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
