import React from 'react'
import { usePipelineStore } from '../store/pipelineStore'

const styles = {
  panel: {
    width: '240px',
    minWidth: '240px',
    background: '#16213e',
    borderLeft: '1px solid #0f3460',
    overflowY: 'auto' as const,
    padding: '12px',
  },
  heading: {
    fontSize: '11px',
    fontWeight: 700,
    textTransform: 'uppercase' as const,
    color: '#e94560',
    marginBottom: '12px',
    letterSpacing: '1px',
  },
  section: {
    marginBottom: '16px',
  },
  sectionTitle: {
    fontSize: '11px',
    fontWeight: 600,
    color: '#4ecca3',
    marginBottom: '6px',
  },
  row: {
    display: 'flex',
    justifyContent: 'space-between' as const,
    fontSize: '11px',
    padding: '2px 0',
    color: '#ccc',
  },
  label: { color: '#888' },
  value: { color: '#e0e0e0', fontFamily: 'monospace' },
  patchItem: {
    padding: '4px 8px',
    fontSize: '11px',
    borderRadius: '3px',
    marginBottom: '2px',
    cursor: 'pointer',
    transition: 'background 0.1s',
  },
}

const typeColors: Record<string, string> = {
  plane: '#4ecca3',
  planar: '#4ecca3',
  cylinder: '#e94560',
  cylindrical: '#e94560',
  sphere: '#f5a623',
  spherical: '#f5a623',
  bspline: '#54a0ff',
  fillet: '#f368e0',
  curved: '#7b68ee',
  conical: '#ff9ff3',
  freeform: '#888',
}

export function FeatureInspector() {
  const { selectedPatchId, setSelectedPatch, primitives, patches, features } = usePipelineStore()

  const selectedPrimitive = primitives.find((p) => p.patch_id === selectedPatchId)
  const selectedPatch = patches.find((p) => p.id === selectedPatchId)
  const relatedFeatures = features.filter(
    (f) => f.patch_id === selectedPatchId || f.adjacent_patches?.includes(selectedPatchId ?? -1)
  )

  return (
    <div style={styles.panel}>
      <div style={styles.heading}>Inspector</div>

      {selectedPrimitive ? (
        <>
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Primitive</div>
            <div style={styles.row}>
              <span style={styles.label}>Type</span>
              <span style={{ ...styles.value, color: typeColors[selectedPrimitive.type] || '#fff' }}>
                {selectedPrimitive.type}
              </span>
            </div>
            <div style={styles.row}>
              <span style={styles.label}>Inlier Ratio</span>
              <span style={styles.value}>{(selectedPrimitive.inlier_ratio * 100).toFixed(1)}%</span>
            </div>
            <div style={styles.row}>
              <span style={styles.label}>Faces</span>
              <span style={styles.value}>{selectedPrimitive.face_count}</span>
            </div>
            {selectedPrimitive.radius != null && (
              <div style={styles.row}>
                <span style={styles.label}>Radius</span>
                <span style={styles.value}>{selectedPrimitive.radius.toFixed(3)}</span>
              </div>
            )}
            {selectedPrimitive.normal && (
              <div style={styles.row}>
                <span style={styles.label}>Normal</span>
                <span style={styles.value}>
                  [{selectedPrimitive.normal.map((v) => v.toFixed(2)).join(', ')}]
                </span>
              </div>
            )}
            {(selectedPrimitive as any).surface_class && (
              <div style={styles.row}>
                <span style={styles.label}>Surface</span>
                <span style={styles.value}>{(selectedPrimitive as any).surface_class}</span>
              </div>
            )}
            {(selectedPrimitive as any).rmse != null && (
              <div style={styles.row}>
                <span style={styles.label}>RMSE</span>
                <span style={styles.value}>{(selectedPrimitive as any).rmse.toFixed(4)}</span>
              </div>
            )}
            {selectedPrimitive.axis && (
              <div style={styles.row}>
                <span style={styles.label}>Axis</span>
                <span style={styles.value}>
                  [{selectedPrimitive.axis.map((v) => v.toFixed(2)).join(', ')}]
                </span>
              </div>
            )}
          </div>

          {relatedFeatures.length > 0 && (
            <div style={styles.section}>
              <div style={styles.sectionTitle}>Features</div>
              {relatedFeatures.map((f, i) => (
                <div key={i} style={styles.row}>
                  <span style={styles.label}>{f.type}</span>
                  <span style={styles.value}>
                    {f.estimated_radius ? `R=${f.estimated_radius.toFixed(2)}` : ''}
                    {f.angle_degrees ? `${f.angle_degrees.toFixed(1)}\u00b0` : ''}
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      ) : (
        <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '40px' }}>
          Select a patch to inspect
        </div>
      )}

      {patches.length > 0 && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Patches ({patches.length})</div>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {patches.map((patch) => {
              const prim = primitives.find((p) => p.patch_id === patch.id)
              return (
                <div
                  key={patch.id}
                  style={{
                    ...styles.patchItem,
                    background: selectedPatchId === patch.id ? '#0f3460' : 'transparent',
                  }}
                  onClick={() => setSelectedPatch(patch.id === selectedPatchId ? null : patch.id)}
                >
                  <span style={{ color: typeColors[prim?.type || patch.classification] || '#888' }}>
                    {prim?.type || patch.classification}
                  </span>
                  {' '}#{patch.id} ({patch.face_count} faces)
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
