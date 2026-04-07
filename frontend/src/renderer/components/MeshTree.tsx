import React from 'react'
import { usePipelineStore } from '../store/pipelineStore'

export function MeshTree() {
  const { patches, selectedPatchId, setSelectedPatch } = usePipelineStore()

  if (patches.length === 0) return null

  return (
    <div style={{ marginTop: '12px' }}>
      <div
        style={{
          fontSize: '11px',
          fontWeight: 700,
          textTransform: 'uppercase',
          color: '#e94560',
          marginBottom: '8px',
          letterSpacing: '1px',
        }}
      >
        Mesh Patches
      </div>
      {patches.map((patch) => (
        <div
          key={patch.id}
          onClick={() => setSelectedPatch(patch.id === selectedPatchId ? null : patch.id)}
          style={{
            padding: '3px 8px',
            fontSize: '11px',
            borderRadius: '3px',
            cursor: 'pointer',
            background: selectedPatchId === patch.id ? '#0f3460' : 'transparent',
            marginBottom: '1px',
          }}
        >
          #{patch.id} — {patch.classification} ({patch.face_count})
        </div>
      ))}
    </div>
  )
}
