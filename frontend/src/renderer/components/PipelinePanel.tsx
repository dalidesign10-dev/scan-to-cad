import React, { useState } from 'react'
import { usePipelineStore } from '../store/pipelineStore'

const stageOrder = ['idle', 'loaded', 'preprocessed', 'segmented', 'fitted', 'features_detected', 'exported']

function getStepStatus(stage: string, requiredStage: string): 'done' | 'active' | 'pending' {
  const currentIdx = stageOrder.indexOf(stage)
  const requiredIdx = stageOrder.indexOf(requiredStage)
  if (currentIdx > requiredIdx) return 'done'
  if (currentIdx === requiredIdx) return 'active'
  return 'pending'
}

const accentColor: Record<string, string> = {
  done: '#22c55e',
  active: '#06b6d4',
  pending: 'transparent',
}

function StepCard({
  title,
  requiredStage,
  children,
  last,
}: {
  title: string
  requiredStage: string
  children: React.ReactNode
  last?: boolean
}) {
  const stage = usePipelineStore((s) => s.stage)
  const status = getStepStatus(stage, requiredStage)

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <div
        style={{
          background: '#1a1a1a',
          border: '1px solid #2a2a2a',
          borderRadius: '8px',
          padding: '14px',
          borderLeft: `3px solid ${accentColor[status]}`,
        }}
      >
        <div style={{
          fontSize: '12px',
          fontWeight: 600,
          color: status === 'done' ? '#22c55e' : status === 'active' ? '#f0f0f0' : '#666',
          marginBottom: children ? '8px' : 0,
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}>
          {status === 'done' && (
            <span style={{ fontSize: '11px' }}>&#10003;</span>
          )}
          {title}
        </div>
        {children}
      </div>
      {/* Connector line */}
      {!last && (
        <div style={{
          width: '2px',
          height: '8px',
          background: '#2a2a2a',
          marginLeft: '20px',
        }} />
      )}
    </div>
  )
}

export function PipelinePanel({ collapsed, onToggleCollapse }: { collapsed: boolean; onToggleCollapse: () => void }) {
  const {
    stage, loading,
    runCleanup,
    meshInfo,
    runIntentSegmentation, intentSummary, intentRegions,
    showIntentRegionColors, toggleIntentRegionColors,
    showIntentGizmos, toggleIntentGizmos,
    intentColorMode, setIntentColorMode,
    runLiveReconstruction, stopReconstruction, reconstructionEvents,
    reconstructionRunning, reconstructionProgress,
    runAIClassify, classifyResult,
    runSnapToSurfaces, snapResult, showSnappedMesh, toggleSnappedMesh,
    runDeviationAnalysis, deviationResult, showDeviationHeatmap, toggleDeviationHeatmap,
    runConstructFaces, trimResult,
    runExportStep, stepResult, downloadStep,
  } = usePipelineStore()

  const [poissonDepth, setPoissonDepth] = useState(10)
  const [sampleCount, setSampleCount] = useState(400000)
  const [intentTargetFaces, setIntentTargetFaces] = useState(80000)
  const [intentMinRegionFaces, setIntentMinRegionFaces] = useState(20)
  const [intentGrowthMode, setIntentGrowthMode] = useState<'dihedral' | 'fit_driven'>('fit_driven')
  const [showCleanupAdv, setShowCleanupAdv] = useState(false)

  const inputStyle: React.CSSProperties = {
    width: '64px',
    padding: '4px 6px',
    background: '#1a1a1a',
    border: '1px solid #2a2a2a',
    borderRadius: '6px',
    color: '#f0f0f0',
    fontSize: '11px',
  }

  const paramRow: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: '11px',
    color: '#999',
    marginTop: '6px',
  }

  const actionBtn = (bg: string): React.CSSProperties => ({
    width: '100%',
    padding: '7px 12px',
    border: 'none',
    borderRadius: '8px',
    background: bg,
    color: '#0a0a0a',
    fontSize: '12px',
    fontWeight: 500,
    cursor: 'pointer',
    marginTop: '8px',
    transition: 'opacity 0.15s ease',
  })

  // Collapsed state: thin strip with icon
  if (collapsed) {
    return (
      <div style={{
        width: '48px',
        minWidth: '48px',
        background: '#0f0f0f',
        borderRight: '1px solid #2a2a2a',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        paddingTop: '12px',
        gap: '12px',
      }}>
        <button
          onClick={onToggleCollapse}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#666',
            fontSize: '14px',
            cursor: 'pointer',
            padding: '4px',
          }}
          title="Expand pipeline"
        >
          &#x276F;
        </button>
        <span style={{
          writingMode: 'vertical-rl',
          textOrientation: 'mixed',
          fontSize: '10px',
          textTransform: 'uppercase',
          letterSpacing: '2px',
          color: '#666',
          fontWeight: 500,
        }}>
          Pipeline
        </span>
      </div>
    )
  }

  return (
    <div style={{
      width: '240px',
      minWidth: '240px',
      background: '#0f0f0f',
      borderRight: '1px solid #2a2a2a',
      overflowY: 'auto',
      padding: '12px',
      display: 'flex',
      flexDirection: 'column',
      gap: '0px',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '12px',
      }}>
        <span style={{
          fontSize: '10px',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '2px',
          color: '#666',
        }}>
          Pipeline
        </span>
        <button
          onClick={onToggleCollapse}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#666',
            fontSize: '14px',
            cursor: 'pointer',
            padding: '2px 4px',
          }}
          title="Collapse pipeline"
        >
          &#x276E;
        </button>
      </div>

      {/* Step 1: Load Mesh */}
      <StepCard title="1. Load Mesh" requiredStage="idle">
        {meshInfo && (
          <div style={{ fontSize: '11px', color: '#888', lineHeight: '1.5' }}>
            <span style={{ color: '#f0f0f0' }}>{meshInfo.vertices.toLocaleString()}</span> verts
            {' / '}
            <span style={{ color: '#f0f0f0' }}>{meshInfo.faces.toLocaleString()}</span> faces
            {meshInfo.is_watertight && (
              <span style={{ color: '#22c55e', marginLeft: '6px' }}>watertight</span>
            )}
          </div>
        )}
      </StepCard>

      {/* Step 1b: Cleanup */}
      <StepCard title="1b. Cleanup (Poisson)" requiredStage="loaded">
        <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px', lineHeight: '1.4' }}>
          Watertight remesh via screened Poisson reconstruction.
        </div>
        <button
          style={{ ...actionBtn('#22c55e'), opacity: meshInfo && !loading ? 1 : 0.4 }}
          onClick={() => runCleanup({ poisson_depth: poissonDepth, sample_count: sampleCount })}
          disabled={!meshInfo || loading}
        >
          {loading ? 'Cleaning...' : 'Run Cleanup'}
        </button>
        <div
          style={{
            fontSize: '10px',
            color: '#666',
            marginTop: '8px',
            cursor: 'pointer',
            userSelect: 'none',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
          }}
          onClick={() => setShowCleanupAdv((v) => !v)}
        >
          <span style={{ fontSize: '8px' }}>{showCleanupAdv ? '\u25BE' : '\u25B8'}</span>
          Advanced
        </div>
        {showCleanupAdv && (
          <div style={{ marginTop: '6px' }}>
            <div style={paramRow}>
              <span>Poisson depth</span>
              <input
                type="number"
                style={inputStyle}
                min={7}
                max={12}
                value={poissonDepth}
                onChange={(e) => setPoissonDepth(Number(e.target.value))}
              />
            </div>
            <div style={paramRow}>
              <span>Sample count</span>
              <input
                type="number"
                style={inputStyle}
                min={50000}
                max={2000000}
                step={50000}
                value={sampleCount}
                onChange={(e) => setSampleCount(Number(e.target.value))}
              />
            </div>
          </div>
        )}
      </StepCard>

      {/* Step E0: Intent Reconstruction */}
      <StepCard title="E0. Intent Reconstruction" requiredStage="preprocessed">
        <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px', lineHeight: '1.4' }}>
          Proxy decimate, hybrid sharp-edge boundaries, region grow, plane/cylinder/unknown fits.
        </div>
        <div style={paramRow}>
          <span>Proxy target faces</span>
          <input
            type="number"
            style={inputStyle}
            min={5000}
            max={100000}
            step={1000}
            value={intentTargetFaces}
            onChange={(e) => setIntentTargetFaces(Number(e.target.value))}
          />
        </div>
        <div style={paramRow}>
          <span>Min region faces</span>
          <input
            type="number"
            style={inputStyle}
            min={4}
            max={500}
            value={intentMinRegionFaces}
            onChange={(e) => setIntentMinRegionFaces(Number(e.target.value))}
          />
        </div>
        <div style={paramRow}>
          <span>Growth mode</span>
          <select
            style={{ ...inputStyle, width: '80px' }}
            value={intentGrowthMode}
            onChange={(e) =>
              setIntentGrowthMode(e.target.value as 'dihedral' | 'fit_driven')
            }
          >
            <option value="dihedral">dihedral</option>
            <option value="fit_driven">fit_driven</option>
          </select>
        </div>
        <button
          style={{ ...actionBtn('#06b6d4'), opacity: meshInfo && !loading ? 1 : 0.4 }}
          onClick={() =>
            runIntentSegmentation({
              target_proxy_faces: intentTargetFaces,
              min_region_faces: intentMinRegionFaces,
              growth_mode: intentGrowthMode,
            })
          }
          disabled={!meshInfo || loading}
        >
          {loading ? 'Running E0...' : 'Run E0 Intent'}
        </button>

        {/* Intent Summary */}
        {intentSummary && (
          <div style={{
            fontSize: '11px',
            color: '#888',
            marginTop: '10px',
            lineHeight: '1.6',
            background: '#141414',
            borderRadius: '6px',
            padding: '10px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span><span style={{ color: '#f0f0f0' }}>{intentSummary.n_regions}</span> regions</span>
              <span><span style={{ color: '#f0f0f0' }}>{intentSummary.n_boundaries}</span> bdys</span>
            </div>
            <div style={{ marginTop: '4px' }}>
              <span style={{ color: '#22c55e' }}>{intentSummary.n_high_plane_fits}</span> planes
              {' / '}
              <span style={{ color: '#06b6d4' }}>{intentSummary.n_high_cylinder_fits}</span> cyl
              {' / '}
              <span style={{ color: '#f59e0b' }}>{intentSummary.n_high_cone_fits ?? 0}</span> cone
              {' / '}
              <span style={{ color: '#666' }}>{intentSummary.n_unknown_regions}</span> unk
            </div>
            <div style={{ marginTop: '4px', color: '#666' }}>
              families:
              {' '}<span style={{ color: '#22c55e' }}>{intentSummary.n_plane_families ?? 0}</span> plane
              {' '}<span style={{ color: '#06b6d4' }}>{intentSummary.n_cylinder_families ?? 0}</span> cyl
              {' '}<span style={{ color: '#f59e0b' }}>{intentSummary.n_cone_families ?? 0}</span> cone
            </div>
            <div style={{ marginTop: '4px' }}>
              area-explained:
              {' '}<span style={{ color: '#fbbf24', fontWeight: 600 }}>
                {(intentSummary.explained_area_high_pct ?? 0).toFixed(1)}%
              </span>
            </div>
            <div style={{ marginTop: '4px', color: '#555', fontSize: '10px' }}>
              rmse: plane {intentSummary.mean_rmse_plane?.toFixed(3) ?? '0'}
              {' '} cyl {intentSummary.mean_rmse_cylinder?.toFixed(3) ?? '0'}
              {' '} cone {intentSummary.mean_rmse_cone?.toFixed(3) ?? '0'}
              {' '} {(intentSummary.elapsed_sec ?? 0).toFixed(1)}s
            </div>
            <div style={{ color: '#555', fontSize: '10px' }}>
              proxy {intentSummary.proxy?.proxy_faces?.toLocaleString() ?? '?'}
              {' / '}{intentSummary.proxy?.full_faces?.toLocaleString() ?? '?'} faces
            </div>
          </div>
        )}

        {/* Region toggles */}
        {intentRegions && intentRegions.length > 0 && (
          <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{
              fontSize: '11px',
              color: '#999',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
            }}>
              <input
                type="checkbox"
                checked={showIntentRegionColors}
                onChange={toggleIntentRegionColors}
                style={{ accentColor: '#06b6d4' }}
              />
              Tint by region / type / confidence
            </label>
            <label style={{
              fontSize: '11px',
              color: '#999',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
            }}>
              <input
                type="checkbox"
                checked={showIntentGizmos}
                onChange={toggleIntentGizmos}
                style={{ accentColor: '#06b6d4' }}
              />
              Show axes / normals / sharp edges
            </label>
            <div style={paramRow}>
              <span>Color / gizmo by</span>
              <select
                style={{ ...inputStyle, width: '72px' }}
                value={intentColorMode}
                onChange={(e) =>
                  setIntentColorMode(e.target.value as 'region' | 'family')
                }
              >
                <option value="region">region</option>
                <option value="family">family</option>
              </select>
            </div>
          </div>
        )}
      </StepCard>

      {/* Step E1: Snap to Surfaces */}
      {/* Live AI Reconstruction */}
      <StepCard title="AI Reconstruct" requiredStage="preprocessed">
        <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px', lineHeight: '1.4' }}>
          AI processes each region: classify → build clean surface → check deviation. Watch it work in real-time.
        </div>
        {!reconstructionRunning ? (
          <button
            style={{ ...actionBtn('#8b5cf6'), opacity: intentSummary && !loading ? 1 : 0.4 }}
            onClick={() => runLiveReconstruction()}
            disabled={!intentSummary || loading}
          >
            Start AI Reconstruction
          </button>
        ) : (
          <button
            style={{ ...actionBtn('#ef4444') }}
            onClick={() => stopReconstruction()}
          >
            Stop
          </button>
        )}
        {reconstructionProgress && (
          <div style={{
            fontSize: '11px', color: '#888', marginTop: '8px',
          }}>
            <div style={{
              width: '100%', height: '4px', background: '#2a2a2a',
              borderRadius: '2px', overflow: 'hidden', marginBottom: '6px',
            }}>
              <div style={{
                width: `${(reconstructionProgress.step / reconstructionProgress.total) * 100}%`,
                height: '100%', background: '#8b5cf6', borderRadius: '2px',
                transition: 'width 0.3s ease',
              }} />
            </div>
            {reconstructionProgress.step} / {reconstructionProgress.total} regions
          </div>
        )}
        {reconstructionEvents.length > 0 && (
          <div style={{
            fontSize: '10px', color: '#888', marginTop: '8px',
            maxHeight: '200px', overflowY: 'auto',
            background: '#141414', borderRadius: '6px', padding: '8px',
          }}>
            {reconstructionEvents.slice(-8).map((ev, i) => {
              const statusColor = ev.status === 'built' ? '#22c55e'
                : ev.status === 'skipped' ? '#666' : '#f59e0b'
              return (
                <div key={i} style={{ padding: '2px 0', borderBottom: '1px solid #1a1a1a' }}>
                  <span style={{ color: statusColor }}>R{ev.region_id}</span>
                  {' '}
                  <span style={{ color: '#8b5cf6' }}>{ev.classification}</span>
                  {' '}
                  <span style={{ color: '#666' }}>({ev.confidence})</span>
                  {ev.deviation && (
                    <span style={{ color: '#999', marginLeft: '4px' }}>
                      dev={ev.deviation.mean.toFixed(3)}
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </StepCard>

      {/* AI Classification */}
      <StepCard title="AI Classify" requiredStage="preprocessed">
        <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px', lineHeight: '1.4' }}>
          Claude AI analyzes each region and classifies as PLANE, CYLINDER, CONE, FILLET, CHAMFER, or FREEFORM.
        </div>
        <button
          style={{ ...actionBtn('#8b5cf6'), opacity: intentSummary && !loading ? 1 : 0.4 }}
          onClick={() => runAIClassify()}
          disabled={!intentSummary || loading}
        >
          {loading ? 'Classifying...' : 'Run AI Classification'}
        </button>
        {classifyResult && (
          <div style={{
            fontSize: '11px', color: '#888', marginTop: '10px', lineHeight: '1.6',
            background: '#141414', borderRadius: '6px', padding: '10px',
          }}>
            <div>
              <span style={{ color: '#8b5cf6', fontWeight: 600 }}>{classifyResult.n_classified}</span> classified
              {classifyResult.n_changed > 0 && (
                <span style={{ color: '#f59e0b' }}> ({classifyResult.n_changed} reclassified)</span>
              )}
            </div>
            <div style={{ marginTop: '6px' }}>
              {Object.entries(classifyResult.type_counts).map(([type, count]) => (
                <div key={type} style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 0' }}>
                  <span style={{ color: '#999' }}>{type}</span>
                  <span style={{ color: '#f0f0f0', fontFamily: 'monospace' }}>{count as number}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </StepCard>

      {/* Step E1: Snap to Surfaces */}
      <StepCard title="E1. Snap to Surfaces" requiredStage="preprocessed">
        <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px', lineHeight: '1.4' }}>
          Snap HIGH-region vertices to analytic surfaces. Produces a geometrically clean mesh.
        </div>
        <button
          style={{ ...actionBtn('#a855f7'), opacity: intentSummary && !loading ? 1 : 0.4 }}
          onClick={() => runSnapToSurfaces()}
          disabled={!intentSummary || loading}
        >
          {loading ? 'Snapping...' : 'Snap to Surfaces'}
        </button>
        {snapResult && (
          <div style={{
            fontSize: '11px',
            color: '#888',
            marginTop: '10px',
            lineHeight: '1.6',
            background: '#141414',
            borderRadius: '6px',
            padding: '10px',
          }}>
            <div>
              <span style={{ color: '#a855f7', fontWeight: 600 }}>
                {snapResult.pct_snapped.toFixed(1)}%
              </span> vertices snapped
            </div>
            <div style={{ marginTop: '4px' }}>
              <span style={{ color: '#f0f0f0' }}>{snapResult.n_snapped}</span> surface
              {' / '}
              <span style={{ color: '#f0f0f0' }}>{snapResult.n_edge}</span> edge
              {' / '}
              <span style={{ color: '#f0f0f0' }}>{snapResult.n_corner}</span> corner
            </div>
            <div style={{ marginTop: '4px', color: '#555', fontSize: '10px' }}>
              mean displacement: {snapResult.mean_displacement.toFixed(4)}
            </div>
          </div>
        )}
        {snapResult && (
          <label style={{
            fontSize: '11px',
            color: '#999',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            cursor: 'pointer',
            marginTop: '8px',
          }}>
            <input
              type="checkbox"
              checked={showSnappedMesh}
              onChange={toggleSnappedMesh}
              style={{ accentColor: '#a855f7' }}
            />
            Show snapped mesh
          </label>
        )}
      </StepCard>

      {/* Quality Check */}
      <StepCard title="Quality Check" requiredStage="preprocessed">
        <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px', lineHeight: '1.4' }}>
          Compare original scan against fitted surfaces. Shows deviation heatmap (green = close, red = far).
        </div>
        <button
          style={{ ...actionBtn('#f43f5e'), opacity: snapResult && !loading ? 1 : 0.4 }}
          onClick={() => runDeviationAnalysis()}
          disabled={!snapResult || loading}
        >
          {loading ? 'Analyzing...' : 'Run Deviation Analysis'}
        </button>
        {deviationResult && (
          <div style={{
            fontSize: '11px', color: '#888', marginTop: '10px', lineHeight: '1.6',
            background: '#141414', borderRadius: '6px', padding: '10px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Mean deviation</span>
              <span style={{ color: '#f0f0f0', fontFamily: 'monospace' }}>{deviationResult.mean_deviation.toFixed(4)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '2px' }}>
              <span>Max deviation</span>
              <span style={{ color: '#f0f0f0', fontFamily: 'monospace' }}>{deviationResult.max_deviation.toFixed(4)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '2px' }}>
              <span>95th percentile</span>
              <span style={{ color: '#f0f0f0', fontFamily: 'monospace' }}>{deviationResult.p95_deviation.toFixed(4)}</span>
            </div>
            {/* Color legend */}
            <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <div style={{ width: '60px', height: '8px', borderRadius: '4px', background: 'linear-gradient(to right, #22c55e, #eab308, #ef4444)' }} />
              <span style={{ fontSize: '9px', color: '#666' }}>0 → {deviationResult.color_scale_max.toFixed(2)}</span>
            </div>
          </div>
        )}
        {deviationResult && (
          <label style={{
            fontSize: '11px', color: '#999', display: 'flex', alignItems: 'center',
            gap: '6px', cursor: 'pointer', marginTop: '8px',
          }}>
            <input type="checkbox" checked={showDeviationHeatmap} onChange={toggleDeviationHeatmap}
              style={{ accentColor: '#f43f5e' }} />
            Show deviation heatmap
          </label>
        )}
      </StepCard>

      {/* Step E2: Trimmed Faces */}
      <StepCard title="E2. Trimmed Faces" requiredStage="preprocessed">
        <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px', lineHeight: '1.4' }}>
          Extract boundary loops and UV parameterization for each face on its analytic surface.
        </div>
        <button
          style={{ ...actionBtn('#eab308'), opacity: snapResult && !loading ? 1 : 0.4 }}
          onClick={() => runConstructFaces()}
          disabled={!snapResult || loading}
        >
          {loading ? 'Constructing...' : 'Construct Faces'}
        </button>
        {trimResult && (
          <div style={{
            fontSize: '11px',
            color: '#888',
            marginTop: '10px',
            lineHeight: '1.6',
            background: '#141414',
            borderRadius: '6px',
            padding: '10px',
          }}>
            <div>
              <span style={{ color: '#eab308', fontWeight: 600 }}>
                {trimResult.n_trimmed_faces}
              </span> trimmed faces
              {trimResult.n_with_holes > 0 && (
                <span style={{ color: '#666' }}> ({trimResult.n_with_holes} with holes)</span>
              )}
            </div>
            <div style={{ marginTop: '4px' }}>
              <span style={{ color: '#22c55e' }}>{trimResult.n_plane_faces}</span> plane
              {' / '}
              <span style={{ color: '#06b6d4' }}>{trimResult.n_cylinder_faces}</span> cyl
              {' / '}
              <span style={{ color: '#f59e0b' }}>{trimResult.n_cone_faces}</span> cone
            </div>
            <div style={{ marginTop: '4px', color: '#555', fontSize: '10px' }}>
              {trimResult.total_boundary_vertices.toLocaleString()} boundary vertices
            </div>
          </div>
        )}
      </StepCard>

      {/* Step E3: Export STEP */}
      <StepCard title="E3. Export STEP" requiredStage="preprocessed" last>
        <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px', lineHeight: '1.4' }}>
          Build polyhedral B-Rep from snapped mesh and export as STEP for SolidWorks / Fusion 360.
        </div>
        <button
          style={{ ...actionBtn('#10b981'), opacity: snapResult && !loading ? 1 : 0.4 }}
          onClick={() => runExportStep()}
          disabled={!snapResult || loading}
        >
          {loading ? 'Exporting...' : 'Export STEP'}
        </button>
        {stepResult && (
          <div style={{
            fontSize: '11px',
            color: '#888',
            marginTop: '10px',
            lineHeight: '1.6',
            background: '#141414',
            borderRadius: '6px',
            padding: '10px',
          }}>
            <div>
              <span style={{ color: '#10b981', fontWeight: 600 }}>
                {stepResult.n_faces_built}
              </span> B-Rep faces built
              {stepResult.n_faces_failed > 0 && (
                <span style={{ color: '#ef4444' }}> ({stepResult.n_faces_failed} failed)</span>
              )}
            </div>
            <div style={{ marginTop: '4px', color: '#555', fontSize: '10px' }}>
              STEP file: {(stepResult.step_size / 1024).toFixed(0)} KB
            </div>
            <button
              style={{
                ...actionBtn('#10b981'),
                marginTop: '8px',
                background: 'transparent',
                border: '1px solid #10b981',
                color: '#10b981',
              }}
              onClick={() => downloadStep()}
            >
              Download STEP File
            </button>
          </div>
        )}
      </StepCard>
    </div>
  )
}
