import React, { useState } from 'react'
import { usePipelineStore } from '../store/pipelineStore'

const styles = {
  panel: {
    width: '260px',
    minWidth: '260px',
    background: '#16213e',
    borderRight: '1px solid #0f3460',
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
  card: {
    background: '#1a1a2e',
    border: '1px solid #0f3460',
    borderRadius: '6px',
    padding: '10px 12px',
    marginBottom: '8px',
  },
  cardTitle: {
    fontSize: '12px',
    fontWeight: 600,
    marginBottom: '6px',
  },
  cardDone: {
    borderColor: '#4ecca3',
  },
  cardActive: {
    borderColor: '#e94560',
  },
  btn: {
    padding: '5px 12px',
    border: 'none',
    borderRadius: '3px',
    background: '#0f3460',
    color: '#e0e0e0',
    fontSize: '11px',
    cursor: 'pointer',
    width: '100%',
    marginTop: '6px',
  },
  paramRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between' as const,
    fontSize: '11px',
    marginTop: '4px',
  },
  input: {
    width: '60px',
    padding: '2px 4px',
    background: '#0f3460',
    border: '1px solid #233',
    borderRadius: '3px',
    color: '#e0e0e0',
    fontSize: '11px',
  },
}

const stageOrder = ['idle', 'loaded', 'preprocessed', 'segmented', 'fitted', 'features_detected', 'exported']

function StepCard({
  title,
  requiredStage,
  children,
}: {
  title: string
  requiredStage: string
  children: React.ReactNode
}) {
  const stage = usePipelineStore((s) => s.stage)
  const currentIdx = stageOrder.indexOf(stage)
  const requiredIdx = stageOrder.indexOf(requiredStage)
  const isDone = currentIdx > requiredIdx + 1 || (currentIdx > requiredIdx)
  const isActive = currentIdx === requiredIdx

  return (
    <div
      style={{
        ...styles.card,
        ...(isDone ? styles.cardDone : {}),
        ...(isActive ? styles.cardActive : {}),
      }}
    >
      <div style={styles.cardTitle}>
        {isDone ? '\u2713 ' : ''}{title}
      </div>
      {children}
    </div>
  )
}

export function PipelinePanel() {
  const {
    stage, loading,
    runCleanup, runPreprocess, runSegmentation, runFitting, runExtractCage, runFeatureDetection,
    meshInfo, patches, primitives, features, infiniteSurfaces, edges,
    showInfiniteSurfaces, toggleInfiniteSurfaces, runIntersection,
    cadPreview, showCadPreview, toggleCadPreview, runCadPreview,
    polyhedralCad, showPolyhedralCad, togglePolyhedralCad, runPolyhedralCad,
    selectedPatchIds, mergeMap, mergeSelectedPatches, clearMergeGroups, clearPatchSelection,
    createInfinitePlaneFromSelection, intersectUserPlanes, clearUserPlanes,
    userPlanes, selectedPlaneIds, togglePlaneSelection, clearPlaneSelection,
    history, future, undo, redo,
    runPoint2Cyl, point2cylResult, showPoint2Cyl, togglePoint2Cyl,
    buildPolyhedralBrep, filletSharpEdges, chamferSharpEdges, exportBrep,
  } = usePipelineStore()

  const [brepTarget, setBrepTarget] = useState(2000)
  const [filletRadius, setFilletRadius] = useState(0.3)
  const [filletMinAngle, setFilletMinAngle] = useState(100)
  const [chamferDist, setChamferDist] = useState(0.3)

  const groupCount = (() => {
    const roots = new Set<number>()
    for (const k of Object.keys(mergeMap)) roots.add(mergeMap[Number(k)])
    return roots.size
  })()

  const [smoothIter, setSmoothIter] = useState(10)
  const [angleThresh, setAngleThresh] = useState(60)
  const [poissonDepth, setPoissonDepth] = useState(10)
  const [sampleCount, setSampleCount] = useState(400000)
  const [showCleanupAdv, setShowCleanupAdv] = useState(false)
  const [cageMinInlier, setCageMinInlier] = useState(0.85)
  const [cagePlaneAngle, setCagePlaneAngle] = useState(5)
  const [showCageAdv, setShowCageAdv] = useState(false)

  const canPreprocess = stage === 'loaded' || stage === 'preprocessed'
  const canSegment = stage === 'preprocessed' || stage === 'loaded'
  const canFit = stage === 'segmented'
  const canDetect = stage === 'fitted'

  return (
    <div style={styles.panel}>
      <div style={styles.heading}>Pipeline</div>

      <StepCard title="1. Load Mesh" requiredStage="idle">
        {meshInfo && (
          <div style={{ fontSize: '11px', color: '#aaa' }}>
            {meshInfo.vertices.toLocaleString()} verts, {meshInfo.faces.toLocaleString()} faces
            {meshInfo.is_watertight && ' (watertight)'}
          </div>
        )}
      </StepCard>

      <StepCard title="1b. Cleanup (Poisson)" requiredStage="loaded">
        <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '6px' }}>
          Watertight remesh of the raw scan via screened Poisson reconstruction.
        </div>
        <button
          style={{ ...styles.btn, opacity: meshInfo && !loading ? 1 : 0.5, background: '#4ecca3', color: '#0a0a1a', fontWeight: 600 }}
          onClick={() => runCleanup({ poisson_depth: poissonDepth, sample_count: sampleCount })}
          disabled={!meshInfo || loading}
        >
          {loading ? 'Cleaning…' : 'Run Cleanup'}
        </button>
        <div
          style={{ fontSize: '10px', color: '#888', marginTop: '6px', cursor: 'pointer', userSelect: 'none' }}
          onClick={() => setShowCleanupAdv((v) => !v)}
        >
          {showCleanupAdv ? '▾ Advanced' : '▸ Advanced'}
        </div>
        {showCleanupAdv && (
          <div style={{ marginTop: '4px' }}>
            <div style={styles.paramRow}>
              <span>Poisson depth</span>
              <input
                type="number"
                style={styles.input}
                min={7}
                max={12}
                value={poissonDepth}
                onChange={(e) => setPoissonDepth(Number(e.target.value))}
              />
            </div>
            <div style={styles.paramRow}>
              <span>Sample count</span>
              <input
                type="number"
                style={styles.input}
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

      <StepCard title="2. Preprocess" requiredStage="loaded">
        <div style={styles.paramRow}>
          <span>Smoothing iterations</span>
          <input
            type="number"
            style={styles.input}
            value={smoothIter}
            onChange={(e) => setSmoothIter(Number(e.target.value))}
            min={0}
            max={100}
          />
        </div>
        <button
          style={{ ...styles.btn, opacity: canPreprocess && !loading ? 1 : 0.5 }}
          onClick={() => runPreprocess({ iterations: smoothIter })}
          disabled={!canPreprocess || loading}
        >
          {loading ? 'Processing...' : 'Run Preprocessing'}
        </button>
      </StepCard>

      <StepCard title="3. Segment" requiredStage="preprocessed">
        <div style={styles.paramRow}>
          <span>Angle threshold</span>
          <input
            type="number"
            style={styles.input}
            value={angleThresh}
            onChange={(e) => setAngleThresh(Number(e.target.value))}
            min={1}
            max={90}
          />
        </div>
        <button
          style={{ ...styles.btn, opacity: canSegment && !loading ? 1 : 0.5 }}
          onClick={() => runSegmentation({ angle_threshold: angleThresh })}
          disabled={!canSegment || loading}
        >
          {loading ? 'Segmenting...' : 'Run Segmentation'}
        </button>
        {patches.length > 0 && (
          <div style={{ fontSize: '11px', color: '#aaa', marginTop: '4px', lineHeight: '1.5' }}>
            {patches.length} patches:{' '}
            <span style={{ color: '#4ecca3' }}>
              {patches.filter((p) => p.classification === 'planar').length} planar
            </span>
            ,{' '}
            <span style={{ color: '#e94560' }}>
              {patches.filter((p) => p.classification === 'cylindrical').length} cyl
            </span>
            ,{' '}
            <span style={{ color: '#f5a623' }}>
              {patches.filter((p) => p.classification === 'spherical').length} sph
            </span>
            <br />
            <span style={{ color: '#f368e0' }}>
              {patches.filter((p) => p.classification === 'fillet').length} fillets
            </span>
            ,{' '}
            <span style={{ color: '#888' }}>
              {patches.filter((p) => ['freeform', 'curved'].includes(p.classification)).length} freeform
            </span>
          </div>
        )}
        {patches.length > 0 && (
          <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #0f3460' }}>
            <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '4px' }}>
              Click faces to select. Shift/Ctrl-click to add.
            </div>
            <div style={{ fontSize: '11px', color: '#e0e0e0', marginBottom: '4px' }}>
              Selected: <span style={{ color: '#feca57' }}>{selectedPatchIds.length}</span>
              {groupCount > 0 && <> · Groups: <span style={{ color: '#4ecca3' }}>{groupCount}</span></>}
            </div>
            <button
              style={{ ...styles.btn, opacity: selectedPatchIds.length >= 2 ? 1 : 0.5 }}
              onClick={mergeSelectedPatches}
              disabled={selectedPatchIds.length < 2}
            >
              Merge Selected ({selectedPatchIds.length})
            </button>
            <button
              style={{ ...styles.btn, opacity: selectedPatchIds.length >= 1 ? 1 : 0.5 }}
              onClick={createInfinitePlaneFromSelection}
              disabled={selectedPatchIds.length === 0}
            >
              Create ∞ Plane from Selection
            </button>

            {userPlanes.length > 0 && (
              <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #0f3460' }}>
                <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '4px' }}>
                  Planes ({userPlanes.length}) — click to select pair to intersect
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '6px' }}>
                  {userPlanes.map((p) => {
                    const sel = selectedPlaneIds.includes(p.id)
                    return (
                      <button
                        key={p.id}
                        onClick={(e) => togglePlaneSelection(p.id, e.shiftKey || e.ctrlKey || e.metaKey)}
                        style={{
                          fontSize: '10px',
                          padding: '3px 7px',
                          border: '1px solid ' + (sel ? '#feca57' : '#0f3460'),
                          background: sel ? '#3a3520' : '#1a1a2e',
                          color: sel ? '#feca57' : '#aaa',
                          borderRadius: '3px',
                          cursor: 'pointer',
                        }}
                      >
                        #{p.id}
                      </button>
                    )
                  })}
                </div>
                <button
                  style={{ ...styles.btn, opacity: selectedPlaneIds.length >= 2 ? 1 : 0.5 }}
                  onClick={intersectUserPlanes}
                  disabled={selectedPlaneIds.length < 2}
                >
                  Intersect Selected ({selectedPlaneIds.length})
                </button>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <button
                    style={{ ...styles.btn, flex: 1, opacity: selectedPlaneIds.length > 0 ? 1 : 0.5 }}
                    onClick={clearPlaneSelection}
                    disabled={selectedPlaneIds.length === 0}
                  >
                    Deselect
                  </button>
                  <button style={{ ...styles.btn, flex: 1 }} onClick={clearUserPlanes}>
                    Clear All Planes
                  </button>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', gap: '4px', marginTop: '6px' }}>
              <button
                style={{ ...styles.btn, flex: 1, opacity: history.length > 0 ? 1 : 0.5 }}
                onClick={undo}
                disabled={history.length === 0}
                title="Ctrl+Z"
              >
                ↶ Undo
              </button>
              <button
                style={{ ...styles.btn, flex: 1, opacity: future.length > 0 ? 1 : 0.5 }}
                onClick={redo}
                disabled={future.length === 0}
                title="Ctrl+Y"
              >
                ↷ Redo
              </button>
            </div>
            <div style={{ display: 'flex', gap: '4px' }}>
              <button
                style={{ ...styles.btn, flex: 1, opacity: selectedPatchIds.length > 0 ? 1 : 0.5 }}
                onClick={clearPatchSelection}
                disabled={selectedPatchIds.length === 0}
              >
                Clear Selection
              </button>
              <button
                style={{ ...styles.btn, flex: 1, opacity: groupCount > 0 ? 1 : 0.5 }}
                onClick={clearMergeGroups}
                disabled={groupCount === 0}
              >
                Reset Merges
              </button>
            </div>
          </div>
        )}
      </StepCard>

      <StepCard title="4. Fit Primitives" requiredStage="segmented">
        <button
          style={{ ...styles.btn, opacity: canFit && !loading ? 1 : 0.5 }}
          onClick={runFitting}
          disabled={!canFit || loading}
        >
          {loading ? 'Fitting...' : 'Fit Primitives'}
        </button>
        {primitives.length > 0 && (
          <div style={{ fontSize: '11px', color: '#aaa', marginTop: '4px' }}>
            {primitives.filter((p) => p.type === 'plane').length} planes,{' '}
            {primitives.filter((p) => p.type === 'cylinder').length} cylinders,{' '}
            {primitives.filter((p) => p.type === 'sphere').length} spheres,{' '}
            {primitives.filter((p) => p.type === 'bspline').length} B-splines
          </div>
        )}
      </StepCard>

      <StepCard title="4b. Extract Cage" requiredStage="fitted">
        <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '6px' }}>
          Build a low-poly editable cage by keeping high-confidence primitive
          patches and merging compatible neighbors.
        </div>
        <button
          style={{
            ...styles.btn,
            opacity: primitives.length > 0 && !loading ? 1 : 0.5,
            background: '#feca57',
            color: '#0a0a1a',
            fontWeight: 600,
          }}
          onClick={() =>
            runExtractCage({
              min_inlier_ratio: cageMinInlier,
              plane_normal_deg: cagePlaneAngle,
            })
          }
          disabled={primitives.length === 0 || loading}
        >
          {loading ? 'Extracting…' : 'Extract Cage'}
        </button>
        <div
          style={{ fontSize: '10px', color: '#888', marginTop: '6px', cursor: 'pointer', userSelect: 'none' }}
          onClick={() => setShowCageAdv((v) => !v)}
        >
          {showCageAdv ? '▾ Advanced' : '▸ Advanced'}
        </div>
        {showCageAdv && (
          <div style={{ marginTop: '4px' }}>
            <div style={styles.paramRow}>
              <span>Min inlier ratio</span>
              <input
                type="number"
                style={styles.input}
                step={0.05}
                min={0}
                max={1}
                value={cageMinInlier}
                onChange={(e) => setCageMinInlier(Number(e.target.value))}
              />
            </div>
            <div style={styles.paramRow}>
              <span>Plane angle (°)</span>
              <input
                type="number"
                style={styles.input}
                min={1}
                max={30}
                value={cagePlaneAngle}
                onChange={(e) => setCagePlaneAngle(Number(e.target.value))}
              />
            </div>
          </div>
        )}
      </StepCard>

      <StepCard title="5. Detect Features" requiredStage="fitted">
        <button
          style={{ ...styles.btn, opacity: canDetect && !loading ? 1 : 0.5 }}
          onClick={runFeatureDetection}
          disabled={!canDetect || loading}
        >
          {loading ? 'Detecting...' : 'Detect Features'}
        </button>
        {(infiniteSurfaces?.length || 0) > 0 && (
          <>
            <div style={{ fontSize: '11px', color: '#aaa', marginTop: '4px', lineHeight: '1.5' }}>
              <span style={{ color: '#4ecca3' }}>
                {infiniteSurfaces.filter((s: any) => s.type === 'infinite_plane').length} ∞ planes
              </span>
              ,{' '}
              <span style={{ color: '#e94560' }}>
                {infiniteSurfaces.filter((s: any) => s.type === 'infinite_cylinder').length} ∞ cyls
              </span>
              ,{' '}
              <span style={{ color: '#f5a623' }}>
                {infiniteSurfaces.filter((s: any) => s.type === 'infinite_sphere').length} ∞ sph
              </span>
            </div>
            <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px', marginTop: '4px' }}>
              <input type="checkbox" checked={showInfiniteSurfaces} onChange={toggleInfiniteSurfaces} />
              Show ∞ surfaces
            </label>
          </>
        )}
      </StepCard>

      <StepCard title="6. Intersect Surfaces" requiredStage="features_detected">
        <button
          style={{ ...styles.btn, opacity: (features.length > 0) && !loading ? 1 : 0.5 }}
          onClick={runIntersection}
          disabled={features.length === 0 || loading}
        >
          {loading ? 'Intersecting...' : 'Intersect → Edge Curves'}
        </button>
        {(edges?.length || 0) > 0 && (
          <div style={{ fontSize: '11px', color: '#ffff00', marginTop: '4px' }}>
            {edges.length} edge curves (sharp edges)
          </div>
        )}
      </StepCard>

      <StepCard title="7. CAD Preview" requiredStage="fitted">
        <button
          style={{ ...styles.btn, opacity: (primitives.length > 0) && !loading ? 1 : 0.5 }}
          onClick={runCadPreview}
          disabled={primitives.length === 0 || loading}
        >
          {loading ? 'Building...' : 'Build CAD Preview'}
        </button>
        {(cadPreview?.length || 0) > 0 && (
          <>
            <div style={{ fontSize: '11px', color: '#4ecca3', marginTop: '4px' }}>
              {cadPreview.length} CAD surfaces
            </div>
            <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px', marginTop: '4px' }}>
              <input type="checkbox" checked={showCadPreview} onChange={toggleCadPreview} />
              Show CAD preview (hide raw mesh)
            </label>
          </>
        )}
      </StepCard>

      <StepCard title="9. Phase C — Features (chamfer / fillet)" requiredStage="loaded">
        <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '6px' }}>
          Lift the mesh to a polyhedral B-Rep and apply chamfer/fillet on
          its sharpest edges via OCC.
        </div>
        <div style={styles.paramRow}>
          <span>Target faces</span>
          <input
            type="number"
            style={styles.input}
            min={50}
            max={5000}
            step={50}
            value={brepTarget}
            onChange={(e) => setBrepTarget(Number(e.target.value))}
          />
        </div>
        <button
          style={{ ...styles.btn, opacity: meshInfo && !loading ? 1 : 0.5, background: '#7b68ee', color: '#fff', fontWeight: 600 }}
          onClick={() => buildPolyhedralBrep({ target_faces: brepTarget })}
          disabled={!meshInfo || loading}
        >
          {loading ? 'Building…' : 'Build Polyhedral B-Rep'}
        </button>
        <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #0f3460' }}>
          <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '4px' }}>Sharp edge threshold (°)</div>
          <input
            type="number"
            style={{ ...styles.input, width: '100%' }}
            min={30}
            max={170}
            value={filletMinAngle}
            onChange={(e) => setFilletMinAngle(Number(e.target.value))}
          />
          <div style={{ display: 'flex', gap: '4px', marginTop: '6px' }}>
            <input
              type="number"
              style={{ ...styles.input, width: '40%' }}
              min={0.01}
              step={0.05}
              value={filletRadius}
              onChange={(e) => setFilletRadius(Number(e.target.value))}
            />
            <button
              style={{ ...styles.btn, flex: 1, marginTop: 0 }}
              onClick={() => filletSharpEdges({ radius: filletRadius, min_dihedral_deg: filletMinAngle })}
              disabled={loading}
            >
              Fillet
            </button>
          </div>
          <div style={{ display: 'flex', gap: '4px', marginTop: '4px' }}>
            <input
              type="number"
              style={{ ...styles.input, width: '40%' }}
              min={0.01}
              step={0.05}
              value={chamferDist}
              onChange={(e) => setChamferDist(Number(e.target.value))}
            />
            <button
              style={{ ...styles.btn, flex: 1, marginTop: 0 }}
              onClick={() => chamferSharpEdges({ distance: chamferDist, min_dihedral_deg: filletMinAngle })}
              disabled={loading}
            >
              Chamfer
            </button>
          </div>
        </div>
        <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #0f3460' }}>
          <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '4px' }}>Export current B-Rep</div>
          <div style={{ display: 'flex', gap: '4px' }}>
            <button style={{ ...styles.btn, flex: 1, marginTop: 0 }} onClick={() => exportBrep('stl')} disabled={loading}>STL</button>
            <button style={{ ...styles.btn, flex: 1, marginTop: 0 }} onClick={() => exportBrep('step')} disabled={loading}>STEP</button>
            <button style={{ ...styles.btn, flex: 1, marginTop: 0 }} onClick={() => exportBrep('brep')} disabled={loading}>BREP</button>
          </div>
        </div>
      </StepCard>

      <StepCard title="🤖 Point2Cyl (AI)" requiredStage="loaded">
        <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '6px' }}>
          Decompose the mesh into extrusion cylinders using a pretrained
          PointNet++ model (Fusion360 dataset).
        </div>
        <button
          style={{ ...styles.btn, opacity: !loading ? 1 : 0.5, background: '#5f27cd' }}
          onClick={runPoint2Cyl}
          disabled={loading || !meshInfo}
        >
          {loading ? 'Running…' : 'Run Point2Cyl'}
        </button>
        {point2cylResult && (
          <>
            <div style={{ fontSize: '11px', color: '#feca57', marginTop: '4px' }}>
              {point2cylResult.n_segments} extrusions detected
            </div>
            <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px', marginTop: '4px' }}>
              <input type="checkbox" checked={showPoint2Cyl} onChange={togglePoint2Cyl} />
              Show extrusions
            </label>
          </>
        )}
      </StepCard>

      <StepCard title="8. Final Polyhedral CAD" requiredStage="fitted">
        <button
          style={{ ...styles.btn, opacity: (primitives.length > 0) && !loading ? 1 : 0.5 }}
          onClick={runPolyhedralCad}
          disabled={primitives.length === 0 || loading}
        >
          {loading ? 'Building...' : 'Build Polyhedral CAD'}
        </button>
        {polyhedralCad && (
          <>
            <div style={{ fontSize: '11px', color: '#4ecca3', marginTop: '4px' }}>
              {polyhedralCad.faces.length} CAD faces
            </div>
            <label style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px', marginTop: '4px' }}>
              <input type="checkbox" checked={showPolyhedralCad} onChange={togglePolyhedralCad} />
              Show Polyhedral CAD
            </label>
          </>
        )}
      </StepCard>
    </div>
  )
}
