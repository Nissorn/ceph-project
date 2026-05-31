import React, { useState, useEffect, useRef, useMemo, useCallback, forwardRef, useImperativeHandle } from 'react';
import { Stage, Layer, Image as KonvaImage, Circle, Line, Text, Group } from 'react-konva';
import type { KonvaEventObject } from 'konva/lib/Node';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Keypoint {
  id: string;
  name: string;
  x: number; // image-space pixels
  y: number;
  confidence?: number; // AI prediction confidence 0-1; absent = manually placed
}

export interface PolygonShape {
  id: string;
  name: string;
  points: number[]; // flat [x1,y1,...] image-space pixels
  fill: string;
  stroke: string;
}

// ── Plan B 3-level bone thickness lines from backend ─────────────────────────
// New 6-segment schema: at each of 3 levels, two independent tooth→bone gaps
// are measured — one on the PALATAL side, one on the LABIAL side.
// Each segment has its own start/end pixel coordinates so the frontend draws
// the exact gap, NOT a line through the tooth.
export interface Segment6 {
  // Palatal side (tooth surface → palatal bone surface)
  palatal_distance_mm: number;
  palatal_tooth_x: number;  palatal_tooth_y: number;
  palatal_bone_x:  number;  palatal_bone_y:  number;
  // Labial side (labial bone surface → tooth surface)
  labial_distance_mm: number;
  labial_tooth_x:  number;  labial_tooth_y:  number;
  labial_bone_x:   number;  labial_bone_y:   number;
}
export type Lines3Level = {
  cervical: Segment6;
  middle:   Segment6;
  apical:   Segment6;
};

/** Two global-minimum bone gap lines for "Min Distance" mode.
 *  Origin (x1, y1) is the tooth SURFACE; endpoint (x2, y2) is the bone surface.
 *  labial_mm / palatal_mm are pre-computed by the backend for accurate labels. */
export interface GlobalMinLines {
  labial_line:  number[][];   // [[x_tooth, y_tooth], [x_bone, y_bone]] image-px
  palatal_line: number[][];
  labial_mm:    number;
  palatal_mm:   number;
}

interface Props {
  imageFile: File;
  initialKeypoints?: Keypoint[];
  initialPolygons?: PolygonShape[];
  boneThickness?: Lines3Level;    // Standard mode — 3-level measurement lines
  globalMinLines?: GlobalMinLines; // Min Distance mode — 2 bold bottleneck lines
  originalAnalysis?: any;          // Full payload for AI Reset
  onKeypointsChange?: (kps: Keypoint[]) => void;
  onPolygonsChange?: (polys: PolygonShape[]) => void;
  onRecalculate?: (results: any) => void;
}

// ── Geometry Math Utility ──────────────────────────────────────────────────
// Removed auto-snap math utility as per UX simplification.

// ── Medical imaging colour palette ───────────────────────────────────────────

const POLY_PALETTE = [
  { fill: 'rgba(6, 182, 212, 0.6)',  stroke: 'rgba(6, 182, 212, 0.9)'  },  // Cyan Accent    — Upper_incisor (Translucent for Z-Index manual tucking)
  { fill: 'rgba(236, 72, 153, 0.15)', stroke: 'rgba(236, 72, 153, 0.9)' }, // Pink/Magenta — Labial_bone
  { fill: 'rgba(16, 185, 129, 0.15)', stroke: 'rgba(16, 185, 129, 0.9)' }, // Emerald Green  — Palatal_bone
];

// ── Confidence colour tiers ────────────────────────────────────────────────────
//  conf >= 0.85  → normal  (amber ring)
//  0.70 <= conf < 0.85 → warning  (yellow ring)
//  conf <  0.70  → critical (red ring + label flag)

const KP_RING_CRITICAL = '#ef4444';  // Red    — conf < 0.70
const KP_RING_WARNING  = '#eab308';  // Yellow — 0.70 ≤ conf < 0.85
const KP_RING_NORMAL   = '#fbbf24';  // Amber  — conf ≥ 0.85
const KP_DOT_NORMAL    = '#ffffff';
const KP_RING_SELECTED = '#ffffff';
const KP_DOT_SELECTED  = '#f59e0b';

// ── Correct clinical landmark defaults (normalised 0–1 for lateral ceph) ─────

const KP_DEFS: { name: string; fx: number; fy: number }[] = [
  { name: 'Upper_tip',       fx: 0.62, fy: 0.56 },
  { name: 'Upper_apex',      fx: 0.56, fy: 0.38 },
  { name: 'Labial_midroot',  fx: 0.62, fy: 0.47 },
  { name: 'Labial_crest',    fx: 0.62, fy: 0.54 },
  { name: 'Palatal_midroot', fx: 0.51, fy: 0.47 },
  { name: 'Palatal_crest',   fx: 0.51, fy: 0.54 },
  { name: 'ANS',             fx: 0.57, fy: 0.52 },
  { name: 'PNS',             fx: 0.35, fy: 0.52 },
  { name: 'LB',              fx: 0.64, fy: 0.50 },
  { name: 'PB',              fx: 0.49, fy: 0.50 },
];

const POLY_DEFS: { name: string; fracs: number[] }[] = [
  { name: 'Upper_incisor', fracs: [0.52,0.42, 0.64,0.42, 0.66,0.58, 0.50,0.58] },
  { name: 'Labial_bone',   fracs: [0.58,0.40, 0.68,0.40, 0.70,0.52, 0.56,0.52] },
  { name: 'Palatal_bone',  fracs: [0.44,0.44, 0.56,0.44, 0.56,0.58, 0.42,0.58] },
];

function fracsToImg(fracs: number[], w: number, h: number): number[] {
  return fracs.map((v, i) => (i % 2 === 0 ? v * w : v * h));
}

// ── Geometry: nearest-edge insertion ─────────────────────────────────────────

function nearestEdgeInsert(
  pts: number[], cx: number, cy: number
): { insertAt: number; px: number; py: number; dist: number } {
  const n = pts.length / 2;
  let best = { insertAt: 2, px: 0, py: 0, dist: Infinity };
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ax = pts[i*2], ay = pts[i*2+1], bx = pts[j*2], by = pts[j*2+1];
    const dx = bx - ax, dy = by - ay;
    const lenSq = dx*dx + dy*dy;
    const t = lenSq === 0 ? 0 : Math.max(0, Math.min(1, ((cx-ax)*dx + (cy-ay)*dy) / lenSq));
    const projX = ax + t*dx, projY = ay + t*dy;
    const dist = Math.hypot(cx - projX, cy - projY);
    if (dist < best.dist) best = { insertAt: j*2, px: projX, py: projY, dist };
  }
  return best;
}

// ── Component ─────────────────────────────────────────────────────────────────

const CephCanvasEditor = forwardRef(function CephCanvasEditor({
  imageFile,
  initialKeypoints,
  initialPolygons,
  boneThickness,
  globalMinLines,
  originalAnalysis,
  onKeypointsChange,
  onPolygonsChange,
  onRecalculate,
}: Props, ref) {
  const containerRef  = useRef<HTMLDivElement>(null);
  const stageRef      = useRef<any>(null);
  const scaleRef      = useRef(1);
  const importFileRef = useRef<HTMLInputElement>(null);

  useImperativeHandle(ref, () => ({
    getCanvasImage: () => {
      if (!stageRef.current) return null;
      return stageRef.current.toDataURL({ pixelRatio: 2, mimeType: 'image/jpeg', quality: 0.85 });
    }
  }));

  const [stageW, setStageW]               = useState(0);
  const [stageH, setStageH]               = useState(0);
  const [img, setImg]                     = useState<HTMLImageElement | null>(null);
  const [keypoints, setKeypoints]         = useState<Keypoint[]>([]);
  const [polygons, setPolygons]           = useState<PolygonShape[]>([]);
  const [selectedId, setSelectedId]       = useState<string | null>(null);
  const [stageScale, setStageScale]       = useState(1);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showPolygons, setShowPolygons]   = useState(true);
  const [showMeasurementLines, setShowMeasurementLines] = useState(true);
  const [pointSize, setPointSize]         = useState(4);
  const [debugInfo, setDebugInfo]         = useState({ x: 0, y: 0, imageX: 0, imageY: 0 });
  const [isDebugMode, setIsDebugMode]     = useState(false);
  const [isToolbarOpen, setIsToolbarOpen]     = useState(true);

  // ── TARGET 1: Undo/Redo state history ─────────────────────────────────────────
  const [historyStack, setHistoryStack]       = useState<{ kps: Keypoint[]; polys: PolygonShape[] }[]>([]);
  const [isFrozen, setIsFrozen]               = useState(false);

  // Push current keypoint+polygon snapshot onto history
  const pushHistory = useCallback(() => {
    setHistoryStack(prev => {
      const next = [...prev, { 
        kps: JSON.parse(JSON.stringify(keypoints)), 
        polys: JSON.parse(JSON.stringify(polygons)) 
      }];
      // cap at 20 entries to avoid unbounded memory growth
      return next.length > 20 ? next.slice(next.length - 20) : next;
    });
  }, [keypoints, polygons]);

  // Pop the last snapshot and restore
  const undo = useCallback(() => {
    if (historyStack.length === 0) return;
    const prev = historyStack[historyStack.length - 1];
    setHistoryStack(s => s.slice(0, -1));
    setKeypoints(JSON.parse(JSON.stringify(prev.kps)));
    setPolygons(JSON.parse(JSON.stringify(prev.polys)));
    console.log('[CephEditor] Undo — restoring previous state.');
  }, [historyStack]);

  // ── TARGET 2: Reset to AI Default ──────────────────────────────────────────────
  const handleResetToAI = useCallback(() => {
    if (!originalAnalysis || isFrozen) return;
    console.log('[CephEditor] Reset to AI Default triggered.');
    
    // Double deep-clone the original analysis to prevent mutation by reference
    const cleanReset = JSON.parse(JSON.stringify(originalAnalysis));
    
    // Restore state
    setHistoryStack([]);
    
    if (onRecalculate) {
      onRecalculate(cleanReset);
    }
    
    // Locally reset keypoints and polygons from the mapped annotations
    if (cleanReset.annotations?.keypoints) setKeypoints(cleanReset.annotations.keypoints);
    if (cleanReset.annotations?.polygons) setPolygons(cleanReset.annotations.polygons);
    
  }, [originalAnalysis, isFrozen, onRecalculate]);

  // ── TARGET 1: Confirm & Save — freeze + POST to API ──────────────────────────
  const handleConfirmAndSave = useCallback(async () => {
    if (isFrozen) return;
    setIsFrozen(true);
    console.log('[CephEditor] Confirm & Save — freezing manual modifications, marshaling coords…');

    const baseUrl = (import.meta.env && (import.meta.env as any).VITE_API_URL) || 'http://localhost:8123';
    const payload = {
      image_name: imageFile.name,
      image_width: img ? img.width : 0,
      image_height: img ? img.height : 0,
      keypoints: keypoints.map(kp => ({ name: kp.name, x: kp.x, y: kp.y, confidence: kp.confidence })),
      polygons: polygons.map(poly => ({ name: poly.name, points: poly.points })),
    };

    try {
      const res = await fetch(`${baseUrl}/api/v1/recalculate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      console.log('[CephEditor] Confirm & Save — live recalculation complete:', data);
      if (onRecalculate) {
        onRecalculate(data.data || data);
      }
      setIsFrozen(false);
    } catch (err) {
      console.error('[CephEditor] Confirm & Save failed:', err);
      // unfreeze on failure so user can retry
      setIsFrozen(false);
    }
  }, [isFrozen, imageFile, keypoints, polygons, img, onRecalculate]);

  // Re-enable editing on canvas when unfrozen
  useEffect(() => {
    if (!isFrozen) return;
    const el = containerRef.current;
    if (!el) return;
    const block = (e: WheelEvent) => e.preventDefault();
    el.addEventListener('wheel', block, { passive: false });
    return () => el.removeEventListener('wheel', block);
  }, [isFrozen]);

  // ── Sync stale-proof ref with state ──────────────────────────────────────────
  useEffect(() => { scaleRef.current = stageScale; }, [stageScale]);

  // ── Load image ───────────────────────────────────────────────────────────────
  useEffect(() => {
    const url = URL.createObjectURL(imageFile);
    const el  = new window.Image();
    el.onload = () => setImg(el);
    el.src    = url;
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  // ── Observe container size ───────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => { setStageW(el.offsetWidth); setStageH(el.offsetHeight); });
    ro.observe(el);
    setStageW(el.offsetWidth);
    setStageH(el.offsetHeight);
    return () => ro.disconnect();
  }, []);

  // ── Block passive browser scroll over canvas ─────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const block = (e: WheelEvent) => e.preventDefault();
    el.addEventListener('wheel', block, { passive: false });
    return () => el.removeEventListener('wheel', block);
  }, []);

  // ── Reactive init mapping syncing dynamic parent metrics ─────────────────────
  useEffect(() => {
    if (!img) return;
    const { width: w, height: h } = img;
    // No fallback templates — render exactly what the API provides, or nothing
    setKeypoints(initialKeypoints ?? []);
    setPolygons(initialPolygons ?? []);
  }, [img, initialKeypoints, initialPolygons]);

  // ── Fit-to-stage transform (content / Layer coordinates) ─────────────────────
  const { offX, offY, fitScale } = useMemo(() => {
    if (!img || stageW === 0 || stageH === 0) return { offX: 0, offY: 0, fitScale: 1 };
    const s = Math.min(stageW / img.width, stageH / img.height);
    return { fitScale: s, offX: (stageW - img.width * s) / 2, offY: (stageH - img.height * s) / 2 };
  }, [img, stageW, stageH]);

  const toContent = useCallback(
    (ix: number, iy: number): [number, number] => [ix * fitScale + offX, iy * fitScale + offY],
    [fitScale, offX, offY]
  );
  const toImage = useCallback(
    (cx: number, cy: number): [number, number] => [(cx - offX) / fitScale, (cy - offY) / fitScale],
    [fitScale, offX, offY]
  );

  // ── Propagate changes upward ──────────────────────────────────────────────────
  useEffect(() => { onKeypointsChange?.(keypoints); }, [keypoints]); // eslint-disable-line
  useEffect(() => { onPolygonsChange?.(polygons); },   [polygons]);  // eslint-disable-line

  // ── Zoom (mouse wheel) ───────────────────────────────────────────────────────
  const handleWheel = useCallback((e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;

    const scaleBy = 1.05; // Smooth increment
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    // 1. Calculate the position of the pointer relative to the un-scaled stage
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };

    // 2. Determine new scale based on wheel direction
    const direction = e.evt.deltaY > 0 ? -1 : 1; 
    let newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;
    newScale = Math.max(0.1, Math.min(newScale, 50)); // Allow deep zoom

    // 3. Calculate new position to keep the pointer exactly over the same image pixel
    const newPos = {
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    };

    // 4. Update scale and position on the stage instance and update React state
    stage.scale({ x: newScale, y: newScale });
    stage.position(newPos);
    scaleRef.current = newScale;
    setStageScale(newScale);
    stage.batchDraw();
  }, []);

  const resetZoom = useCallback(() => {
    scaleRef.current = 1;
    setStageScale(1);
    const stage = stageRef.current;
    if (stage) {
      stage.scale({ x: 1, y: 1 });
      stage.position({ x: 0, y: 0 });
      stage.batchDraw();
    }
  }, []);

  // ── Shape event handlers ──────────────────────────────────────────────────────

  const moveKp = useCallback((id: string, e: KonvaEventObject<DragEvent>) => {
    if (isFrozen) { e.target.stopDrag(); return; }
    const [ix, iy] = toImage(e.target.x(), e.target.y());
    console.log(`[CephEditor] onDragEnd "${id}" → image x:${ix.toFixed(2)}, y:${iy.toFixed(2)}`);
    setKeypoints(prev => prev.map(k => k.id === id ? { ...k, x: ix, y: iy } : k));
  }, [isFrozen, toImage]);

  const moveVertex = useCallback((polyId: string, vi: number, e: KonvaEventObject<DragEvent>) => {
    if (isFrozen) { e.target.stopDrag(); return; }
    const [ix, iy] = toImage(e.target.x(), e.target.y());
    setPolygons(prev => prev.map(p => {
      if (p.id !== polyId) return p;
      const pts = [...p.points];
      pts[vi*2] = ix; pts[vi*2+1] = iy;
      return { ...p, points: pts };
    }));
  }, [isFrozen, toImage]);

  const deleteVertex = useCallback((polyId: string, vi: number, e: KonvaEventObject<MouseEvent>) => {
    e.cancelBubble = true;
    pushHistory();
    setPolygons(prev => prev.map(p => {
      if (p.id !== polyId || p.points.length <= 6) return p;
      const pts = [...p.points];
      pts.splice(vi*2, 2);
      return { ...p, points: pts };
    }));
  }, [pushHistory]);

  const handleStageClick = useCallback((e: KonvaEventObject<MouseEvent>) => {
    if (e.target === e.target.getStage()) {
      if (!e.evt.shiftKey) {
        setSelectedId(null);
        return;
      }
    }

    if (isFrozen) return;

    if (e.evt.shiftKey) {
      const stage = stageRef.current;
      if (!stage) return;
      
      const raw      = stage.getPointerPosition()!;
      const sz       = scaleRef.current;
      const cx       = (raw.x - (stage.x() as number)) / sz;
      const cy       = (raw.y - (stage.y() as number)) / sz;
      const [ix, iy] = toImage(cx, cy);

      let bestPolyId: string | null = null;
      let bestEdge: { insertAt: number; px: number; py: number; dist: number } | null = null;
      let minGlobalDist = Infinity;

      polygons.forEach(p => {
        if (!showPolygons) return;
        const edge = nearestEdgeInsert(p.points, ix, iy);
        if (edge.dist < minGlobalDist) {
          minGlobalDist = edge.dist;
          bestPolyId = p.id;
          bestEdge = edge;
        }
      });

      const screenDist = minGlobalDist * fitScale * stageScale;
      if (bestPolyId && bestEdge && screenDist < 20) {
        pushHistory();
        setPolygons(prev => prev.map(p => {
          if (p.id !== bestPolyId) return p;
          const pts = [...p.points];
          pts.splice(bestEdge!.insertAt, 0, bestEdge!.px, bestEdge!.py);
          return { ...p, points: pts };
        }));
        setSelectedId(bestPolyId);
      }
    }
  }, [polygons, showPolygons, fitScale, stageScale, pushHistory, toImage, isFrozen]);

  const setCursor = useCallback((e: KonvaEventObject<MouseEvent>, cur: string) => {
    e.target.getStage()?.container().style.setProperty('cursor', cur);
  }, []);

  // ── Debug overlay: screen → image coordinate tracer ──────────────────────────
  const handleMouseMove = useCallback((e: KonvaEventObject<MouseEvent>) => {
    const stage = stageRef.current;
    if (!stage) return;
    const raw = stage.getPointerPosition();
    if (!raw) return;
    const sz = scaleRef.current;
    // screen → content space (undo stage pan + scale)
    const cx = (raw.x - (stage.x() as number)) / sz;
    const cy = (raw.y - (stage.y() as number)) / sz;
    // content → image space (undo fit-scale + letterbox offset)
    const [ix, iy] = toImage(cx, cy);
    setDebugInfo({ x: Math.round(raw.x), y: Math.round(raw.y), imageX: Math.round(ix), imageY: Math.round(iy) });
  }, [toImage]);

  const selectedName = useMemo(
    () => [...keypoints, ...polygons].find(s => s.id === selectedId)?.name,
    [selectedId, keypoints, polygons]
  );

  // ── Derived: are image coords outside the original image bounds? ─────────────
  const imgOob = img
    ? debugInfo.imageX < 0 || debugInfo.imageX > img.width
      || debugInfo.imageY < 0 || debugInfo.imageY > img.height
    : false;

  // ── Export / Import JSON (round-trip test) ────────────────────────────────────
  const handleExport = useCallback(() => {
    if (!img) return;
    const payload = {
      version: 1,
      imageName: imageFile.name,
      imageWidth: img.width,
      imageHeight: img.height,
      keypoints,
      polygons,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'ceph-debug-export.json';
    a.click();
    URL.revokeObjectURL(url);
  }, [img, imageFile, keypoints, polygons]);

  const handleImportFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target?.result as string);
        if (!Array.isArray(data.keypoints) || !Array.isArray(data.polygons)) {
          console.error('[CephEditor] Import failed: missing keypoints or polygons arrays');
          return;
        }
        if (img && (data.imageWidth !== img.width || data.imageHeight !== img.height)) {
          console.warn(`[CephEditor] Import dimension mismatch: file=${data.imageWidth}×${data.imageHeight} current=${img.width}×${img.height}`);
        }
        setKeypoints(data.keypoints);
        setPolygons(data.polygons);
        console.log(`[CephEditor] Import OK — ${data.keypoints.length} keypoints, ${data.polygons.length} polygons`);
      } catch (err) {
        console.error('[CephEditor] Import parse error:', err);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  }, [img]);

  // ── Toolbar kbd chip ──────────────────────────────────────────────────────────
  const chip = 'bg-white/10 border border-white/20 text-white/80 px-1.5 py-0.5 rounded font-mono text-[10px]';

  return (
    <div className="w-full h-full border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">

      {/* ── Canvas container (fills all space, anchors floating toolbar) ──── */}
      <div ref={containerRef} className="relative w-full h-full overflow-hidden bg-slate-950">

        {img && stageW > 0 && stageH > 0 && (
          <Stage
            ref={stageRef}
            width={stageW}
            height={stageH}
            scaleX={stageScale}
            scaleY={stageScale}
            draggable
            onWheel={handleWheel}
            onClick={handleStageClick}
            onMouseEnter={(e) => setCursor(e, 'grab')}
            onMouseLeave={(e) => setCursor(e, 'default')}
            onMouseDown={(e) => { if (e.target === e.target.getStage()) setCursor(e, 'grabbing'); }}
            onMouseUp={(e) => setCursor(e, 'grab')}
            onMouseMove={handleMouseMove}
          >
            <Layer>
              {/* X-ray image */}
              <KonvaImage
                image={img}
                x={offX} y={offY}
                width={img.width * fitScale}
                height={img.height * fitScale}
                listening={false}
              />

              {/* ── Polygons (Shapes Only) ──────────────────────────────────────────────── */}
              {showPolygons && [...polygons].sort((a, b) => a.name === 'Upper_incisor' ? 1 : b.name === 'Upper_incisor' ? -1 : 0).map((poly) => {
                const stagePts: number[] = [];
                for (let i = 0; i < poly.points.length; i += 2) {
                  const [cx, cy] = toContent(poly.points[i], poly.points[i+1]);
                  stagePts.push(cx, cy);
                }
                const isSel = selectedId === poly.id;

                return (
                  <Group key={`shape-${poly.id}`}>
                    <Line
                      points={stagePts}
                      closed
                      fill={poly.fill}
                      stroke={isSel ? '#ffffff' : poly.stroke}
                      strokeWidth={(isSel ? 2 : 1.5) / stageScale}
                      hitStrokeWidth={10 / stageScale}
                      listening={false}
                    />
                    {/* Polygon label — shadow replaces stroke for legibility */}
                    <Text
                      x={stagePts[0] ?? 0}
                      y={(stagePts[1] ?? 0) - (16 / stageScale)}
                      text={poly.name}
                      fontSize={11 / stageScale} fontStyle="bold"
                      fill="white"
                      shadowColor="black" shadowBlur={4} shadowOpacity={1}
                      shadowOffsetX={1} shadowOffsetY={1}
                      listening={false}
                    />
                  </Group>
                );
              })}

              {/* ── Keypoints ─────────────────────────────────────────────── */}
              {showLandmarks && keypoints.map((kp) => {
                const [kx, ky] = toContent(kp.x, kp.y);
                const isSel = selectedId === kp.id;
                const conf  = kp.confidence;

                // Three-tier confidence tier
                const isCritical = conf !== undefined && conf < 0.70;
                const isWarning  = conf !== undefined && conf >= 0.70 && conf < 0.85;

                // Ring colour: critical > warning > normal (selected overrides all)
                const ringColor = isSel
                  ? KP_RING_SELECTED
                  : isCritical ? KP_RING_CRITICAL
                  : isWarning  ? KP_RING_WARNING
                  : KP_RING_NORMAL;

                if (isCritical) {
                  console.warn(
                    `[CephEditor] ⚠ CRITICAL low confidence — "${kp.name}": conf=${conf?.toFixed(3)} < 0.70  ` +
                    `(x=${kp.x.toFixed(1)}, y=${kp.y.toFixed(1)}). Manual review required.`
                  );
                } else if (isWarning) {
                  console.warn(
                    `[CephEditor] ⚠ Low confidence — "${kp.name}": conf=${conf?.toFixed(3)} < 0.85. ` +
                    `Review recommended.`
                  );
                }

                return (
                  <Group key={kp.id}>
                    {/* Critical outer ring — red */}
                    {isCritical && (
                      <Circle
                        x={kx} y={ky}
                        radius={(pointSize + 5) / stageScale}
                        stroke={KP_RING_CRITICAL}
                        strokeWidth={2.0 / stageScale}
                        opacity={0.9}
                        listening={false}
                      />
                    )}
                    {/* Warning outer ring — yellow (subtle, wider) */}
                    {isWarning && (
                      <Circle
                        x={kx} y={ky}
                        radius={(pointSize + 4) / stageScale}
                        stroke={KP_RING_WARNING}
                        strokeWidth={1.5 / stageScale}
                        opacity={0.8}
                        listening={false}
                      />
                    )}
                    <Circle
                      x={kx} y={ky}
                      radius={pointSize / stageScale}
                      hitStrokeWidth={20 / stageScale}
                      fill={isSel ? KP_DOT_SELECTED : KP_DOT_NORMAL}
                      stroke={ringColor}
                      strokeWidth={1.5 / stageScale}
                      draggable
                      onDragStart={pushHistory}
                      onDragEnd={(e) => moveKp(kp.id, e)}
                      onClick={() => setSelectedId(kp.id)}
                      onMouseEnter={(e) => setCursor(e, 'move')}
                      onMouseLeave={(e) => setCursor(e, 'grab')}
                    />
                    {/* Keypoint label */}
                    <Text
                      x={kx + (8 / stageScale)} y={ky - (6 / stageScale)}
                      text={
                        kp.name +
                        (isCritical ? ' ⚠ CRITICAL' : isWarning ? ' ⚠' : '')
                      }
                      fontSize={10 / stageScale} fontStyle="bold"
                      fill={isCritical ? '#fca5a5' : isWarning ? '#fde68a' : 'white'}
                      shadowColor="black" shadowBlur={4} shadowOpacity={1}
                      shadowOffsetX={1} shadowOffsetY={1}
                      listening={false}
                    />
                  </Group>
                );
              })}

              {/* ── Plan B 3-level measurement lines (bone thickness) ─────────── */}
              {showMeasurementLines && (() => {
                // Resolve boneThickness: prefer live API data, fall back to a visible
                // mock so the UI is never blank — makes debugging much easier.
                const live = boneThickness;
                const mockLines: Lines3Level | null = (() => {
                  if (!img || !live) {
                    // Use image midpoint landmarks as approximate labial/palatal anchor
                    // so the mock lines are actually visible on whatever image is loaded.
                    const cx = img ? img.width  / 2 : 800;
                    const cy = img ? img.height / 2 : 900;
                    return {
                      cervical: {
                        palatal_distance_mm: 1.9,
                        palatal_tooth_x: cx - 60, palatal_tooth_y: cy - 30,
                        palatal_bone_x:  cx - 110, palatal_bone_y: cy - 30,
                        labial_distance_mm: 2.4,
                        labial_tooth_x: cx + 60,  labial_tooth_y: cy - 30,
                        labial_bone_x:  cx + 110, labial_bone_y:  cy - 30,
                      },
                      middle: {
                        palatal_distance_mm: 1.5,
                        palatal_tooth_x: cx - 55, palatal_tooth_y: cy,
                        palatal_bone_x:  cx - 105, palatal_bone_y: cy,
                        labial_distance_mm: 1.8,
                        labial_tooth_x: cx + 55,  labial_tooth_y: cy,
                        labial_bone_x:  cx + 105, labial_bone_y:  cy,
                      },
                      apical: {
                        palatal_distance_mm: 0.9,
                        palatal_tooth_x: cx - 50, palatal_tooth_y: cy + 30,
                        palatal_bone_x:  cx - 95,  palatal_bone_y: cy + 30,
                        labial_distance_mm: 0.6,
                        labial_tooth_x: cx + 50,  labial_tooth_y: cy + 30,
                        labial_bone_x:  cx + 95,  labial_bone_y:  cy + 30,
                      },
                    };
                  }
                  return null;
                })();

                if (!live) {
                  // eslint-disable-next-line no-console
                  console.warn(
                    '[CephEditor] boneThickness prop is undefined — Plan B lines use mock data.',
                    'Check that the backend API returns bone_thickness.lines_3_level.',
                    mockLines,
                  );
                }

                const bt = live ?? (globalMinLines ? null : mockLines);
                if (!bt) return null;

                const levels: Array<{
                  key: 'cervical' | 'middle' | 'apical';
                  label: string;
                  color: string;
                  dotFill: string;
                }> = [
                  { key: 'cervical', label: 'C', color: '#06b6d4', dotFill: '#67e8f9' },  // Cyan   — cervical
                  { key: 'middle',   label: 'M', color: '#f472b6', dotFill: '#f9a8d4' },  // Pink   — middle
                  { key: 'apical',   label: 'A', color: '#4ade80', dotFill: '#86efac' },  // Green  — apical
                ];

                return levels.map(({ key, label, color, dotFill }) => {
                  const lv = bt[key];
                  if (!lv) return null;

                  // ── Segment colours ──────────────────────────────────────────────
                  // Palatal gap:  blue-violet  (always this colour)
                  // Labial  gap:  warm amber    (always this colour)
                  const PALATAL_COL  = '#a78bfa';   // violet — palatal side
                  const LABIAL_COL   = '#fb923c';   // orange  — labial side
                  const PALATAL_DOT  = '#c4b5fd';
                  const LABIAL_DOT   = '#fdba74';

                  const SW   = 1.5 / stageScale;   // stroke width (crisp at all zoom levels)
                  const dotR = 3.0 / stageScale;    // endpoint circle radius

                  // Helper: convert image-px → stage-px via toContent()
                  const toStage = (imgX: number, imgY: number) => {
                    const [sx, sy] = toContent(imgX, imgY);
                    if (Number.isNaN(sx) || Number.isNaN(sy)) return null;
                    return [sx, sy] as [number, number];
                  };

                  // ── Palatal gap segment ─────────────────────────────────────────
                  const pTooth = toStage(lv.palatal_tooth_x, lv.palatal_tooth_y);
                  const pBone  = toStage(lv.palatal_bone_x,  lv.palatal_bone_y);

                  // ── Labial gap segment ─────────────────────────────────────────
                  const lTooth = toStage(lv.labial_tooth_x,  lv.labial_tooth_y);
                  const lBone  = toStage(lv.labial_bone_x,   lv.labial_bone_y);

                  if (!pTooth || !pBone || !lTooth || !lBone) return null;

                  const [pTx, pTy] = pTooth;
                  const [pBx, pBy] = pBone;
                  const [lBx, lBy] = lBone;
                  const [lTx, lTy] = lTooth;

                  // Vertical offset so palatal labels sit above, labial below
                  const labelOff = 12 / stageScale;

                  return (
                    <Group key={key} listening={false}>
                      {/* ── Palatal gap (tooth surface → palatal bone) ────────────── */}
                      <Line
                        points={[pTx, pTy, pBx, pBy]}
                        stroke={PALATAL_COL}
                        strokeWidth={SW}
                        opacity={0.95}
                      />
                      {/* Palatal tooth-side endpoint */}
                      <Line
                        points={[pTx - dotR, pTy, pTx + dotR, pTy]}
                        stroke={PALATAL_DOT}
                        strokeWidth={1 / stageScale}
                        opacity={1}
                      />
                      {/* Palatal bone-side endpoint */}
                      <Line
                        points={[pBx - dotR, pBy, pBx + dotR, pBy]}
                        stroke={PALATAL_DOT}
                        strokeWidth={1 / stageScale}
                        opacity={1}
                      />
                      {/* Palatal label — "P: X.Xmm" above the palatal gap */}
                      <Text
                        x={(pTx + pBx) / 2 - (18 / stageScale)}
                        y={Math.min(pTy, pBy) - labelOff - (8 / stageScale)}
                        text={`P: ${lv.palatal_distance_mm.toFixed(1)}mm`}
                        fontSize={9 / stageScale}
                        fontStyle="bold"
                        fill={PALATAL_COL}
                        shadowColor="black"
                        shadowBlur={3 / stageScale}
                        shadowOpacity={0.7}
                        shadowOffsetX={1 / stageScale}
                        shadowOffsetY={1 / stageScale}
                      />

                      {/* ── Labial gap (labial bone → tooth surface) ────────────── */}
                      <Line
                        points={[lBx, lBy, lTx, lTy]}
                        stroke={LABIAL_COL}
                        strokeWidth={SW}
                        opacity={0.95}
                      />
                      {/* Labial bone-side endpoint */}
                      <Line
                        points={[lBx - dotR, lBy, lBx + dotR, lBy]}
                        stroke={LABIAL_DOT}
                        strokeWidth={1 / stageScale}
                        opacity={1}
                      />
                      {/* Labial tooth-side endpoint */}
                      <Line
                        points={[lTx - dotR, lTy, lTx + dotR, lTy]}
                        stroke={LABIAL_DOT}
                        strokeWidth={1 / stageScale}
                        opacity={1}
                      />
                      {/* Labial label — "L: X.Xmm" below the labial gap */}
                      <Text
                        x={(lTx + lBx) / 2 + (4 / stageScale)}
                        y={Math.max(lTy, lBy) + (3 / stageScale)}
                        text={`L: ${lv.labial_distance_mm.toFixed(1)}mm`}
                        fontSize={9 / stageScale}
                        fontStyle="bold"
                        fill={LABIAL_COL}
                        shadowColor="black"
                        shadowBlur={3 / stageScale}
                        shadowOpacity={0.7}
                        shadowOffsetX={1 / stageScale}
                        shadowOffsetY={1 / stageScale}
                      />

                      {/* ── Level badge (C / M / A) centred between palatal & labial gaps */}
                      <Text
                        x={(pBx + lBx) / 2 - (6 / stageScale)}
                        y={(pBy + lBy) / 2 - (5 / stageScale)}
                        text={label}
                        fontSize={10 / stageScale}
                        fontStyle="bold"
                        fill="white"
                        shadowColor={color}
                        shadowBlur={6 / stageScale}
                        shadowOpacity={0.9}
                      />
                    </Group>
                  );
                });
              })()}

              {/* ── Global Min Distance mode — 2 bold bottleneck lines ───────────────
                   Rendered ONLY when globalMinLines prop is present.
                   The standard boneThickness mock is suppressed in this mode
                   so exactly 2 lines are visible (labial + palatal minimum). */}
              {showMeasurementLines && globalMinLines && (() => {
                const LABIAL_COL  = '#fb923c';   // Warm orange  — labial bottleneck
                const PALATAL_COL = '#a78bfa';   // Violet       — palatal bottleneck
                const SW  = 2.5 / stageScale;    // Thicker than standard lines for emphasis
                const dotR = 4.0 / stageScale;

                const toStage = (imgX: number, imgY: number) => {
                  const [sx, sy] = toContent(imgX, imgY);
                  if (Number.isNaN(sx) || Number.isNaN(sy)) return null;
                  return [sx, sy] as [number, number];
                };

                const lines = [
                  {
                    seg: globalMinLines.labial_line,
                    color: LABIAL_COL,
                    label: `⚡ L min: ${globalMinLines.labial_mm.toFixed(2)}mm`,
                    id: 'gmin-labial',
                  },
                  {
                    seg: globalMinLines.palatal_line,
                    color: PALATAL_COL,
                    label: `⚡ P min: ${globalMinLines.palatal_mm.toFixed(2)}mm`,
                    id: 'gmin-palatal',
                  },
                ];

                return lines.map(({ seg, color, label, id }) => {
                  if (!Array.isArray(seg) || seg.length < 2) return null;
                  const s0 = toStage(seg[0][0], seg[0][1]);
                  const s1 = toStage(seg[1][0], seg[1][1]);
                  if (!s0 || !s1) return null;
                  const [x1, y1] = s0;
                  const [x2, y2] = s1;
                  const mx = (x1 + x2) / 2;
                  const my = (y1 + y2) / 2;

                  return (
                    <Group key={id} listening={false}>
                      {/* Main bottleneck line */}
                      <Line
                        points={[x1, y1, x2, y2]}
                        stroke={color}
                        strokeWidth={SW}
                        opacity={0.95}
                      />
                      {/* Tooth-surface endpoint tick */}
                      <Line
                        points={[x1 - dotR, y1, x1 + dotR, y1]}
                        stroke={color}
                        strokeWidth={1.5 / stageScale}
                        opacity={1}
                      />
                      {/* Bone-surface endpoint tick */}
                      <Line
                        points={[x2 - dotR, y2, x2 + dotR, y2]}
                        stroke={color}
                        strokeWidth={1.5 / stageScale}
                        opacity={1}
                      />
                      {/* mm label at midpoint */}
                      <Text
                        x={mx + 5 / stageScale}
                        y={my - 12 / stageScale}
                        text={label}
                        fontSize={10 / stageScale}
                        fontStyle="bold"
                        fill={color}
                        shadowColor="black"
                        shadowBlur={4 / stageScale}
                        shadowOpacity={0.85}
                        shadowOffsetX={1 / stageScale}
                        shadowOffsetY={1 / stageScale}
                      />
                    </Group>
                  );
                });
              })()}

              {/* ── Polygons (Vertex Handles Only, Rendered Last/Top) ────────────── */}
              {showPolygons && polygons.map((poly) => {
                const isSel = selectedId === poly.id;
                const nv    = poly.points.length / 2;
                return (
                  <Group key={`handles-${poly.id}`}>
                    {Array.from({ length: nv }, (_, vi) => {
                      const [vx, vy] = toContent(poly.points[vi*2], poly.points[vi*2+1]);
                      return (
                        <Circle
                          key={vi}
                          x={vx} y={vy}
                          radius={pointSize / stageScale}
                          hitStrokeWidth={40 / stageScale}
                          fill={isSel ? '#ffffff' : poly.stroke}
                          stroke={poly.stroke}
                          strokeWidth={1 / stageScale}
                          draggable
                          onDragStart={(e) => {
                            e.target.moveToTop();
                            pushHistory();
                          }}
                          onDragMove={(e) => moveVertex(poly.id, vi, e)}
                          onDblClick={(e) => { pushHistory(); deleteVertex(poly.id, vi, e); }}
                          onClick={(e) => {
                            setSelectedId(poly.id);
                            if (e.evt.altKey) { pushHistory(); deleteVertex(poly.id, vi, e); }
                            e.cancelBubble = true;
                          }}
                          onMouseEnter={(e) => {
                            setCursor(e, 'pointer');
                            e.target.scale({ x: 1.5, y: 1.5 });
                          }}
                          onMouseLeave={(e) => {
                            setCursor(e, 'default');
                            e.target.scale({ x: 1, y: 1 });
                          }}
                        />
                      );
                    })}
                  </Group>
                );
              })}
            </Layer>
          </Stage>
        )}

        {!img && (
          <div className="flex items-center justify-center h-full text-slate-500 text-sm">
            Loading image…
          </div>
        )}

        {/* ── Dev panel (top-right): coord overlay + Export/Import ────────── */}
        {img && isDebugMode && (
          <div className="absolute top-3 right-3 z-50 flex flex-col gap-2 items-end">
            {/* Coordinate overlay */}
            <div className="bg-black/85 backdrop-blur-sm font-mono text-[11px] leading-relaxed px-3 py-2.5 rounded-lg border border-white/10 pointer-events-none select-none">
              <div className="text-white/35 text-[9px] uppercase tracking-widest mb-1.5">Debug Coords</div>
              <div className="flex items-center gap-3">
                <span className="text-white/40 w-10 text-[10px]">Screen</span>
                <span className="text-white tabular-nums">{debugInfo.x}, {debugInfo.y}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-white/40 w-10 text-[10px]">Image</span>
                <span className={`tabular-nums ${imgOob ? 'text-red-400' : 'text-green-400'}`}>
                  {debugInfo.imageX}, {debugInfo.imageY}
                  {imgOob && <span className="ml-1 text-red-400">⚠ OOB</span>}
                </span>
              </div>
              <div className="text-white/25 text-[9px] mt-1.5 tabular-nums">
                max {img.width} × {img.height} px
              </div>
            </div>
            {/* Export / Import actions */}
            <div className="bg-black/80 backdrop-blur-sm border border-amber-500/25 rounded-lg flex overflow-hidden text-[11px]">
              <button
                onClick={handleExport}
                title="Export keypoints and polygons as JSON"
                className="px-3 py-1.5 text-amber-300/80 hover:text-amber-200 hover:bg-amber-500/10 transition-colors"
              >
                Export JSON
              </button>
              <span className="w-px bg-white/10 self-stretch" />
              <button
                onClick={() => importFileRef.current?.click()}
                title="Import keypoints and polygons from JSON"
                className="px-3 py-1.5 text-amber-300/80 hover:text-amber-200 hover:bg-amber-500/10 transition-colors"
              >
                Import JSON
              </button>
            </div>
          </div>
        )}

        {/* ── Adapted Multi-Line Responsive Glass Pill Toolbar ──────────────── */}
        {img && (
          isToolbarOpen ? (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1.5 w-[95%] md:w-full z-50 animate-fade-in pointer-events-none">

              {/* Instructions helper row */}
              <div className="hidden sm:flex items-center gap-1.5 text-[11px] text-slate-200 bg-black/60 px-3 py-1 rounded-full backdrop-blur-sm border border-white/10 pointer-events-auto select-none font-medium">
                <kbd className="bg-white/15 border border-white/20 text-white/90 px-1.5 py-0.5 rounded font-mono text-[10px]">Drag</kbd>
                <span className="text-white/40">move</span>
                <span className="text-white/20 mx-1">·</span>
                <kbd className="bg-white/15 border border-white/20 text-white/90 px-1.5 py-0.5 rounded font-mono text-[10px]">Scroll</kbd>
                <span className="text-white/40">zoom</span>
                <span className="text-white/20 mx-1">·</span>
                <kbd className="bg-white/15 border border-white/20 text-white/90 px-1.5 py-0.5 rounded font-mono text-[10px]">⇧+Click</kbd>
                <span className="text-white/40">add pt</span>
                <span className="text-white/20 mx-1">·</span>
                <kbd className="bg-white/15 border border-white/20 text-white/90 px-1.5 py-0.5 rounded font-mono text-[10px]">DblClick</kbd>
                <span className="text-white/40">del pt</span>
              </div>

              {/* Main functional control bar */}
              <div className="w-full md:w-auto bg-black/90 backdrop-blur-md text-white px-5 py-2.5 rounded-xl md:rounded-full border border-white/15 flex flex-wrap md:flex-nowrap gap-x-4 gap-y-2 text-xs items-center justify-center shadow-2xl pointer-events-auto">

                {/* Visibility toggles */}
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-1.5 cursor-pointer select-none whitespace-nowrap text-white font-medium">
                    <input
                      type="checkbox" checked={showLandmarks}
                      onChange={(e) => setShowLandmarks(e.target.checked)}
                      className="accent-amber-400 w-3.5 h-3.5 cursor-pointer rounded"
                    />
                    <span className="text-white/90">Landmarks</span>
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer select-none whitespace-nowrap text-white font-medium">
                    <input
                      type="checkbox" checked={showPolygons}
                      onChange={(e) => setShowPolygons(e.target.checked)}
                      className="accent-cyan-400 w-3.5 h-3.5 cursor-pointer rounded"
                    />
                    <span className="text-white/90">Polygons</span>
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer select-none whitespace-nowrap text-white font-medium">
                    <input
                      type="checkbox" checked={showMeasurementLines}
                      onChange={(e) => setShowMeasurementLines(e.target.checked)}
                      className="accent-pink-400 w-3.5 h-3.5 cursor-pointer rounded"
                    />
                    <span className="text-white/90">Lines</span>
                  </label>
                </div>

                <span className="text-white/25 select-none font-light">|</span>

                {/* Point size slider */}
                <label className="flex items-center gap-2 select-none whitespace-nowrap">
                  <span className="text-white/60 text-[11px] hidden sm:inline font-medium">Size</span>
                  <input
                    type="range" min="1" max="10" step="0.5"
                    value={pointSize}
                    onChange={(e) => setPointSize(Number(e.target.value))}
                    className="w-14 sm:w-16 accent-orange-400 cursor-pointer"
                  />
                  <span className="tabular-nums font-mono w-5 text-right text-white/80">{pointSize}</span>
                </label>

                {/* Zoom indicator */}
                {stageScale > 1 && (
                  <>
                    <span className="text-white/25 select-none font-light">|</span>
                    <button
                      onClick={resetZoom}
                      className="flex items-center justify-center gap-0.5 text-white/70 hover:text-white transition-colors whitespace-nowrap w-12 text-center tabular-nums font-mono bg-white/10 hover:bg-white/20 px-2 py-1 rounded text-[11px] font-medium border border-white/10"
                      title="Reset Zoom"
                    >
                      {Math.round(stageScale * 100)}%
                    </button>
                  </>
                )}

                <span className="text-white/25 select-none font-light">|</span>

                {/* Undo */}
                <button
                  onClick={undo}
                  disabled={historyStack.length === 0}
                  title="Undo last change"
                  className={`flex items-center justify-center gap-1 px-3 py-1 rounded-lg transition-colors whitespace-nowrap text-[11px] font-semibold border ${
                    historyStack.length === 0
                      ? 'text-white/25 border-white/5 cursor-not-allowed bg-white/5'
                      : 'text-white/80 border-white/20 hover:bg-white/10 hover:text-white hover:border-white/30'
                  }`}
                >
                  ↩ Undo
                </button>

                <span className="text-white/25 select-none font-light">|</span>

                <span className="text-white/25 select-none font-light">|</span>

                {/* Reset to AI Default */}
                {originalAnalysis && (
                  <button
                    onClick={handleResetToAI}
                    disabled={isFrozen}
                    title="Revert all manual edits back to original AI prediction"
                    className={`flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-bold transition-all whitespace-nowrap border ${
                      isFrozen
                        ? 'bg-white/10 border-white/20 text-white/40 cursor-not-allowed'
                        : 'text-rose-400 border-rose-500/40 hover:bg-rose-500/20 hover:text-rose-300'
                    }`}
                  >
                    Reset to AI
                  </button>
                )}

                <span className="text-white/25 select-none font-light">|</span>

                {/* Confirm & Save — high-contrast amber pill */}
                <button
                  onClick={handleConfirmAndSave}
                  disabled={isFrozen}
                  title="Confirm manual edits and trigger live recalculation"
                  className={`flex items-center justify-center gap-1.5 px-4 py-1.5 rounded-full text-[11px] font-bold transition-all whitespace-nowrap shadow-md border ${
                    isFrozen
                      ? 'bg-white/10 border-white/20 text-white/40 cursor-not-allowed'
                      : 'bg-amber-400 hover:bg-amber-300 text-slate-900 hover:brightness-110 border-amber-500/40 shadow-amber-400/20'
                  }`}
                >
                  {isFrozen ? '✓ Saved' : 'Confirm & Save'}
                </button>

                {/* Selected element name */}
                {selectedName && (
                  <>
                    <span className="text-white/25 select-none font-light">|</span>
                    <span className="font-mono text-cyan-300 truncate max-w-[80px] sm:max-w-[100px] whitespace-nowrap text-[11px] font-semibold">
                      ● {selectedName}
                    </span>
                  </>
                )}

                <span className="text-white/25 select-none font-light">|</span>

                {/* Debug mode toggle */}
                <button
                  onClick={() => setIsDebugMode(v => !v)}
                  title={isDebugMode ? 'Hide debug tools' : 'Show debug tools'}
                  className={`px-2.5 py-1 rounded-lg transition-colors text-[11px] whitespace-nowrap font-semibold border ${
                    isDebugMode
                      ? 'bg-amber-500/25 text-amber-300 border-amber-500/40'
                      : 'text-white/50 border-white/10 hover:text-white/80 hover:border-white/20'
                  }`}
                >
                  Dev
                </button>

                <span className="text-white/25 select-none font-light">|</span>

                {/* Hide Toolbar Button */}
                <button
                  onClick={() => setIsToolbarOpen(false)}
                  title="Minimize toolbar"
                  className="text-white/40 hover:text-white/80 transition-colors p-1.5 rounded-lg hover:bg-white/10 border border-transparent hover:border-white/10"
                >
                  <span className="text-xs font-bold leading-none">✕</span>
                </button>

              </div>
            </div>
          ) : (
            <button
              onClick={() => setIsToolbarOpen(true)}
              title="Expand toolbar"
              className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/85 hover:bg-black text-white/80 hover:text-white px-3.5 py-1.5 rounded-full border border-white/10 flex items-center gap-1.5 text-xs backdrop-blur-md shadow-xl transition-all z-50 animate-fade-in"
            >
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              <span>Show Toolbar</span>
            </button>
          )
        )}

        {/* Hidden file input for JSON import */}
        <input
          ref={importFileRef}
          type="file"
          accept=".json"
          className="hidden"
          onChange={handleImportFile}
        />

      </div>
    </div>
  );
});

export default CephCanvasEditor;
