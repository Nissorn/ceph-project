import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Stage, Layer, Image as KonvaImage, Circle, Line, Text, Group } from 'react-konva';
import type { KonvaEventObject } from 'konva/lib/Node';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Keypoint {
  id: string;
  name: string;
  x: number; // image-space pixels
  y: number;
}

export interface PolygonShape {
  id: string;
  name: string;
  points: number[]; // flat [x1,y1,...] image-space pixels
  fill: string;
  stroke: string;
}

interface Props {
  imageFile: File;
  initialKeypoints?: Keypoint[];
  initialPolygons?: PolygonShape[];
  onKeypointsChange?: (kps: Keypoint[]) => void;
  onPolygonsChange?: (polys: PolygonShape[]) => void;
  measurementLines?: {
    labial_crest_line: number[][];
    labial_midroot_line: number[][];
    labial_apex_line: number[][];
    palatal_crest_line: number[][];
    palatal_midroot_line: number[][];
    palatal_apex_line: number[][];
  } | null;
}

// ── Medical imaging colour palette ───────────────────────────────────────────

const POLY_PALETTE = [
  { fill: 'rgba(6, 182, 212, 0.15)',  stroke: 'rgba(6, 182, 212, 0.9)'  },  // Cyan Accent    — Upper_incisor
  { fill: 'rgba(236, 72, 153, 0.15)', stroke: 'rgba(236, 72, 153, 0.9)' }, // Pink/Magenta — Labial_bone
  { fill: 'rgba(16, 185, 129, 0.15)', stroke: 'rgba(16, 185, 129, 0.9)' }, // Emerald Green  — Palatal_bone
];

const KP_RING_NORMAL   = '#fbbf24'; // Vibrant Amber
const KP_DOT_NORMAL    = '#ffffff';
const KP_RING_SELECTED = '#ffffff';
const KP_DOT_SELECTED  = '#f59e0b'; // Deep glowing Amber

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
): { insertAt: number; px: number; py: number } {
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

export default function CephCanvasEditor({
  imageFile, initialKeypoints, initialPolygons,
  onKeypointsChange, onPolygonsChange,
  measurementLines,
}: Props) {
  const containerRef  = useRef<HTMLDivElement>(null);
  const stageRef      = useRef<any>(null);
  const scaleRef      = useRef(1);
  const importFileRef = useRef<HTMLInputElement>(null);

  const [stageW, setStageW]               = useState(0);
  const [stageH, setStageH]               = useState(0);
  const [img, setImg]                     = useState<HTMLImageElement | null>(null);
  const [keypoints, setKeypoints]         = useState<Keypoint[]>([]);
  const [polygons, setPolygons]           = useState<PolygonShape[]>([]);
  const [selectedId, setSelectedId]       = useState<string | null>(null);
  const [stageScale, setStageScale]       = useState(1);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showPolygons, setShowPolygons]   = useState(true);
  const [showRulers, setShowRulers]       = useState(true);
  const [pointSize, setPointSize]         = useState(4);
  const [debugInfo, setDebugInfo]         = useState({ x: 0, y: 0, imageX: 0, imageY: 0 });
  const [isDebugMode, setIsDebugMode]     = useState(false);
  const [isToolbarOpen, setIsToolbarOpen] = useState(true);

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

  // ── Geometry: Visual Guides (U1 and PP lines, measurement projections) ─────
  const u1Line = useMemo(() => {
    const upperTip = keypoints.find(k => k.name === 'Upper_tip');
    const upperApex = keypoints.find(k => k.name === 'Upper_apex');
    if (!upperTip || !upperApex) return null;
    const dx = upperApex.x - upperTip.x;
    const dy = upperApex.y - upperTip.y;
    // Extend past tip by 15% and apex by 30% (upwards)
    const p1 = [upperTip.x - dx * 0.15, upperTip.y - dy * 0.15];
    const p2 = [upperApex.x + dx * 0.3, upperApex.y + dy * 0.3];
    const [cx1, cy1] = toContent(p1[0], p1[1]);
    const [cx2, cy2] = toContent(p2[0], p2[1]);
    return [cx1, cy1, cx2, cy2];
  }, [keypoints, toContent]);

  const ppLine = useMemo(() => {
    const ans = keypoints.find(k => k.name === 'ANS');
    const pns = keypoints.find(k => k.name === 'PNS');
    if (!ans || !pns) return null;
    const dx = pns.x - ans.x;
    const dy = pns.y - ans.y;
    // Extend past both by 20%
    const p1 = [ans.x - dx * 0.2, ans.y - dy * 0.2];
    const p2 = [pns.x + dx * 0.2, pns.y + dy * 0.2];
    const [cx1, cy1] = toContent(p1[0], p1[1]);
    const [cx2, cy2] = toContent(p2[0], p2[1]);
    return [cx1, cy1, cx2, cy2];
  }, [keypoints, toContent]);

  const visualProjectionLines = useMemo(() => {
    if (!measurementLines) return [];
    const lines = [
      { line: measurementLines.labial_crest_line, color: '#f59e0b' },
      { line: measurementLines.labial_midroot_line, color: '#ef4444' },
      { line: measurementLines.labial_apex_line, color: '#ef4444' },
      { line: measurementLines.palatal_crest_line, color: '#f59e0b' },
      { line: measurementLines.palatal_midroot_line, color: '#ef4444' },
      { line: measurementLines.palatal_apex_line, color: '#ef4444' },
    ];
    return lines
      .filter(item => item.line && item.line.length === 2)
      .map(item => {
        const [p1, p2] = item.line;
        const [cx1, cy1] = toContent(p1[0], p1[1]);
        const [cx2, cy2] = toContent(p2[0], p2[1]);
        return { points: [cx1, cy1, cx2, cy2], color: item.color };
      });
  }, [measurementLines, toContent]);

  // ── Propagate changes upward ──────────────────────────────────────────────────
  useEffect(() => { onKeypointsChange?.(keypoints); }, [keypoints]); // eslint-disable-line
  useEffect(() => { onPolygonsChange?.(polygons); },   [polygons]);  // eslint-disable-line

  // ── Zoom (mouse wheel) ───────────────────────────────────────────────────────
  const handleWheel = useCallback((e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;
    const oldScale = scaleRef.current;
    const pointer  = stage.getPointerPosition()!;
    const mouseAt  = {
      x: (pointer.x - (stage.x() as number)) / oldScale,
      y: (pointer.y - (stage.y() as number)) / oldScale,
    };
    const direction = e.evt.deltaY < 0 ? 1 : -1;
    const newScale  = Math.max(1.0, Math.min(12, oldScale * Math.pow(1.08, direction)));
    scaleRef.current = newScale;
    setStageScale(newScale);
    if (newScale <= 1.0) {
      stage.position({ x: 0, y: 0 });
    } else {
      stage.position({
        x: pointer.x - mouseAt.x * newScale,
        y: pointer.y - mouseAt.y * newScale,
      });
    }
  }, []);

  const resetZoom = useCallback(() => {
    scaleRef.current = 1;
    setStageScale(1);
    stageRef.current?.position({ x: 0, y: 0 });
  }, []);

  // ── Shape event handlers ──────────────────────────────────────────────────────

  const moveKp = useCallback((id: string, e: KonvaEventObject<DragEvent>) => {
    const [ix, iy] = toImage(e.target.x(), e.target.y());
    console.log(`[CephEditor] onDragEnd "${id}" → image x:${ix.toFixed(2)}, y:${iy.toFixed(2)}`);
    setKeypoints(prev => prev.map(k => k.id === id ? { ...k, x: ix, y: iy } : k));
  }, [toImage]);

  const moveVertex = useCallback((polyId: string, vi: number, e: KonvaEventObject<DragEvent>) => {
    const [ix, iy] = toImage(e.target.x(), e.target.y());
    setPolygons(prev => prev.map(p => {
      if (p.id !== polyId) return p;
      const pts = [...p.points];
      pts[vi*2] = ix; pts[vi*2+1] = iy;
      return { ...p, points: pts };
    }));
  }, [toImage]);

  const deleteVertex = useCallback((polyId: string, vi: number, e: KonvaEventObject<MouseEvent>) => {
    e.cancelBubble = true;
    setPolygons(prev => prev.map(p => {
      if (p.id !== polyId || p.points.length <= 6) return p;
      const pts = [...p.points];
      pts.splice(vi*2, 2);
      return { ...p, points: pts };
    }));
  }, []);

  const addEdgePoint = useCallback((polyId: string, e: KonvaEventObject<MouseEvent>) => {
    if (!e.evt.shiftKey) return;
    e.cancelBubble = true;
    const stage = stageRef.current;
    if (!stage) return;
    const raw      = stage.getPointerPosition()!;
    const sz       = scaleRef.current;
    const cx       = (raw.x - (stage.x() as number)) / sz;
    const cy       = (raw.y - (stage.y() as number)) / sz;
    const [ix, iy] = toImage(cx, cy);
    setPolygons(prev => prev.map(p => {
      if (p.id !== polyId) return p;
      const { insertAt, px, py } = nearestEdgeInsert(p.points, ix, iy);
      const pts = [...p.points];
      pts.splice(insertAt, 0, px, py);
      return { ...p, points: pts };
    }));
  }, [toImage]);

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
            onClick={(e) => { if (e.target === e.target.getStage()) setSelectedId(null); }}
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

              {/* ── Polygons ──────────────────────────────────────────────── */}
              {showPolygons && polygons.map((poly) => {
                const stagePts: number[] = [];
                for (let i = 0; i < poly.points.length; i += 2) {
                  const [cx, cy] = toContent(poly.points[i], poly.points[i+1]);
                  stagePts.push(cx, cy);
                }
                const isSel = selectedId === poly.id;
                const nv    = poly.points.length / 2;

                return (
                  <Group key={poly.id}>
                    <Line
                      points={stagePts}
                      closed
                      fill={poly.fill}
                      stroke={isSel ? '#ffffff' : poly.stroke}
                      strokeWidth={(isSel ? 2 : 1.5) / stageScale}
                      hitStrokeWidth={10 / stageScale}
                      onClick={(e) => { setSelectedId(poly.id); addEdgePoint(poly.id, e); }}
                      onMouseEnter={(e) => setCursor(e, 'pointer')}
                      onMouseLeave={(e) => setCursor(e, 'grab')}
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
                    {/* Vertex handles */}
                    {Array.from({ length: nv }, (_, vi) => {
                      const [vx, vy] = toContent(poly.points[vi*2], poly.points[vi*2+1]);
                      return (
                        <Circle
                          key={vi}
                          x={vx} y={vy}
                          radius={pointSize / stageScale}
                          hitStrokeWidth={20 / stageScale}
                          fill={isSel ? '#ffffff' : poly.stroke}
                          stroke={poly.stroke}
                          strokeWidth={1 / stageScale}
                          draggable
                          onDragMove={(e) => moveVertex(poly.id, vi, e)}
                          onDblClick={(e) => deleteVertex(poly.id, vi, e)}
                          onClick={(e) => {
                            setSelectedId(poly.id);
                            if (e.evt.altKey) deleteVertex(poly.id, vi, e);
                            e.cancelBubble = true;
                          }}
                          onMouseEnter={(e) => setCursor(e, 'crosshair')}
                          onMouseLeave={(e) => setCursor(e, 'grab')}
                        />
                      );
                    })}
                  </Group>
                );
              })}

              {/* ── Geometric Overlays / Rulers ──────────────────────────── */}
              {showRulers && (
                <Group>
                  {/* U1 Axis (Long axis of incisor) - dashed orange */}
                  {u1Line && (
                    <Line
                      points={u1Line}
                      stroke="#f97316"
                      strokeWidth={2 / stageScale}
                      dash={[6, 6]}
                      listening={false}
                    />
                  )}
                  {/* Palatal Plane (PP) - dashed blue */}
                  {ppLine && (
                    <Line
                      points={ppLine}
                      stroke="#3b82f6"
                      strokeWidth={2 / stageScale}
                      dash={[6, 6]}
                      listening={false}
                    />
                  )}
                  {/* Measurement Projections - solid lines */}
                  {visualProjectionLines.map((line, idx) => (
                    <Line
                      key={idx}
                      points={line.points}
                      stroke={line.color}
                      strokeWidth={2.5 / stageScale}
                      listening={false}
                    />
                  ))}
                </Group>
              )}

              {/* ── Keypoints ─────────────────────────────────────────────── */}
              {showLandmarks && keypoints.map((kp) => {
                const [kx, ky] = toContent(kp.x, kp.y);
                const isSel    = selectedId === kp.id;
                return (
                  <Group key={kp.id}>
                    <Circle
                      x={kx} y={ky}
                      radius={pointSize / stageScale}
                      hitStrokeWidth={20 / stageScale}
                      fill={isSel ? KP_DOT_SELECTED : KP_DOT_NORMAL}
                      stroke={isSel ? KP_RING_SELECTED : KP_RING_NORMAL}
                      strokeWidth={1.5 / stageScale}
                      draggable
                      onDragEnd={(e) => moveKp(kp.id, e)}
                      onClick={() => setSelectedId(kp.id)}
                      onMouseEnter={(e) => setCursor(e, 'move')}
                      onMouseLeave={(e) => setCursor(e, 'grab')}
                    />
                    {/* Keypoint label — shadow replaces stroke for legibility */}
                    <Text
                      x={kx + (8 / stageScale)} y={ky - (6 / stageScale)}
                      text={kp.name} fontSize={10 / stageScale} fontStyle="bold"
                      fill="white"
                      shadowColor="black" shadowBlur={4} shadowOpacity={1}
                      shadowOffsetX={1} shadowOffsetY={1}
                      listening={false}
                    />
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
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1.5 w-[95%] md:w-auto max-w-4xl z-50 animate-fade-in pointer-events-none">

              {/* Instructions helper row (hidden on tight screens to save vertical space) */}
              <div className="hidden sm:flex items-center gap-1.5 text-[10px] text-white/50 bg-black/40 px-3 py-0.5 rounded-full backdrop-blur-sm border border-white/5 pointer-events-auto select-none">
                <span><kbd className={chip}>Drag</kbd> move</span>
                <span>·</span>
                <span><kbd className={chip}>Scroll</kbd> zoom</span>
                <span>·</span>
                <span><kbd className={chip}>⇧+Click</kbd> add pt</span>
                <span>·</span>
                <span><kbd className={chip}>DblClick</kbd> del pt</span>
              </div>

              {/* Main functional control bar */}
              <div className="w-full md:w-auto bg-black/85 backdrop-blur-md text-white/90 px-4 py-2 rounded-xl md:rounded-full border border-white/10 flex flex-wrap md:flex-nowrap gap-x-3 gap-y-1.5 text-xs items-center justify-center shadow-2xl pointer-events-auto">

                {/* Visibility toggles */}
                <div className="flex items-center gap-2.5">
                  <label className="flex items-center gap-1 cursor-pointer select-none whitespace-nowrap">
                    <input
                      type="checkbox" checked={showLandmarks}
                      onChange={(e) => setShowLandmarks(e.target.checked)}
                      className="accent-amber-400 w-3 h-3 cursor-pointer"
                    />
                    Landmarks
                  </label>
                  <label className="flex items-center gap-1 cursor-pointer select-none whitespace-nowrap">
                    <input
                      type="checkbox" checked={showPolygons}
                      onChange={(e) => setShowPolygons(e.target.checked)}
                      className="accent-cyan-400 w-3 h-3 cursor-pointer"
                    />
                    Polygons
                  </label>
                  <label className="flex items-center gap-1 cursor-pointer select-none whitespace-nowrap">
                    <input
                      type="checkbox" checked={showRulers}
                      onChange={(e) => setShowRulers(e.target.checked)}
                      className="accent-rose-400 w-3 h-3 cursor-pointer"
                    />
                    Rulers
                  </label>
                </div>

                <span className="text-white/20 select-none">|</span>

                {/* Point size slider */}
                <label className="flex items-center gap-1.5 select-none whitespace-nowrap">
                  <span className="text-white/60 hidden sm:inline">Size</span>
                  <input
                    type="range" min="1" max="10" step="0.5"
                    value={pointSize}
                    onChange={(e) => setPointSize(Number(e.target.value))}
                    className="w-14 sm:w-16 accent-orange-400 cursor-pointer"
                  />
                  <span className="tabular-nums font-mono w-4 text-right text-white/80">{pointSize}</span>
                </label>

                {/* Zoom indicator — fixed width via tabular-nums w-12 to prevent zoom text growth reflows */}
                {stageScale > 1 && (
                  <>
                    <span className="text-white/20 select-none">|</span>
                    <button
                      onClick={resetZoom}
                      className="flex items-center justify-center gap-0.5 text-white/60 hover:text-white transition-colors whitespace-nowrap w-12 text-center tabular-nums font-mono bg-white/5 hover:bg-white/10 px-1.5 py-0.5 rounded text-[11px]"
                      title="Reset Zoom"
                    >
                      {Math.round(stageScale * 100)}%
                    </button>
                  </>
                )}

                {/* Selected element name */}
                {selectedName && (
                  <>
                    <span className="text-white/20 select-none">|</span>
                    <span className="font-mono text-cyan-300 truncate max-w-[80px] sm:max-w-[100px] whitespace-nowrap">
                      ● {selectedName}
                    </span>
                  </>
                )}

                <span className="text-white/20 select-none">|</span>

                {/* Debug mode toggle */}
                <button
                  onClick={() => setIsDebugMode(v => !v)}
                  title={isDebugMode ? 'Hide debug tools' : 'Show debug tools'}
                  className={`px-1.5 py-0.5 rounded transition-colors text-[11px] whitespace-nowrap ${
                    isDebugMode
                      ? 'bg-amber-500/30 text-amber-300 border border-amber-500/40 font-medium'
                      : 'text-white/40 hover:text-white/70'
                  }`}
                >
                  Dev
                </button>

                <span className="text-white/20 select-none">|</span>

                {/* Hide Toolbar Button */}
                <button
                  onClick={() => setIsToolbarOpen(false)}
                  title="Minimize toolbar"
                  className="text-white/40 hover:text-white transition-colors p-1 rounded hover:bg-white/5"
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
}
