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
  points: number[]; // flat [x1,y1,x2,y2,...] in image-space pixels
  fill: string;
  stroke: string;
}

interface Props {
  imageFile: File;
  initialKeypoints?: Keypoint[];
  initialPolygons?: PolygonShape[];
  onKeypointsChange?: (kps: Keypoint[]) => void;
  onPolygonsChange?: (polys: PolygonShape[]) => void;
}

// ── Default landmark positions (normalized 0–1 for a typical lateral ceph) ───

const KP_DEFS: { name: string; fx: number; fy: number }[] = [
  { name: 'Sella (S)',      fx: 0.35, fy: 0.26 },
  { name: 'Nasion (N)',     fx: 0.56, fy: 0.25 },
  { name: 'Orbitale (Or)',  fx: 0.66, fy: 0.34 },
  { name: 'Porion (Po)',    fx: 0.24, fy: 0.34 },
  { name: 'A-point',        fx: 0.56, fy: 0.52 },
  { name: 'B-point',        fx: 0.49, fy: 0.65 },
  { name: 'Pogonion (Pg)',  fx: 0.48, fy: 0.72 },
  { name: 'Menton (Me)',    fx: 0.45, fy: 0.77 },
  { name: 'ANS',            fx: 0.58, fy: 0.52 },
  { name: 'PNS',            fx: 0.35, fy: 0.52 },
];

const POLY_DEFS: { name: string; fracs: number[]; fill: string; stroke: string }[] = [
  {
    name: 'Maxillary Bone',
    fracs: [0.38,0.44, 0.60,0.44, 0.62,0.56, 0.36,0.56],
    fill: 'rgba(255,100,100,0.18)',
    stroke: '#ff6464',
  },
  {
    name: 'Mandibular Bone',
    fracs: [0.30,0.60, 0.55,0.60, 0.52,0.78, 0.38,0.80, 0.28,0.70],
    fill: 'rgba(100,220,100,0.15)',
    stroke: '#64dc64',
  },
  {
    name: 'Cranial Base',
    fracs: [0.18,0.15, 0.58,0.15, 0.58,0.36, 0.28,0.36],
    fill: 'rgba(100,160,255,0.15)',
    stroke: '#64a0ff',
  },
];

function fracsToImageCoords(fracs: number[], w: number, h: number): number[] {
  return fracs.map((v, i) => (i % 2 === 0 ? v * w : v * h));
}

// ── Geometry: nearest edge insertion ─────────────────────────────────────────

function nearestEdgeInsert(
  points: number[],
  cx: number,
  cy: number
): { insertAt: number; px: number; py: number } {
  const n = points.length / 2;
  let best = { insertAt: 2, px: 0, py: 0, dist: Infinity };

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ax = points[i * 2],     ay = points[i * 2 + 1];
    const bx = points[j * 2],     by = points[j * 2 + 1];
    const dx = bx - ax,           dy = by - ay;
    const lenSq = dx * dx + dy * dy;
    const t = lenSq === 0
      ? 0
      : Math.max(0, Math.min(1, ((cx - ax) * dx + (cy - ay) * dy) / lenSq));
    const projX = ax + t * dx,    projY = ay + t * dy;
    const dist = Math.hypot(cx - projX, cy - projY);
    if (dist < best.dist) {
      best = { insertAt: j * 2, px: projX, py: projY, dist };
    }
  }
  return best;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function CephCanvasEditor({
  imageFile,
  initialKeypoints,
  initialPolygons,
  onKeypointsChange,
  onPolygonsChange,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [stageW, setStageW]     = useState(0);
  const [stageH, setStageH]     = useState(0);
  const [img, setImg]           = useState<HTMLImageElement | null>(null);
  const [keypoints, setKeypoints] = useState<Keypoint[]>([]);
  const [polygons, setPolygons]   = useState<PolygonShape[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Load image from File object
  useEffect(() => {
    const url = URL.createObjectURL(imageFile);
    const el  = new window.Image();
    el.onload = () => setImg(el);
    el.src    = url;
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  // Observe container for resize
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setStageW(el.offsetWidth);
      setStageH(el.offsetHeight);
    });
    ro.observe(el);
    setStageW(el.offsetWidth);
    setStageH(el.offsetHeight);
    return () => ro.disconnect();
  }, []);

  // Initialise shapes when image first loads
  useEffect(() => {
    if (!img) return;
    const { width: w, height: h } = img;

    setKeypoints(
      initialKeypoints ??
        KP_DEFS.map((def, i) => ({
          id:   `kp-${i}`,
          name: def.name,
          x:    def.fx * w,
          y:    def.fy * h,
        }))
    );

    setPolygons(
      initialPolygons ??
        POLY_DEFS.map((def, i) => ({
          id:     `poly-${i}`,
          name:   def.name,
          points: fracsToImageCoords(def.fracs, w, h),
          fill:   def.fill,
          stroke: def.stroke,
        }))
    );
  }, [img]); // eslint-disable-line react-hooks/exhaustive-deps

  // Compute fit-to-stage transform
  const { offX, offY, scale } = useMemo(() => {
    if (!img || stageW === 0 || stageH === 0) return { offX: 0, offY: 0, scale: 1 };
    const s = Math.min(stageW / img.width, stageH / img.height);
    return {
      scale: s,
      offX:  (stageW - img.width  * s) / 2,
      offY:  (stageH - img.height * s) / 2,
    };
  }, [img, stageW, stageH]);

  // Coordinate converters
  const toStage = useCallback(
    (ix: number, iy: number): [number, number] => [ix * scale + offX, iy * scale + offY],
    [scale, offX, offY]
  );
  const toImage = useCallback(
    (sx: number, sy: number): [number, number] => [(sx - offX) / scale, (sy - offY) / scale],
    [scale, offX, offY]
  );

  // Propagate changes upward
  useEffect(() => { onKeypointsChange?.(keypoints); }, [keypoints]);  // eslint-disable-line
  useEffect(() => { onPolygonsChange?.(polygons); },   [polygons]);   // eslint-disable-line

  // ── Keypoint drag ────────────────────────────────────────────────────────────
  const moveKp = useCallback((id: string, e: KonvaEventObject<DragEvent>) => {
    const [ix, iy] = toImage(e.target.x(), e.target.y());
    setKeypoints(prev => prev.map(k => (k.id === id ? { ...k, x: ix, y: iy } : k)));
  }, [toImage]);

  // ── Polygon vertex drag (live update of the Line) ────────────────────────────
  const moveVertex = useCallback(
    (polyId: string, vi: number, e: KonvaEventObject<DragEvent>) => {
      const [ix, iy] = toImage(e.target.x(), e.target.y());
      setPolygons(prev =>
        prev.map(p => {
          if (p.id !== polyId) return p;
          const pts = [...p.points];
          pts[vi * 2]     = ix;
          pts[vi * 2 + 1] = iy;
          return { ...p, points: pts };
        })
      );
    },
    [toImage]
  );

  // ── Delete vertex (Dbl-Click OR Alt+Click) ───────────────────────────────────
  const deleteVertex = useCallback(
    (polyId: string, vi: number, e: KonvaEventObject<MouseEvent>) => {
      e.cancelBubble = true;
      setPolygons(prev =>
        prev.map(p => {
          if (p.id !== polyId || p.points.length <= 6) return p; // min 3 vertices
          const pts = [...p.points];
          pts.splice(vi * 2, 2);
          return { ...p, points: pts };
        })
      );
    },
    []
  );

  // ── Add vertex at nearest edge (Shift+Click on Line) ────────────────────────
  const addEdgePoint = useCallback(
    (polyId: string, e: KonvaEventObject<MouseEvent>) => {
      if (!e.evt.shiftKey) return;
      e.cancelBubble = true;
      const pos = e.target.getStage()!.getPointerPosition()!;
      const [ix, iy] = toImage(pos.x, pos.y);
      setPolygons(prev =>
        prev.map(p => {
          if (p.id !== polyId) return p;
          const { insertAt, px, py } = nearestEdgeInsert(p.points, ix, iy);
          const pts = [...p.points];
          pts.splice(insertAt, 0, px, py);
          return { ...p, points: pts };
        })
      );
    },
    [toImage]
  );

  // Cursor helpers
  const setCursor = useCallback((e: KonvaEventObject<MouseEvent>, cur: string) => {
    const stage = e.target.getStage();
    if (stage) stage.container().style.cursor = cur;
  }, []);

  const selectedName = useMemo(
    () => [...keypoints, ...polygons].find(s => s.id === selectedId)?.name,
    [selectedId, keypoints, polygons]
  );

  return (
    <div className="flex flex-col w-full h-full">
      {/* ── Toolbar ─────────────────────────────────────────────────────────── */}
      <div className="shrink-0 flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-1.5 bg-slate-900/95 text-[11px] text-slate-400 rounded-t-2xl select-none border-b border-slate-700/50">
        <span className="font-bold text-orange-400 mr-1">Ceph Editor</span>
        <span className="hidden sm:inline text-slate-600">|</span>
        <span><kbd className="bg-slate-700/80 text-slate-200 px-1.5 py-0.5 rounded text-[10px] font-mono">Drag</kbd> vertex to move</span>
        <span><kbd className="bg-slate-700/80 text-slate-200 px-1.5 py-0.5 rounded text-[10px] font-mono">Shift+Click</kbd> edge → add point</span>
        <span><kbd className="bg-slate-700/80 text-slate-200 px-1.5 py-0.5 rounded text-[10px] font-mono">Dbl-Click</kbd> / <kbd className="bg-slate-700/80 text-slate-200 px-1.5 py-0.5 rounded text-[10px] font-mono">Alt+Click</kbd> vertex → delete</span>
        {selectedName && (
          <span className="ml-auto text-orange-300 font-mono truncate max-w-[160px]" title={selectedName}>
            ● {selectedName}
          </span>
        )}
      </div>

      {/* ── Canvas ─────────────────────────────────────────────────────────── */}
      <div ref={containerRef} className="flex-1 min-h-0 bg-slate-950 rounded-b-2xl overflow-hidden">
        {img && stageW > 0 && stageH > 0 && (
          <Stage
            width={stageW}
            height={stageH}
            onClick={(e) => {
              if (e.target === e.target.getStage()) setSelectedId(null);
            }}
          >
            <Layer>
              {/* Background X-ray image */}
              <KonvaImage
                image={img}
                x={offX}
                y={offY}
                width={img.width  * scale}
                height={img.height * scale}
                listening={false}
              />

              {/* ── Polygons ──────────────────────────────────────────────── */}
              {polygons.map((poly) => {
                const stagePts: number[] = [];
                for (let i = 0; i < poly.points.length; i += 2) {
                  const [sx, sy] = toStage(poly.points[i], poly.points[i + 1]);
                  stagePts.push(sx, sy);
                }
                const isSel       = selectedId === poly.id;
                const vertexCount = poly.points.length / 2;

                return (
                  <Group key={poly.id}>
                    {/* Filled polygon outline — Shift+Click to add vertex */}
                    <Line
                      points={stagePts}
                      closed
                      fill={poly.fill}
                      stroke={isSel ? '#ffffff' : poly.stroke}
                      strokeWidth={isSel ? 2 : 1.5}
                      hitStrokeWidth={10}
                      onClick={(e) => {
                        setSelectedId(poly.id);
                        addEdgePoint(poly.id, e);
                      }}
                      onMouseEnter={(e) => setCursor(e, 'pointer')}
                      onMouseLeave={(e) => setCursor(e, 'default')}
                    />

                    {/* Polygon name label */}
                    <Text
                      x={stagePts[0] ?? 0}
                      y={(stagePts[1] ?? 0) - 16}
                      text={poly.name}
                      fontSize={11}
                      fontStyle="bold"
                      fill={poly.stroke}
                      shadowColor="black"
                      shadowBlur={4}
                      shadowOpacity={1}
                      listening={false}
                    />

                    {/* Vertex control points */}
                    {Array.from({ length: vertexCount }, (_, vi) => {
                      const [vx, vy] = toStage(
                        poly.points[vi * 2],
                        poly.points[vi * 2 + 1]
                      );
                      return (
                        <Circle
                          key={vi}
                          x={vx}
                          y={vy}
                          radius={isSel ? 6 : 5}
                          fill={isSel ? '#ffffff' : poly.stroke}
                          stroke="rgba(0,0,0,0.6)"
                          strokeWidth={1}
                          draggable
                          // Live-update polygon as vertex is dragged
                          onDragMove={(e) => moveVertex(poly.id, vi, e)}
                          // Delete vertex on double-click
                          onDblClick={(e) => deleteVertex(poly.id, vi, e)}
                          // Delete vertex on Alt+Click; select polygon otherwise
                          onClick={(e) => {
                            setSelectedId(poly.id);
                            if (e.evt.altKey) deleteVertex(poly.id, vi, e);
                            e.cancelBubble = true;
                          }}
                          onMouseEnter={(e) => setCursor(e, 'crosshair')}
                          onMouseLeave={(e) => setCursor(e, 'default')}
                        />
                      );
                    })}
                  </Group>
                );
              })}

              {/* ── Keypoints ─────────────────────────────────────────────── */}
              {keypoints.map((kp) => {
                const [kx, ky] = toStage(kp.x, kp.y);
                const isSel    = selectedId === kp.id;
                const color    = isSel ? '#ffe500' : '#ff9900';

                return (
                  <Group key={kp.id}>
                    {/* Outer ring (visual only) */}
                    <Circle
                      x={kx} y={ky}
                      radius={9}
                      fill="transparent"
                      stroke={color}
                      strokeWidth={1.5}
                      listening={false}
                    />
                    {/* Inner dot */}
                    <Circle
                      x={kx} y={ky}
                      radius={3}
                      fill={color}
                      listening={false}
                    />
                    {/* Invisible draggable hitbox covering the full landmark */}
                    <Circle
                      x={kx} y={ky}
                      radius={10}
                      fill="transparent"
                      stroke="transparent"
                      draggable
                      onDragEnd={(e) => moveKp(kp.id, e)}
                      onClick={() => setSelectedId(kp.id)}
                      onMouseEnter={(e) => setCursor(e, 'move')}
                      onMouseLeave={(e) => setCursor(e, 'default')}
                    />
                    {/* Label */}
                    <Text
                      x={kx + 12}
                      y={ky - 6}
                      text={kp.name}
                      fontSize={10}
                      fill="#ffc060"
                      shadowColor="black"
                      shadowBlur={3}
                      shadowOpacity={0.9}
                      listening={false}
                    />
                  </Group>
                );
              })}
            </Layer>
          </Stage>
        )}

        {/* Loading state before image decodes */}
        {!img && (
          <div className="flex items-center justify-center h-full text-slate-500 text-sm">
            Loading image…
          </div>
        )}
      </div>
    </div>
  );
}
