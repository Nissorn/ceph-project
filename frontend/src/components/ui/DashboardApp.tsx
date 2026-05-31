import React, { useState, useEffect, useCallback, useRef } from 'react';
import { jsPDF } from 'jspdf';
import UploadZone from './UploadZone';
import MetricCard from './MetricCard';
import CephCanvasEditor, { type Lines3Level, type GlobalMinLines } from './CephCanvasEditor';

const DistanceItem = ({ label, value, severity }: { label: string; value: number; severity?: string }) => {
  let statusColor = 'text-emerald-500 dark:text-emerald-400';
  let badgeColor = 'bg-emerald-500/10 border-emerald-500/20 dark:bg-emerald-950/20';
  let warningIcon = null;
  let desc = 'Safe';

  if (severity === 'Critical' || value < 0.5) {
    statusColor = 'text-rose-500 dark:text-rose-400';
    badgeColor = 'bg-rose-500/10 border-rose-500/20 dark:bg-rose-950/20 animate-pulse';
    warningIcon = <span className="text-xs shrink-0" title="Critical Warning: Thin bone">⚠️</span>;
    desc = 'Thin (< 0.5mm)';
  } else if (value <= 1.0) {
    statusColor = 'text-amber-500 dark:text-amber-400';
    badgeColor = 'bg-amber-500/10 border-amber-500/20 dark:bg-amber-950/20';
    desc = 'Monitor';
  } else {
    desc = 'Thick (> 1.0mm)';
  }

  return (
    <div className={`p-2.5 rounded-lg border ${badgeColor} flex justify-between items-center transition-all duration-200 hover:scale-[1.02] shadow-sm`}>
      <div className="flex flex-col">
        <span className="text-[10px] font-bold uppercase text-slate-400 dark:text-slate-500 tracking-wider">
          {label}
        </span>
        <span className="text-[9px] text-slate-400 dark:text-slate-500 mt-0.5">
          {desc}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        {warningIcon}
        <span className={`text-base font-bold tracking-tight ${statusColor}`}>
          {value.toFixed(2)}
        </span>
        <span className="text-[10px] text-slate-400 dark:text-slate-500 font-normal">
          mm
        </span>
      </div>
    </div>
  );
};

const getShortBoneTypeLabel = (type: string) => {
  if (type.includes('Type 1')) return 'Thick Bone';
  if (type.includes('Type 2')) return 'Mono-Plate';
  if (type.includes('Type 3')) return 'Double-Plate';
  if (type.includes('Type 4')) return 'Vulnerably Thin';
  return 'Unknown';
};

/**
 * Adapts the backend's flat measurement_lines dict to the Lines3Level schema
 * the CephCanvasEditor canvas expects for the Plan B 3-level dashed rulers.
 *
 * Backend schema (6 keys, each [[x1,y1],[x2,y2]]):
 *   labial_crest_line, labial_midroot_line, labial_apex_line,
 *   palatal_crest_line, palatal_midroot_line, palatal_apex_line
 *
 * Canvas schema (Lines3Level):
 *   cervical, middle, apical  →  each a Segment6 with palatal + labial endpoints
 *
 * Mapping: cervical=crest, middle=midroot, apical=apex
 */
function adaptMeasurementLinesToBoneThickness(ml: any, metrics?: any): Lines3Level | undefined {
  if (!ml) return undefined;
  try {
    const pick = (key: string) => {
      const seg: number[][] = ml[key];
      if (!Array.isArray(seg) || seg.length < 2) return null;
      return { x1: seg[0][0], y1: seg[0][1], x2: seg[1][0], y2: seg[1][1] };
    };

    const lCrest = pick('labial_crest_line');
    const lMid = pick('labial_midroot_line');
    const lApex = pick('labial_apex_line');
    const pCrest = pick('palatal_crest_line');
    const pMid = pick('palatal_midroot_line');
    const pApex = pick('palatal_apex_line');

    if (!lCrest || !lMid || !lApex || !pCrest || !pMid || !pApex) return undefined;

    // Pull mm distances from the metrics block for accurate ruler labels.
    const lc = Number(metrics?.labial_crest_mm ?? 0);
    const lm = Number(metrics?.labial_midroot_mm ?? 0);
    const la = Number(metrics?.labial_apex_mm ?? 0);
    const pc = Number(metrics?.palatal_crest_mm ?? 0);
    const pm = Number(metrics?.palatal_midroot_mm ?? 0);
    const pa = Number(metrics?.palatal_apex_mm ?? 0);

    const makeSegment6 = (
      l: { x1: number; y1: number; x2: number; y2: number },
      p: { x1: number; y1: number; x2: number; y2: number },
      lDistMm: number,
      pDistMm: number,
    ) => ({
      // Labial: landmark → tooth surface  (line start → end)
      labial_distance_mm: lDistMm,
      labial_tooth_x: l.x1, labial_tooth_y: l.y1,
      labial_bone_x: l.x2, labial_bone_y: l.y2,
      // Palatal: landmark → tooth surface
      palatal_distance_mm: pDistMm,
      palatal_tooth_x: p.x1, palatal_tooth_y: p.y1,
      palatal_bone_x: p.x2, palatal_bone_y: p.y2,
    });

    return {
      cervical: makeSegment6(lCrest, pCrest, lc, pc),
      middle: makeSegment6(lMid, pMid, lm, pm),
      apical: makeSegment6(lApex, pApex, la, pa),
    };
  } catch {
    console.warn('[DashboardApp] adaptMeasurementLinesToBoneThickness: parse error', ml);
    return undefined;
  }
}

// NOTE: adaptZonalMeasurementLines removed — replaced by GlobalMinLines direct pass-through.
// The backend now pre-computes mm values (labial_mm, palatal_mm) and the frontend
// passes global_min_lines directly to the CephCanvasEditor globalMinLines prop.
// No client-side pixel→mm math needed; no hardcoded calibration possible.

export default function DashboardApp() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  // Measurement mode: 'standard' = existing 3-line method | 'zonal' = Zonal Min Distance
  const [measurementMode, setMeasurementMode] = useState<'standard' | 'zonal'>('standard');
  const [cervicalOffsetMm, setCervicalOffsetMm] = useState<number>(1.5);

  const [patientName, setPatientName] = useState('');
  const [patientId, setPatientId] = useState('');
  const [reportDate, setReportDate] = useState(() => new Date().toISOString().split('T')[0]);
  const [originalAnalysis, setOriginalAnalysis] = useState<any>(null);
  const cephCanvasRef = useRef<any>(null);

  // Optimized Object URL lifecycle management to prevent browser memory leaks
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    // Free browser memory when file updates or component unmounts
    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  const handleFileSelect = useCallback((selectedFile: File) => {
    setFile(selectedFile);
    setResults(null);
    setError(null);
    setOriginalAnalysis(null);
    setMeasurementMode('standard');  // reset mode on new image
    setCervicalOffsetMm(1.5);        // reset offset
  }, []);

  const handleReset = useCallback(() => {
    setFile(null);
    setResults(null);
    setError(null);
    setOriginalAnalysis(null);
    setMeasurementMode('standard');  // reset mode on clear
    setCervicalOffsetMm(1.5);        // reset offset
  }, []);

  const processAndSetResults = useCallback((payload: any) => {
    // Clinical label order: Upper_tip, Upper_apex, Labial_midroot, Labial_crest,
    // Palatal_midroot, Palatal_crest, ANS, PNS, LB, PB
    const KP_NAMES = [
      'Upper_tip', 'Upper_apex', 'Labial_midroot', 'Labial_crest',
      'Palatal_midroot', 'Palatal_crest', 'ANS', 'PNS', 'LB', 'PB',
    ];
    // Backend returns segmentation as keyed dict; matches CephCanvasEditor POLY_PALETTE order
    const SEG_NAMES = ['Upper_incisor', 'Labial_bone', 'Palatal_bone'] as const;
    const SEG_PALETTE = [
      { fill: 'rgba(6, 182, 212, 0.15)', stroke: 'rgba(6, 182, 212, 0.9)' },
      { fill: 'rgba(236, 72, 153, 0.15)', stroke: 'rgba(236, 72, 153, 0.9)' },
      { fill: 'rgba(16, 185, 129, 0.15)', stroke: 'rgba(16, 185, 129, 0.9)' },
    ];

    const apiKeypoints = Array.isArray(payload.landmarks)
      ? payload.landmarks.map((kp: any, i: number) => ({
        id: `kp-${i}`,
        name: KP_NAMES[i] ?? kp?.name ?? `kp-${i}`,
        x: Number(kp?.x ?? 0),
        y: Number(kp?.y ?? 0),
      }))
      : undefined;

    // Backend polygon: [[x,y],...] → CephCanvasEditor needs flat [x,y,x,y,...]
    const apiPolygons = payload.segmentation
      ? SEG_NAMES.map((name, i) => {
        const seg = payload.segmentation[name];
        const flatPoints: number[] = Array.isArray(seg?.polygon)
          ? (seg.polygon as number[][]).flatMap((pt: number[]) => [pt[0], pt[1]])
          : [];
        return {
          id: `poly-${i}`,
          name,
          points: flatPoints,
          fill: SEG_PALETTE[i].fill,
          stroke: SEG_PALETTE[i].stroke,
        };
      })
      : undefined;

    // Safely parse and cleanly format numeric metrics to prevent excessive float layouts
    const rawAngle = payload.metrics?.u1_pp_angle_deg ?? 112.5;
    const u1_pp_angle = typeof rawAngle === 'number' && !isNaN(rawAngle) ? Number(rawAngle.toFixed(1)) : 112.5;

    const u1_pp_status = u1_pp_angle > 115 ? 'warning' : u1_pp_angle < 105 ? 'warning' : 'normal';

    const clinical_assessments = payload.clinical_assessment || {};

    // Extract new 6 distances in mm and their severity levels
    const labial_crest = Number((payload.metrics?.labial_crest_mm ?? 1.2).toFixed(2));
    const labial_crest_severity = payload.metrics?.labial_crest_severity ?? 'Monitor';
    const labial_midroot = Number((payload.metrics?.labial_midroot_mm ?? 1.5).toFixed(2));
    const labial_midroot_severity = payload.metrics?.labial_midroot_severity ?? 'Monitor';
    const labial_apex = Number((payload.metrics?.labial_apex_mm ?? 1.0).toFixed(2));
    const labial_apex_severity = payload.metrics?.labial_apex_severity ?? 'Monitor';
    const palatal_crest = Number((payload.metrics?.palatal_crest_mm ?? 1.4).toFixed(2));
    const palatal_crest_severity = payload.metrics?.palatal_crest_severity ?? 'Monitor';
    const palatal_midroot = Number((payload.metrics?.palatal_midroot_mm ?? 1.6).toFixed(2));
    const palatal_midroot_severity = payload.metrics?.palatal_midroot_severity ?? 'Monitor';
    const palatal_apex = Number((payload.metrics?.palatal_apex_mm ?? 1.1).toFixed(2));
    const palatal_apex_severity = payload.metrics?.palatal_apex_severity ?? 'Monitor';

    // CRITICAL: extract mm_per_pixel from _debug for zonal distance computation.
    // Per design rule: NEVER hardcode or guess the calibration value.
    // The backend now exposes it in payload._debug.mm_per_pixel.
    const mmPerPixel = Number(payload._debug?.mm_per_pixel ?? 0);
    if (!mmPerPixel || mmPerPixel <= 0) {
      console.warn('[DashboardApp] mm_per_pixel missing or zero in _debug — zonal mm labels will fallback to 0');
    }

    const normalizedResults = {
      u1_pp_angle,
      u1_pp_status,
      labial_crest,
      labial_crest_severity,
      labial_midroot,
      labial_midroot_severity,
      labial_apex,
      labial_apex_severity,
      palatal_crest,
      palatal_crest_severity,
      palatal_midroot,
      palatal_midroot_severity,
      palatal_apex,
      palatal_apex_severity,
      clinical_assessments,
      metrics: payload.metrics || null,
      measurement_lines: payload.measurement_lines || null,
      global_min_lines: (payload.global_min_lines as GlobalMinLines) || null,  // 2-line global min sweep
      mm_per_pixel: mmPerPixel,                                            // NEW: dynamic calibration
      annotations: { keypoints: apiKeypoints, polygons: apiPolygons },
    };

    setResults(normalizedResults);
    setMeasurementMode('standard');  // always start on standard after a fresh analysis
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);

    // Deterministic network timeout protection via AbortController
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 120 seconds max boundary

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8123/api/v1/analyze', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        // Attempt to extract downstream structured JSON error safely
        let serverErrorMsg = '';
        try {
          const errorJson = await response.json();
          serverErrorMsg = errorJson?.detail || errorJson?.message || errorJson?.error || '';
        } catch (_) {
          // Non-JSON failure response (e.g. 502/504 gateway timeouts)
        }
        throw new Error(serverErrorMsg ? `Server Error: ${serverErrorMsg}` : `HTTP Request Failed (${response.status})`);
      }

      let data;
      try {
        data = await response.json();
      } catch (jsonErr) {
        throw new Error('Received malformed response payload from the analysis server.');
      }

      console.log("REAL API PAYLOAD RECEIVED:", data);
      const payload = data?.data || data || {};
      console.log("UNWRAPPED PAYLOAD:", payload);

      setOriginalAnalysis(JSON.parse(JSON.stringify(payload)));
      processAndSetResults(payload);
    } catch (err: any) {
      console.error("Analysis failed:", err);
      if (err.name === 'AbortError') {
        setError('Analysis request timed out after 120 seconds. Verify inference engine availability.');
      } else {
        setError(err.message || 'Failed to connect to the backend analysis service.');
      }
    } finally {
      clearTimeout(timeoutId);
      setIsLoading(false);
    }
  }, [file]);

  // Dynamically pull active assessment based on current slider/mode
  const activeAssessment = results ? (
    measurementMode === 'standard'
      ? results.clinical_assessments?.['standard']
      : results.clinical_assessments?.[cervicalOffsetMm.toFixed(1)]
  ) : null;

  const exportPDFReport = useCallback(async () => {
    if (!cephCanvasRef.current || !results) return;
    
    try {
      const imageData = cephCanvasRef.current.getCanvasImage();
      if (!imageData) throw new Error("Could not capture canvas image");
      
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });
      
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      
      // Document Settings
      pdf.setFont("helvetica");
      let currentY = 20;
      const margin = 20;
      const contentWidth = pageWidth - (margin * 2);

      // Logo/Header
      pdf.setFontSize(20);
      pdf.setFont("helvetica", "bold");
      pdf.setTextColor(30, 58, 138); // Indigo-900
      pdf.text("Advanced Cephalometric Analysis Report", margin, currentY);
      
      currentY += 15;

      // Patient Metadata Section
      pdf.setDrawColor(226, 232, 240); // Slate-200
      pdf.line(margin, currentY, pageWidth - margin, currentY);
      currentY += 8;
      
      pdf.setFontSize(11);
      pdf.setTextColor(15, 23, 42); // Slate-900
      pdf.setFont("helvetica", "bold");
      pdf.text(`Patient Name:`, margin, currentY);
      pdf.setFont("helvetica", "normal");
      pdf.text(`${patientName || 'Anonymous'}`, margin + 35, currentY);
      
      pdf.setFont("helvetica", "bold");
      pdf.text(`Patient ID:`, margin + 100, currentY);
      pdf.setFont("helvetica", "normal");
      pdf.text(`${patientId || 'N/A'}`, margin + 125, currentY);
      
      currentY += 8;
      pdf.setFont("helvetica", "bold");
      pdf.text(`Date:`, margin, currentY);
      pdf.setFont("helvetica", "normal");
      pdf.text(`${reportDate}`, margin + 35, currentY);
      
      currentY += 10;
      pdf.line(margin, currentY, pageWidth - margin, currentY);
      currentY += 15;

      // X-Ray Image
      const imgProps = pdf.getImageProperties(imageData);
      const imgRatio = imgProps.height / imgProps.width;
      
      const renderImgWidth = contentWidth;
      const renderImgHeight = contentWidth * imgRatio;
      
      pdf.addImage(imageData, 'JPEG', margin, currentY, renderImgWidth, renderImgHeight);
      
      currentY += renderImgHeight + 15;
      
      if (currentY > pageHeight - 50) {
        pdf.addPage();
        currentY = 20;
      }

      // Clinical Results Section
      pdf.setFontSize(14);
      pdf.setFont("helvetica", "bold");
      pdf.text("Clinical Assessment", margin, currentY);
      currentY += 10;
      
      pdf.setFontSize(11);
      pdf.setFont("helvetica", "bold");
      pdf.text("U1-PP Angle:", margin, currentY);
      pdf.setFont("helvetica", "normal");
      pdf.text(`${results.u1_pp_angle}° (${activeAssessment?.u1_pp_angle_class ?? 'Normal Inclination'})`, margin + 35, currentY);
      
      currentY += 8;
      pdf.setFont("helvetica", "bold");
      pdf.text("Phenotype:", margin, currentY);
      pdf.setFont("helvetica", "normal");
      pdf.text(`${activeAssessment?.bone_thickness_type ?? 'Type 1 - Thick'}`, margin + 35, currentY);
      
      currentY += 30;
      
      // Signature Line
      if (currentY > pageHeight - 20) {
          pdf.addPage();
          currentY = 20;
      }
      
      pdf.setDrawColor(15, 23, 42); // Slate-900
      pdf.line(pageWidth - margin - 60, currentY, pageWidth - margin, currentY);
      pdf.setFontSize(10);
      pdf.text("(Dr. Signature)", pageWidth - margin - 45, currentY + 5);

      pdf.save(`Ceph_Report_${patientId || 'Anonymous'}.pdf`);
    } catch (err) {
      console.error("PDF Generation failed:", err);
      alert("Failed to generate PDF. Check console for details.");
    }
  }, [patientName, patientId, reportDate, results, activeAssessment]);

  return (
    <div className="flex-1 flex overflow-hidden relative">
      {/* Left Panel - Image Viewer */}
      <div className="flex-[2] relative overflow-hidden flex flex-col p-4">
        <div className="bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700/60 shadow-sm rounded-2xl p-3 flex-grow overflow-hidden flex flex-col relative">

          {file ? (
            <div className="flex-1 min-h-0 relative flex items-center justify-center bg-slate-900 rounded-lg overflow-hidden group border border-slate-800/60">
              {/* Filename badge */}
              <div className="absolute top-4 left-4 z-20 bg-black/50 text-slate-100 px-4 py-1.5 rounded-full text-xs font-medium tracking-wide backdrop-blur-md pointer-events-none border border-white/10">
                {file.name}
              </div>

              {/* Remove button */}
              <button
                onClick={handleReset}
                className="absolute top-3 right-3 z-20 bg-black/40 hover:bg-red-500/80 text-white p-2 rounded-full transition duration-150 focus:outline-none focus:ring-2 focus:ring-white/50"
                title="Remove Image"
                aria-label="Remove image"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
              </button>

              {/* After analysis: interactive Ceph Editor; before: plain preview */}
              {results && !isLoading ? (
                <div className="absolute inset-0 z-10">
                  <CephCanvasEditor
                    ref={cephCanvasRef}
                    imageFile={file}
                    originalAnalysis={originalAnalysis}
                    initialKeypoints={results.annotations?.keypoints}
                    initialPolygons={results.annotations?.polygons}
                    boneThickness={
                      measurementMode === 'standard'
                        ? adaptMeasurementLinesToBoneThickness(results.measurement_lines, results.metrics)
                        : undefined  // suppress 3-line rulers in Min Distance mode
                    }
                    globalMinLines={
                      measurementMode === 'zonal' && results?.global_min_lines
                        ? results.global_min_lines[cervicalOffsetMm.toFixed(1)]
                        : undefined
                    }
                    onRecalculate={processAndSetResults}
                  />
                </div>
              ) : previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Cephalogram preview"
                  className="w-full h-full object-contain opacity-80"
                />
              ) : null}

              {isLoading && (
                <div className="absolute inset-0 bg-slate-900/60 flex items-center justify-center z-30 rounded-lg backdrop-blur-sm transition-all duration-200">
                  <div className="flex flex-col items-center gap-4">
                    <svg className="animate-spin h-10 w-10 text-singapodent-accent" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span className="text-white/90 text-xs font-semibold tracking-wider uppercase">Analyzing scan architecture...</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <UploadZone onFileSelect={handleFileSelect} />
          )}

        </div>

        {/* Analyze Errors (Button is now Floating Action Button) */}
        {error && (
          <div className="shrink-0 mt-4 text-red-500 font-medium text-sm flex items-center gap-2">
            <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{error}</span>
          </div>
        )}

      </div>

      {/* Right Panel - Metrics */}
      <div className="w-full md:w-[450px] overflow-y-auto p-4 pb-28 border-l border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 shrink-0 custom-scrollbar">
        
        {/* PATIENT METADATA BANNER */}
        <div className="bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/60 rounded-xl p-4 mb-6 shadow-sm">
          <h3 className="text-xs font-bold uppercase text-slate-500 dark:text-slate-400 tracking-wider mb-3 flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
            Patient Metadata
          </h3>
          <div className="flex flex-col gap-3">
            <div className="flex gap-3">
              <div className="flex-1">
                <label className="block text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-1">Patient Name</label>
                <input type="text" value={patientName} onChange={e => setPatientName(e.target.value)} placeholder="e.g. John Doe" className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-indigo-500 transition-colors" />
              </div>
              <div className="w-1/3">
                <label className="block text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-1">Patient ID</label>
                <input type="text" value={patientId} onChange={e => setPatientId(e.target.value)} placeholder="ID-12345" className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-indigo-500 transition-colors" />
              </div>
            </div>
            <div>
              <label className="block text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-1">Date</label>
              <input type="date" value={reportDate} onChange={e => setReportDate(e.target.value)} className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-indigo-500 transition-colors" />
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-6 pt-2">
          <div className="pl-2 flex items-center justify-between">
            <h2 className="text-xl font-light tracking-tight text-slate-800 dark:text-white">Clinical Assessment</h2>
            <span className="text-xs font-medium text-slate-400 dark:text-slate-500 uppercase tracking-wider">Results</span>
          </div>

          {!results && !isLoading && !error && (
            <div className="h-[280px] flex items-center justify-center border border-dashed border-slate-200 dark:border-slate-700/60 rounded-xl px-8 text-center text-slate-400 dark:text-slate-500">
              <div className="flex flex-col items-center gap-3">
                <svg className="w-8 h-8 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                <span className="text-sm font-medium tracking-wide">Awaiting image upload to calculate biomechanical mappings.</span>
              </div>
            </div>
          )}

          {error && !isLoading && (
            <div className="h-64 flex items-center justify-center border border-red-200 dark:border-red-800/50 bg-red-50/80 dark:bg-red-900/10 rounded-xl px-8 text-center text-red-500">
              <div className="flex flex-col items-center gap-2">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span className="text-sm font-medium">Inference connection failed. Please ensure backend services are active.</span>
              </div>
            </div>
          )}

          {isLoading && (
            <div className="flex flex-col gap-5 mt-2">
              {[1, 2, 3].map(i => (
                <div key={i} className="h-[110px] bg-slate-100 dark:bg-slate-800/60 animate-pulse rounded-xl border border-slate-200/50 dark:border-slate-700/50 relative overflow-hidden">
                  <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/40 dark:via-white/5 to-transparent animate-[shimmer_2s_infinite]"></div>
                  <div className="mt-8 ml-6 w-24 h-4 bg-slate-300 dark:bg-slate-600 rounded-full"></div>
                  <div className="mt-4 ml-6 w-40 h-8 bg-slate-300 dark:bg-slate-600 rounded-full"></div>
                </div>
              ))}
            </div>
          )}

          {results && !isLoading && (
            <div className="flex flex-col gap-5 animate-fade-in">
              <MetricCard
                title="U1-PP Angle"
                value={results.u1_pp_angle}
                subtitle={`° (${activeAssessment?.u1_pp_angle_class ?? 'Normal Inclination'})`}
                status={results.u1_pp_status}
              />
              <MetricCard
                title="Alveolar Bone Phenotype"
                value={activeAssessment?.bone_thickness_type ?? 'Type 1 - Thick'}
                subtitle={getShortBoneTypeLabel(activeAssessment?.bone_thickness_type ?? '')}
                status={(activeAssessment?.bone_thickness_type ?? '').includes('Type 1') ? 'normal' : (activeAssessment?.bone_thickness_type ?? '').includes('Type 4') ? 'critical' : 'warning'}
              />

              {/* Distance Matrix Card */}
              <div className="bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700/60 rounded-xl p-5 flex flex-col gap-4 relative overflow-hidden transition-all duration-300">

                {/* ── Measurement Mode Toggle (Segmented Control) ─────────────────── */}
                <div
                  id="measurement-mode-toggle"
                  className="flex rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700/60 bg-slate-100 dark:bg-slate-900/40 p-0.5 gap-0.5"
                  role="group"
                  aria-label="Measurement mode"
                >
                  <button
                    id="toggle-standard"
                    onClick={() => setMeasurementMode('standard')}
                    aria-selected={measurementMode === 'standard'}
                    role="tab"
                    className={`flex-1 py-1.5 px-2 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-singapodent-accent focus:ring-offset-1 ${measurementMode === 'standard'
                        ? 'bg-white dark:bg-slate-700 text-slate-800 dark:text-white shadow-sm'
                        : 'text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'
                      }`}
                    title="Standard 3-line measurement method"
                  >
                    Standard Lines
                  </button>
                  <button
                    id="toggle-zonal"
                    onClick={() => setMeasurementMode('zonal')}
                    disabled={!results?.global_min_lines}
                    aria-selected={measurementMode === 'zonal'}
                    role="tab"
                    className={`flex-1 py-1.5 px-2 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-singapodent-accent focus:ring-offset-1 disabled:opacity-30 disabled:cursor-not-allowed ${measurementMode === 'zonal'
                        ? 'bg-indigo-600 text-white shadow-sm'
                        : 'text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'
                      }`}
                    title="Zonal minimum-distance sweep method"
                  >
                    ⚡ Zonal Min
                  </button>
                </div>

                <h4 className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400 tracking-wider flex items-center gap-2 border-b border-slate-200 dark:border-slate-700/60 pb-2">
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
                  </svg>
                  Root-to-Bone Distances
                  {measurementMode === 'zonal' && (
                    <span className="ml-auto text-[9px] font-semibold text-indigo-500 dark:text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-1.5 py-0.5 rounded-full uppercase tracking-wide">
                      Zonal Min
                    </span>
                  )}
                </h4>

                {/* Distance grid — switches between Standard (6 slots) and Min Distance (2 bottleneck cards) */}
                {measurementMode === 'zonal' && results?.global_min_lines ? (
                  // ── Min Distance mode: 2 bold bottleneck cards ──────────────────
                  <div className="flex flex-col gap-3">
                    {/* Cervical Offset Controls */}
                    <div className="flex flex-col gap-2 mb-1 p-3 rounded-xl border border-indigo-500/20 bg-indigo-500/5 dark:bg-indigo-500/10 transition-all">
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] font-bold uppercase text-slate-600 dark:text-slate-300 tracking-wide">
                          Cervical Offset: {cervicalOffsetMm.toFixed(1)} mm
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="5"
                        step="0.1"
                        value={cervicalOffsetMm}
                        onChange={(e) => setCervicalOffsetMm(Number(e.target.value))}
                        className="w-full h-1.5 bg-indigo-200 dark:bg-indigo-900 rounded-lg appearance-none cursor-pointer focus:outline-none"
                      />
                      <div className="flex justify-between text-[9px] text-slate-400 font-medium">
                        <span>0 mm</span>
                        <span>5 mm</span>
                      </div>
                    </div>
                    {/* Labial Bottleneck */}
                    {(() => {
                      const currentZonalData = results.global_min_lines[cervicalOffsetMm.toFixed(1)];
                      if (!currentZonalData) return null;

                      const mm = currentZonalData.labial_mm as number;
                      const sev = mm < 0.5 ? 'critical' : mm < 1.0 ? 'warning' : 'normal';
                      const clr = sev === 'critical' ? 'border-red-500/40 bg-red-500/5 dark:bg-red-500/10'
                        : sev === 'warning' ? 'border-amber-500/40 bg-amber-500/5 dark:bg-amber-500/10'
                          : 'border-orange-400/30 bg-orange-400/5 dark:bg-orange-400/10';
                      const valClr = sev === 'critical' ? 'text-red-500' : sev === 'warning' ? 'text-amber-500' : 'text-orange-400';
                      return (
                        <div className={`p-4 rounded-xl border ${clr} flex flex-col gap-1`}>
                          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                            ⚡ Labial Bottleneck
                          </div>
                          <div className={`text-2xl font-bold ${valClr} flex items-end gap-1`}>
                            {mm.toFixed(2)}
                            <span className="text-sm font-semibold text-slate-400 dark:text-slate-500 mb-0.5">mm</span>
                          </div>
                          <div className="text-[9px] text-slate-400 dark:text-slate-500 uppercase tracking-wide">
                            {sev === 'critical' ? '🔴 Critical — Insufficient labial bone' : sev === 'warning' ? '🟡 Warning — Thin labial bone' : '🟢 Monitor — Adequate'}
                          </div>
                        </div>
                      );
                    })()}

                    {/* Palatal Bottleneck */}
                    {(() => {
                      const currentZonalData = results.global_min_lines[cervicalOffsetMm.toFixed(1)];
                      if (!currentZonalData) return null;

                      const mm = currentZonalData.palatal_mm as number;
                      const sev = mm < 0.5 ? 'critical' : mm < 1.0 ? 'warning' : 'normal';
                      const clr = sev === 'critical' ? 'border-red-500/40 bg-red-500/5 dark:bg-red-500/10'
                        : sev === 'warning' ? 'border-amber-500/40 bg-amber-500/5 dark:bg-amber-500/10'
                          : 'border-violet-400/30 bg-violet-400/5 dark:bg-violet-400/10';
                      const valClr = sev === 'critical' ? 'text-red-500' : sev === 'warning' ? 'text-amber-500' : 'text-violet-400';
                      return (
                        <div className={`p-4 rounded-xl border ${clr} flex flex-col gap-1`}>
                          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                            ⚡ Palatal Bottleneck
                          </div>
                          <div className={`text-2xl font-bold ${valClr} flex items-end gap-1`}>
                            {mm.toFixed(2)}
                            <span className="text-sm font-semibold text-slate-400 dark:text-slate-500 mb-0.5">mm</span>
                          </div>
                          <div className="text-[9px] text-slate-400 dark:text-slate-500 uppercase tracking-wide">
                            {sev === 'critical' ? '🔴 Critical — Insufficient palatal bone' : sev === 'warning' ? '🟡 Warning — Thin palatal bone' : '🟢 Monitor — Adequate'}
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                ) : (
                  // ── Standard mode: 6-slot grid ──────────────────────────────────
                  <div className="grid grid-cols-2 gap-4">
                    {/* Labial Column */}
                    <div className="flex flex-col gap-3">
                      <div className="text-center text-[10px] font-bold text-slate-400 dark:text-slate-500 border-b border-slate-200 dark:border-slate-700/60 pb-1.5 uppercase tracking-wide">
                        Labial Plate
                      </div>
                      <DistanceItem label="Crest" value={results.labial_crest} severity={results.labial_crest_severity} />
                      <DistanceItem label="Midroot" value={results.labial_midroot} severity={results.labial_midroot_severity} />
                      <DistanceItem label="Apex (LB)" value={results.labial_apex} severity={results.labial_apex_severity} />
                    </div>

                    {/* Palatal Column */}
                    <div className="flex flex-col gap-3">
                      <div className="text-center text-[10px] font-bold text-slate-400 dark:text-slate-500 border-b border-slate-200 dark:border-slate-700/60 pb-1.5 uppercase tracking-wide">
                        Palatal Plate
                      </div>
                      <DistanceItem label="Crest" value={results.palatal_crest} severity={results.palatal_crest_severity} />
                      <DistanceItem label="Midroot" value={results.palatal_midroot} severity={results.palatal_midroot_severity} />
                      <DistanceItem label="Apex (PB)" value={results.palatal_apex} severity={results.palatal_apex_severity} />
                    </div>
                  </div>
                )}
              </div>

              {/* Biomechanics and Clinical Recommendations Card */}
              <div className="bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700/60 rounded-xl p-5 flex flex-col gap-4 relative overflow-hidden transition-all duration-300">
                <h4 className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400 tracking-wider flex items-center gap-2 border-b border-slate-200 dark:border-slate-700/60 pb-2">
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                  </svg>
                  Orthodontic & Biomechanical Plan
                </h4>

                <div className="flex flex-col gap-3.5">
                  <div className="flex flex-col p-3 bg-slate-100/50 dark:bg-slate-900/40 border border-slate-200/50 dark:border-slate-700/50 rounded-lg">
                    <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Root Apex Position</span>
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-200 mt-0.5 flex items-center gap-1.5">
                      {activeAssessment?.root_apex_position_type === 'Midway' ? '🟢 Midway (Centered)' : activeAssessment?.root_apex_position_type === 'Labial' ? '🔴 Labial Type' : '🟡 Palatal Type'}
                    </span>
                  </div>

                  <div className="flex flex-col p-3 bg-slate-100/50 dark:bg-slate-900/40 border border-slate-200/50 dark:border-slate-700/50 rounded-lg">
                    <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">General Retraction Strategy</span>
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-200 mt-0.5 leading-relaxed">
                      {activeAssessment?.general_retraction_strategy ?? ''}
                    </span>
                  </div>

                  <div className="flex flex-col p-4 bg-emerald-500/5 dark:bg-emerald-500/10 border border-emerald-500/10 dark:border-emerald-500/20 rounded-lg shadow-sm">
                    <h5 className="text-xs font-bold uppercase text-emerald-600 dark:text-emerald-400 mb-1.5 tracking-wider flex items-center gap-1.5">
                      Preferred Biomechanics
                    </h5>
                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                      {activeAssessment?.preferred_biomechanics ?? ''}
                    </p>
                  </div>

                  <div className="flex flex-col p-4 bg-rose-500/5 dark:bg-rose-500/10 border border-rose-500/10 dark:border-rose-500/20 rounded-lg shadow-sm">
                    <h5 className="text-xs font-bold uppercase text-rose-600 dark:text-rose-400 mb-1.5 tracking-wider flex items-center gap-1.5">
                      Biomechanics to Avoid
                    </h5>
                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                      {activeAssessment?.biomechanics_to_avoid ?? ''}
                    </p>
                  </div>

                  <div className="flex flex-col p-4 bg-slate-100/50 dark:bg-slate-900/40 border border-slate-200/50 dark:border-slate-700/50 rounded-lg">
                    <h5 className="text-xs font-bold uppercase text-slate-400 dark:text-slate-500 mb-1.5 tracking-wider flex items-center gap-1.5">
                      Clinical Implication
                    </h5>
                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal italic">
                      "{activeAssessment?.clinical_implication ?? ''}"
                    </p>
                  </div>
                </div>
              </div>

              {/* Disclaimer Card */}
              <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4 flex flex-col gap-2 shadow-sm">
                <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400 text-[10px] font-bold uppercase tracking-wider">
                  <span>⚠️ Clinical Disclaimer</span>
                </div>
                <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-relaxed">
                  {activeAssessment?.bone_thickness_interpretation ?? ''} Estimation model based on 2D lateral cephalometric imaging and does not replace CBCT evaluation.
                </p>
              </div>

              {/* Export PDF Button */}
              <button
                onClick={exportPDFReport}
                className="w-full mt-2 py-3 px-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl shadow-md font-bold transition-all duration-200 flex items-center justify-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                Export PDF Report
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Floating Action Button for Analysis */}
      <div className="absolute bottom-6 right-6 z-50">
        <button
          onClick={handleAnalyze}
          disabled={!file || isLoading}
          className="py-3.5 px-10 text-sm font-semibold bg-singapodent-accent text-singapodent-primary dark:text-white rounded-full shadow-xl shadow-singapodent-accent/20 hover:shadow-2xl hover:brightness-110 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-singapodent-accent/50"
        >
          {isLoading ? 'Processing...' : 'Run AI Analysis'}
        </button>
      </div>
    </div>
  );
}
