import React, { useState, useEffect, useCallback } from 'react';
import UploadZone from './UploadZone';
import MetricCard from './MetricCard';
import CephCanvasEditor from './CephCanvasEditor';

const DistanceItem = ({ label, value }: { label: string; value: number }) => {
  let statusColor = 'text-emerald-500 dark:text-emerald-400';
  let badgeColor = 'bg-emerald-500/10 border-emerald-500/20 dark:bg-emerald-950/20';
  let warningIcon = null;
  let desc = 'Safe';

  if (value < 0.5) {
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

export default function DashboardApp() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

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
  }, []);

  const handleReset = useCallback(() => {
    setFile(null);
    setResults(null);
    setError(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
     
    // Deterministic network timeout protection via AbortController
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 35000); // 35 seconds max boundary

    try {
      const formData = new FormData();
      formData.append('file', file);
       
      // Resilient base URL derivation supporting distinct environments
      const baseUrl = (import.meta.env && import.meta.env.VITE_API_URL) || 'http://localhost:8000';
      const response = await fetch(`${baseUrl}/api/v1/analyze`, {
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

      // Clinical label order: Upper_tip, Upper_apex, Labial_midroot, Labial_crest,
      // Palatal_midroot, Palatal_crest, ANS, PNS, LB, PB
      const KP_NAMES = [
        'Upper_tip','Upper_apex','Labial_midroot','Labial_crest',
        'Palatal_midroot','Palatal_crest','ANS','PNS','LB','PB',
      ];
      // Backend returns segmentation as keyed dict; matches CephCanvasEditor POLY_PALETTE order
      const SEG_NAMES = ['Upper_incisor','Labial_bone','Palatal_bone'] as const;
      const SEG_PALETTE = [
        { fill: 'rgba(6, 182, 212, 0.15)',  stroke: 'rgba(6, 182, 212, 0.9)'  },
        { fill: 'rgba(236, 72, 153, 0.15)', stroke: 'rgba(236, 72, 153, 0.9)' },
        { fill: 'rgba(16, 185, 129, 0.15)', stroke: 'rgba(16, 185, 129, 0.9)' },
      ];

      const apiKeypoints = Array.isArray(payload.landmarks)
        ? payload.landmarks.map((kp: any, i: number) => ({
            id:   `kp-${i}`,
            name: KP_NAMES[i] ?? kp?.name ?? `kp-${i}`,
            x:    Number(kp?.x ?? 0),
            y:    Number(kp?.y ?? 0),
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
              id:     `poly-${i}`,
              name,
              points: flatPoints,
              fill:   SEG_PALETTE[i].fill,
              stroke: SEG_PALETTE[i].stroke,
            };
          })
        : undefined;

      // Safely parse and cleanly format numeric metrics to prevent excessive float layouts
      const rawAngle = payload.metrics?.u1_pp_angle_deg ?? 112.5;
      const u1_pp_angle = typeof rawAngle === 'number' && !isNaN(rawAngle) ? Number(rawAngle.toFixed(1)) : 112.5;

      const u1_pp_status = u1_pp_angle > 115 ? 'warning' : u1_pp_angle < 105 ? 'warning' : 'normal';
      let u1_pp_desc = 'Normal Inclination';
      if (u1_pp_angle < 105.0) {
        u1_pp_desc = 'Retroclined';
      } else if (u1_pp_angle > 115.0) {
        u1_pp_desc = 'Proclined';
      }

      // Extract new 6 distances in mm
      const labial_crest = Number((payload.metrics?.labial_crest_mm ?? 1.2).toFixed(2));
      const labial_midroot = Number((payload.metrics?.labial_midroot_mm ?? 1.5).toFixed(2));
      const labial_apex = Number((payload.metrics?.labial_apex_mm ?? 1.0).toFixed(2));
      const palatal_crest = Number((payload.metrics?.palatal_crest_mm ?? 1.4).toFixed(2));
      const palatal_midroot = Number((payload.metrics?.palatal_midroot_mm ?? 1.6).toFixed(2));
      const palatal_apex = Number((payload.metrics?.palatal_apex_mm ?? 1.1).toFixed(2));

      const bone_thickness_type = payload.metrics?.bone_thickness_type ?? 'Type 1 – Thick';
      const bone_thickness_interpretation = payload.metrics?.bone_thickness_interpretation ?? 'Thick alveolar bone; Favorable bone support.';
      const root_apex_position_type = payload.metrics?.root_apex_position_type ?? 'Midway';
      const general_retraction_strategy = payload.metrics?.general_retraction_strategy ?? '';
      const preferred_biomechanics = payload.metrics?.preferred_biomechanics ?? '';
      const biomechanics_to_avoid = payload.metrics?.biomechanics_to_avoid ?? '';
      const clinical_implication = payload.metrics?.clinical_implication ?? '';

      const normalizedResults = {
        u1_pp_angle,
        u1_pp_status,
        u1_pp_desc,
        labial_crest,
        labial_midroot,
        labial_apex,
        palatal_crest,
        palatal_midroot,
        palatal_apex,
        bone_thickness_type,
        bone_thickness_interpretation,
        root_apex_position_type,
        general_retraction_strategy,
        preferred_biomechanics,
        biomechanics_to_avoid,
        clinical_implication,
        annotations: { keypoints: apiKeypoints, polygons: apiPolygons },
      };

      setResults(normalizedResults);
    } catch (err: any) {
      console.error("Analysis failed:", err);
      if (err.name === 'AbortError') {
        setError('Analysis request timed out after 35 seconds. Verify inference engine availability.');
      } else {
        setError(err.message || 'Failed to connect to the backend analysis service.');
      }
    } finally {
      clearTimeout(timeoutId);
      setIsLoading(false);
    }
  }, [file]);

  return (
    <div className="grid grid-cols-12 gap-8 h-full">
      {/* Left Panel - Image Viewer */}
      <div className="col-span-12 lg:col-span-8 flex flex-col h-full overflow-hidden relative">
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
                  className="absolute top-3 right-3 z-20 bg-black/40 hover:bg-red-500/80 text-white p-2 rounded-full transition duration-150"
                  title="Remove Image"
               >
                 <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
               </button>

               {/* After analysis: interactive Ceph Editor; before: plain preview */}
               {results && !isLoading ? (
                 <div className="absolute inset-0 z-10">
                   <CephCanvasEditor
                     imageFile={file}
                     initialKeypoints={results.annotations?.keypoints}
                     initialPolygons={results.annotations?.polygons}
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
        
        {/* Analyze Button */}
        <div className="shrink-0 mt-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-red-500 font-medium text-sm flex items-center gap-2">
            {error && (
              <>
                <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>{error}</span>
              </>
            )}
          </div>
          <button
            onClick={handleAnalyze}
            disabled={!file || isLoading}
            className="w-full md:w-auto py-3.5 px-10 text-sm font-semibold bg-singapodent-accent text-singapodent-primary dark:text-white rounded-full shadow-sm hover:brightness-105 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-singapodent-accent/50"
          >
            {isLoading ? 'Processing Inference...' : 'Run AI Analysis'}
          </button>
        </div>

      </div>

      {/* Right Panel - Metrics */}
      <div className="col-span-12 lg:col-span-4 h-full pb-20 overflow-y-auto pr-2 custom-scrollbar">
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
               {[1,2,3].map(i => (
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
                subtitle={`° (${results.u1_pp_desc})`} 
                status={results.u1_pp_status} 
              />
              <MetricCard 
                title="Alveolar Bone Phenotype" 
                value={results.bone_thickness_type} 
                subtitle={getShortBoneTypeLabel(results.bone_thickness_type)} 
                status={results.bone_thickness_type.includes('Type 1') ? 'normal' : results.bone_thickness_type.includes('Type 4') ? 'critical' : 'warning'} 
              />

              {/* Distance Matrix Card */}
              <div className="bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700/60 rounded-xl p-5 flex flex-col gap-4 relative overflow-hidden transition-all duration-300">
                <h4 className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400 tracking-wider flex items-center gap-2 border-b border-slate-200 dark:border-slate-700/60 pb-2">
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
                  </svg>
                  Root-to-Bone Distances
                </h4>
                <div className="grid grid-cols-2 gap-4">
                  {/* Labial Column */}
                  <div className="flex flex-col gap-3">
                    <div className="text-center text-[10px] font-bold text-slate-400 dark:text-slate-500 border-b border-slate-200 dark:border-slate-700/60 pb-1.5 uppercase tracking-wide">
                      Labial Plate
                    </div>
                    <DistanceItem label="Crest" value={results.labial_crest} />
                    <DistanceItem label="Midroot" value={results.labial_midroot} />
                    <DistanceItem label="Apex (LB)" value={results.labial_apex} />
                  </div>

                  {/* Palatal Column */}
                  <div className="flex flex-col gap-3">
                    <div className="text-center text-[10px] font-bold text-slate-400 dark:text-slate-500 border-b border-slate-200 dark:border-slate-700/60 pb-1.5 uppercase tracking-wide">
                      Palatal Plate
                    </div>
                    <DistanceItem label="Crest" value={results.palatal_crest} />
                    <DistanceItem label="Midroot" value={results.palatal_midroot} />
                    <DistanceItem label="Apex (PB)" value={results.palatal_apex} />
                  </div>
                </div>
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
                      {results.root_apex_position_type === 'Midway' ? '🟢 Midway (Centered)' : results.root_apex_position_type === 'Labial' ? '🔴 Labial Type' : '🟡 Palatal Type'}
                    </span>
                  </div>

                  <div className="flex flex-col p-3 bg-slate-100/50 dark:bg-slate-900/40 border border-slate-200/50 dark:border-slate-700/50 rounded-lg">
                    <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">General Retraction Strategy</span>
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-200 mt-0.5 leading-relaxed">
                      {results.general_retraction_strategy}
                    </span>
                  </div>

                  <div className="flex flex-col p-4 bg-emerald-500/5 dark:bg-emerald-500/10 border border-emerald-500/10 dark:border-emerald-500/20 rounded-lg shadow-sm">
                    <h5 className="text-xs font-bold uppercase text-emerald-600 dark:text-emerald-400 mb-1.5 tracking-wider flex items-center gap-1.5">
                      Preferred Biomechanics
                    </h5>
                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                      {results.preferred_biomechanics}
                    </p>
                  </div>

                  <div className="flex flex-col p-4 bg-rose-500/5 dark:bg-rose-500/10 border border-rose-500/10 dark:border-rose-500/20 rounded-lg shadow-sm">
                    <h5 className="text-xs font-bold uppercase text-rose-600 dark:text-rose-400 mb-1.5 tracking-wider flex items-center gap-1.5">
                      Biomechanics to Avoid
                    </h5>
                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                      {results.biomechanics_to_avoid}
                    </p>
                  </div>

                  <div className="flex flex-col p-4 bg-slate-100/50 dark:bg-slate-900/40 border border-slate-200/50 dark:border-slate-700/50 rounded-lg">
                    <h5 className="text-xs font-bold uppercase text-slate-400 dark:text-slate-500 mb-1.5 tracking-wider flex items-center gap-1.5">
                      Clinical Implication
                    </h5>
                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal italic">
                      "{results.clinical_implication}"
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
                  {results.bone_thickness_interpretation} Estimation model based on 2D lateral cephalometric imaging and does not replace CBCT evaluation.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
