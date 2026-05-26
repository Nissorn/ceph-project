import React, { useState, useEffect, useCallback } from 'react';
import UploadZone from './UploadZone';
import MetricCard from './MetricCard';
import CephCanvasEditor from './CephCanvasEditor';

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
      
      const rawMaxThick = payload.bone_thickness?.labial_min_mm ?? payload.maxillary?.bone_thickness_mm ?? 0;
      const maxillary_thickness = typeof rawMaxThick === 'number' && !isNaN(rawMaxThick) ? Number(rawMaxThick.toFixed(2)) : 0;

      const rawMandThick = payload.bone_thickness?.mandibular_min_mm ?? payload.mandibular?.bone_thickness_mm ?? 0;
      const mandibular_thickness = typeof rawMandThick === 'number' && !isNaN(rawMandThick) ? Number(rawMandThick.toFixed(2)) : 0;

      // Robust status classifications tailored to precise clinical parameters
      const u1_pp_status = u1_pp_angle > 115 ? 'warning' : u1_pp_angle < 105 ? 'warning' : 'normal';
      const maxillary_status = maxillary_thickness < 2.0 ? 'critical' : maxillary_thickness < 2.5 ? 'warning' : 'normal';
      const mandibular_status = mandibular_thickness < 2.0 ? 'critical' : mandibular_thickness < 2.5 ? 'warning' : 'normal';

      const normalizedResults = {
        u1_pp_angle,
        u1_pp_status,
        maxillary_thickness,
        maxillary_status,
        mandibular_thickness,
        mandibular_status,
        interpretation: payload.classification?.interpretation || payload.interpretation || payload.summary || 'Analysis completed successfully. Review extracted biomechanical structures below.',
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
                subtitle="° Degrees" 
                status={results.u1_pp_status} 
              />
              <MetricCard 
                title="Maxillary Bone" 
                value={results.maxillary_thickness} 
                subtitle="mm" 
                status={results.maxillary_status} 
              />
              <MetricCard 
                title="Mandibular Bone" 
                value={results.mandibular_thickness} 
                subtitle="mm" 
                status={results.mandibular_status} 
              />
              <div className="mt-2 p-5 bg-singapodent-primary/5 dark:bg-singapodent-primary/15 border border-singapodent-primary/15 dark:border-singapodent-primary/20 rounded-xl overflow-hidden shadow-sm">
                 <h4 className="text-xs font-semibold uppercase text-singapodent-primary dark:text-singapodent-accent mb-3 tracking-wider flex items-center gap-2">
                   <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                     <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                   </svg>
                   Clinical Interpretation
                 </h4>
                 <p className="text-sm text-slate-700 dark:text-slate-200 leading-relaxed font-normal">
                   {results.interpretation}
                 </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
