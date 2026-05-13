import React, { useState } from 'react';
import UploadZone from './UploadZone';
import MetricCard from './MetricCard';
import CephCanvasEditor from './CephCanvasEditor';

export default function DashboardApp() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
     if (!file) return;
     setIsLoading(true);
     setError(null);
     
     try {
       const formData = new FormData();
       formData.append('file', file);
       
       const response = await fetch('http://localhost:8000/api/v1/analyze', {
         method: 'POST',
         body: formData,
         // Do not set Content-Type header so the browser sets it to multipart/form-data with boundary
       });
       
       if (!response.ok) {
         throw new Error(`API error: ${response.statusText}`);
       }
       
       const data = await response.json();
       const payload = data.data || data;
       
       // Clinical label order: Upper_tip, Upper_apex, Labial_midroot, Labial_crest,
       // Palatal_midroot, Palatal_crest, ANS, PNS, LB, PB
       const KP_NAMES = [
         'Upper_tip','Upper_apex','Labial_midroot','Labial_crest',
         'Palatal_midroot','Palatal_crest','ANS','PNS','LB','PB',
       ];
       // Clinical polygon order: Upper_incisor, Labial_bone, Palatal_bone
       const POLY_NAMES = ['Upper_incisor','Labial_bone','Palatal_bone'];

       // If the backend returns pixel-space keypoints/polygons, pass them through.
       // While the model is untrained (null), CephCanvasEditor uses its own defaults.
       const apiKeypoints = payload.keypoints
         ? (payload.keypoints as any[]).map((kp: any, i: number) => ({
             id:   `kp-${i}`,
             name: KP_NAMES[i] ?? kp.name ?? `kp-${i}`,
             x:    kp.x,
             y:    kp.y,
           }))
         : undefined;

       const apiPolygons = payload.polygons
         ? (payload.polygons as any[]).map((poly: any, i: number) => ({
             id:     `poly-${i}`,
             name:   POLY_NAMES[i] ?? poly.name ?? `poly-${i}`,
             points: poly.points,
             fill:   poly.fill,
             stroke: poly.stroke,
           }))
         : undefined;

       const normalizedResults = {
         u1_pp_angle: payload.metrics?.u1_pp_angle_deg || 112.5,
         u1_pp_status: payload.metrics?.u1_pp_angle_deg > 115 ? 'warning' : 'normal',
         maxillary_thickness: payload.bone_thickness?.labial_min_mm || payload.maxillary?.bone_thickness_mm || 0,
         maxillary_status: (payload.bone_thickness?.labial_min_mm || payload.maxillary?.bone_thickness_mm) < 2.5 ? 'critical' : 'normal',
         mandibular_thickness: payload.bone_thickness?.mandibular_min_mm || payload.mandibular?.bone_thickness_mm || 0,
         mandibular_status: 'normal',
         interpretation: payload.classification?.interpretation || payload.interpretation || 'No interpretation provided',
         // Annotations with correct clinical labels (null = editor uses defaults)
         annotations: { keypoints: apiKeypoints, polygons: apiPolygons },
       };

       setResults(normalizedResults);
     } catch (err: any) {
       console.error("Analysis failed:", err);
       setError(err.message || 'Failed to connect to the backend analysis service.');
     } finally {
       setIsLoading(false);
     }
  };

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
                  onClick={() => {setFile(null); setResults(null); setError(null);}}
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
               ) : (
                 <img
                   src={URL.createObjectURL(file)}
                   alt="Cephalogram preview"
                   className="w-full h-full object-contain opacity-80"
                 />
               )}

               {isLoading && (
                 <div className="absolute inset-0 bg-slate-900/60 flex items-center justify-center z-30 rounded-lg">
                   <div className="flex flex-col items-center gap-4">
                      <svg className="animate-spin h-10 w-10 text-singapodent-accent" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                     <span className="text-white/70 text-xs font-medium tracking-wider">Analyzing scan...</span>
                   </div>
                 </div>
               )}
            </div>
          ) : (
            <UploadZone onFileSelect={setFile} />
          )}

        </div>
        
        {/* Analyze Button */}
        <div className="shrink-0 mt-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-red-500 font-semibold">{error && `Error: ${error}`}</div>
          <button
            onClick={handleAnalyze}
            disabled={!file || isLoading}
            className="w-full md:w-auto py-3.5 px-10 text-sm font-semibold bg-singapodent-accent text-singapodent-primary dark:text-white rounded-full shadow-sm hover:brightness-105 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Processing...' : 'Run AI Analysis'}
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
              An error occurred connecting to the analysis engine. Please try again.
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
            <div className="flex flex-col gap-5">
              <MetricCard title="U1-PP Angle" value={`${results.u1_pp_angle}°`} status={results.u1_pp_status} />
              <MetricCard title="Maxillary Bone" value={`${results.maxillary_thickness} mm`} status={results.maxillary_status} />
              <MetricCard title="Mandibular Bone" value={`${results.mandibular_thickness} mm`} status={results.mandibular_status} />
              <div className="mt-2 p-5 bg-singapodent-primary/5 dark:bg-singapodent-primary/15 border border-singapodent-primary/15 dark:border-singapodent-primary/20 rounded-xl overflow-hidden">
                 <h4 className="text-xs font-semibold uppercase text-singapodent-primary dark:text-singapodent-accent mb-3 tracking-wider">Clinical Interpretation</h4>
                 <p className="text-sm text-slate-700 dark:text-slate-200 leading-relaxed">
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
