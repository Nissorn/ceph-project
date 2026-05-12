import React, { useState } from 'react';
import UploadZone from './UploadZone';
import MetricCard from './MetricCard';

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
       
       const normalizedResults = {
         u1_pp_angle: payload.metrics?.u1_pp_angle_deg || 112.5,
         u1_pp_status: payload.metrics?.u1_pp_angle_deg > 115 ? 'warning' : 'normal',
         maxillary_thickness: payload.bone_thickness?.labial_min_mm || payload.maxillary?.bone_thickness_mm || 0,
         maxillary_status: (payload.bone_thickness?.labial_min_mm || payload.maxillary?.bone_thickness_mm) < 2.5 ? 'critical' : 'normal',
         mandibular_thickness: payload.bone_thickness?.mandibular_min_mm || payload.mandibular?.bone_thickness_mm || 0,
         mandibular_status: 'normal',
         interpretation: payload.classification?.interpretation || payload.interpretation || 'No interpretation provided',
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
        <div className="bg-white/40 dark:bg-slate-800/40 backdrop-blur-xl border border-white/60 dark:border-slate-700/50 shadow-2xl rounded-3xl p-2 flex-grow overflow-hidden flex flex-col relative">
          
          {file ? (
            <div className="flex-1 min-h-0 relative flex items-center justify-center bg-slate-900 rounded-2xl overflow-hidden group">
               {/* Just showing filename for mock preview */}
               <div className="absolute top-4 left-4 z-10 bg-black/60 text-white px-3 py-1 rounded-full text-sm font-medium backdrop-blur-sm">
                 {file.name}
               </div>

               <img 
                 src={URL.createObjectURL(file)} 
                 alt="Cephalogram preview" 
                 className="w-full h-full object-contain opacity-80"
               />

               {isLoading && (
                 <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center z-20">
                   <div className="flex flex-col items-center">
                    <svg className="animate-spin h-12 w-12 text-singapodent-accent mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span className="text-white font-medium tracking-wide shadow-black">RUNNING AI ANALYSIS...</span>
                   </div>
                 </div>
               )}

               <button 
                  onClick={() => {setFile(null); setResults(null); setError(null);}}
                  className="absolute top-4 right-4 z-10 bg-red-500/80 hover:bg-red-500 text-white p-2 rounded-full backdrop-blur-sm transition opacity-0 group-hover:opacity-100"
                  title="Remove Image"
               >
                 <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
               </button>
            </div>
          ) : (
            <UploadZone onFileSelect={setFile} />
          )}

        </div>
        
        {/* Analyze Button */}
        <div className="shrink-0 mt-4 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-red-500 font-semibold">{error && `Error: ${error}`}</div>
          <button 
            onClick={handleAnalyze}
            disabled={!file || isLoading}
            className="w-full md:w-auto py-4 px-10 text-lg bg-singapodent-primary dark:bg-singapodent-accent text-white font-bold rounded-full shadow-[0_10px_20px_rgba(12,35,64,0.3)] dark:shadow-[0_10px_20px_rgba(242,140,40,0.3)] hover:scale-[1.02] hover:-translate-y-1 transition-all disabled:opacity-50 disabled:hover:scale-100 disabled:hover:translate-y-0 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Processing...' : 'Run Analysis'}
          </button>
        </div>

      </div>

      {/* Right Panel - Metrics */}
      <div className="col-span-12 lg:col-span-4 h-full pb-20 overflow-y-auto pr-2 custom-scrollbar">
        <div className="flex flex-col gap-6">
          <h2 className="text-xl font-bold text-slate-800 dark:text-white mb-2 pl-2">Medical Metrics</h2>
          
          {!results && !isLoading && !error && (
            <div className="h-64 flex items-center justify-center border border-dashed border-slate-300 dark:border-slate-700 rounded-3xl px-8 text-center text-slate-400">
              Awaiting image upload to calculate biomechanical mappings.
            </div>
          )}
          
          {error && !isLoading && (
            <div className="h-64 flex items-center justify-center border border-dashed border-red-300 dark:border-red-800/50 bg-red-50 dark:bg-red-900/10 rounded-3xl px-8 text-center text-red-500">
              An error occurred connecting to the analysis engine. Please try again.
            </div>
          )}

          {isLoading && (
            <div className="flex flex-col gap-4">
               {[1,2,3].map(i => (
                 <div key={i} className="h-28 bg-slate-200/50 dark:bg-slate-800/50 animate-pulse rounded-2xl"></div>
               ))}
            </div>
          )}

          {results && !isLoading && (
            <div className="flex flex-col gap-5">
              <MetricCard title="U1-PP Angle" value={`${results.u1_pp_angle}°`} status={results.u1_pp_status} />
              <MetricCard title="Maxillary Bone" value={`${results.maxillary_thickness} mm`} status={results.maxillary_status} />
              <MetricCard title="Mandibular Bone" value={`${results.mandibular_thickness} mm`} status={results.mandibular_status} />
              <div className="mt-4 p-5 bg-singapodent-primary/5 dark:bg-singapodent-primary/20 border border-singapodent-primary/10 dark:border-singapodent-primary/30 rounded-2xl">
                 <h4 className="text-xs font-bold uppercase text-singapodent-primary dark:text-singapodent-accent mb-2 tracking-wider">Clinical Interpretation</h4>
                 <p className="text-sm text-slate-700 dark:text-slate-300">
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
