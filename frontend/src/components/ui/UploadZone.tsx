import React, { useRef } from 'react';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
}

export default function UploadZone({ onFileSelect }: UploadZoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div
      className="flex flex-col items-center justify-center w-full h-full min-h-[500px] border-2 border-dashed border-slate-200 dark:border-slate-700/60 rounded-xl p-12 text-center cursor-pointer bg-slate-50 dark:bg-slate-800/40 hover:bg-slate-100/80 dark:hover:bg-slate-700/40 hover:border-slate-300 dark:hover:border-slate-600 transition-all duration-200 overflow-hidden relative group"
      onClick={() => fileInputRef.current?.click()}
    >
      <input 
        type="file" 
        className="hidden" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        accept="image/*" 
      />
      <div className="bg-white dark:bg-slate-800/80 border border-slate-100 dark:border-slate-700/80 w-16 h-16 flex items-center justify-center rounded-xl mb-6 text-singapodent-primary dark:text-singapodent-accent group-hover:scale-105 transition-transform duration-200 ease-out z-10">
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      </div>
      <h3 className="text-2xl font-light tracking-tight text-slate-800 dark:text-slate-200 mb-3 z-10">
        Upload Cephalogram Scan
      </h3>
      <p className="text-sm font-medium text-slate-500 dark:text-slate-400 max-w-sm leading-relaxed z-10">
        Drag and drop a medical scan here, or click to browse. Supports high-resolution <span className="text-slate-700 dark:text-slate-300">PNG</span>, <span className="text-slate-700 dark:text-slate-300">JPG</span>, and <span className="text-slate-700 dark:text-slate-300">DICOM</span>.
      </p>
    </div>
  );
}
