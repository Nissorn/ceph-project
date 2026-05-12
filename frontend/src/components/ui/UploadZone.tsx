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
      className="flex flex-col items-center justify-center w-full h-full min-h-[500px] border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-3xl p-12 text-center cursor-pointer bg-slate-50/30 dark:bg-slate-800/30 hover:bg-slate-100/50 dark:hover:bg-slate-700/50 hover:border-singapodent-accent transition-all duration-300"
      onClick={() => fileInputRef.current?.click()}
    >
      <input 
        type="file" 
        className="hidden" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        accept="image/*" 
      />
      <div className="bg-white dark:bg-slate-800 shadow-md p-5 rounded-full mb-6 text-singapodent-primary dark:text-slate-300">
        <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      </div>
      <h3 className="text-xl font-medium text-slate-800 dark:text-slate-200 mb-2">
        Upload Cephalogram Image
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 max-w-sm">
        Drag and drop a medical scan here, or click to browse. Supports high-resolution PNG, JPG, and DICOM (simulated).
      </p>
    </div>
  );
}
