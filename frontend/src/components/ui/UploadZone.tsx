import React, { useRef, useState } from 'react';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
}

export default function UploadZone({ onFileSelect }: UploadZoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateAndProcessFile = (file: File) => {
    setError(null);

    const validExtensions = ['.png', '.jpg', '.jpeg', '.dcm', '.dicom'];
    const fileName = file.name.toLowerCase();
    const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
    const isImageType = file.type.startsWith('image/');

    if (!hasValidExtension && !isImageType) {
      setError('Unsupported file format. Please upload a valid PNG, JPG, or DICOM scan.');
      return;
    }

    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      setError(`File size exceeds 50MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB).`);
      return;
    }

    onFileSelect(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div
      className={`flex flex-col items-center justify-center w-full h-full min-h-[500px] border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 overflow-hidden relative group ${
        isDragging
          ? 'border-singapodent-accent bg-singapodent-accent/5 scale-[0.99]'
          : 'border-slate-200 bg-slate-50 hover:bg-slate-100/80 hover:border-slate-300'
      }`}
      onClick={() => fileInputRef.current?.click()}
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {error && (
        <div
          className="absolute top-6 left-6 right-6 z-20 flex items-center justify-between bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-xl shadow-sm text-sm text-left"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center gap-3">
            <svg className="w-5 h-5 shrink-0 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="font-medium">{error}</span>
          </div>
          <button
            onClick={() => setError(null)}
            className="p-1.5 hover:bg-red-100 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-red-400"
            title="Dismiss error"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}

      <input
        type="file"
        className="hidden"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/png, image/jpeg, .dcm, .dicom"
      />

      <div className={`bg-white border w-16 h-16 flex items-center justify-center rounded-xl mb-6 transition-all duration-300 ease-out z-10 shadow-sm ${
        isDragging
          ? 'border-singapodent-accent text-singapodent-accent scale-110 shadow-md'
          : 'border-slate-100 text-singapodent-primary group-hover:scale-105'
      }`}>
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      </div>

      <h3 className="text-2xl font-light tracking-tight text-slate-800 mb-3 z-10">
        {isDragging ? 'Drop Scan to Upload' : 'Upload Cephalogram Scan'}
      </h3>

      <p className="text-sm font-medium text-slate-500 max-w-sm leading-relaxed z-10">
        Drag and drop a medical scan here, or click to browse. Supports high-resolution <span className="text-slate-700 font-semibold">PNG</span>, <span className="text-slate-700 font-semibold">JPG</span>, and <span className="text-slate-700 font-semibold">DICOM</span> up to 50MB.
      </p>
    </div>
  );
}
