import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  status?: 'normal' | 'warning' | 'critical';
}

export default function MetricCard({ title, value, status = 'normal' }: MetricCardProps) {
  let valueColor = 'text-slate-800 dark:text-white';
  let statusIndicator = 'bg-slate-200 dark:bg-slate-700';

  if (status === 'warning') {
    valueColor = 'text-singapodent-accent';
    statusIndicator = 'bg-singapodent-accent';
  } else if (status === 'critical') {
    valueColor = 'text-red-500 dark:text-red-400';
    statusIndicator = 'bg-red-500';
  } else if (status === 'normal') {
    statusIndicator = 'bg-green-500';
  }

  return (
    <div className="bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700/60 rounded-xl p-5 flex flex-col relative overflow-hidden">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-widest">{title}</h3>
        <div className={`w-2.5 h-2.5 rounded-full ${statusIndicator}`} />
      </div>
      <div className={`mt-4 text-4xl font-light tracking-tight ${valueColor}`}>{value}</div>
    </div>
  );
}
