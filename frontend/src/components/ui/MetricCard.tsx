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
    statusIndicator = 'bg-singapodent-accent shadow-[0_0_10px_rgba(242,140,40,0.5)]';
  } else if (status === 'critical') {
    valueColor = 'text-red-500 dark:text-red-400';
    statusIndicator = 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]';
  } else if (status === 'normal') {
    statusIndicator = 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]';
  }

  return (
    <div className="bg-white/60 dark:bg-slate-800/60 backdrop-blur-md border border-white/40 dark:border-slate-700/50 shadow-xl rounded-2xl p-5 flex flex-col transition-all duration-300 hover:shadow-2xl relative overflow-hidden group">
      <div className={`absolute left-0 top-0 bottom-0 w-1 ${statusIndicator} opacity-80 group-hover:opacity-100 transition-opacity`} />
      <h3 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider ml-2">{title}</h3>
      <div className={`mt-3 text-3xl font-semibold ml-2 ${valueColor}`}>{value}</div>
    </div>
  );
}
