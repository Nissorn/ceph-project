import React, { memo } from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  status?: 'normal' | 'warning' | 'critical';
  subtitle?: string;
}

const MetricCard = memo(function MetricCard({ title, value, status = 'normal', subtitle }: MetricCardProps) {
  const safeValue = value ?? '--';

  let config = {
    badgeBg: 'bg-emerald-50 dark:bg-emerald-950/30',
    badgeText: 'text-emerald-600 dark:text-emerald-400',
    badgeBorder: 'border-emerald-200/60 dark:border-emerald-800/40',
    dotColor: 'bg-emerald-500',
    statusLabel: 'Normal',
    valueColor: 'text-slate-800 dark:text-slate-100',
    borderHover: 'hover:border-emerald-300 dark:hover:border-emerald-500',
  };

  if (status === 'warning') {
    config = {
      badgeBg: 'bg-amber-50 dark:bg-amber-950/30',
      badgeText: 'text-amber-600 dark:text-amber-400',
      badgeBorder: 'border-amber-200/60 dark:border-amber-800/40',
      dotColor: 'bg-amber-500',
      statusLabel: 'Warning',
      valueColor: 'text-singapodent-accent dark:text-amber-400',
      borderHover: 'hover:border-amber-300 dark:hover:border-amber-500',
    };
  } else if (status === 'critical') {
    config = {
      badgeBg: 'bg-rose-50 dark:bg-rose-950/30',
      badgeText: 'text-rose-600 dark:text-rose-400',
      badgeBorder: 'border-rose-200/60 dark:border-rose-800/40',
      dotColor: 'bg-rose-500',
      statusLabel: 'Critical',
      valueColor: 'text-rose-600 dark:text-rose-400',
      borderHover: 'hover:border-rose-300 dark:hover:border-rose-500',
    };
  }

  return (
    <div className={`bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl p-5 flex flex-col relative overflow-hidden transition-colors duration-200 ${config.borderHover} group`}>
      <div className="flex items-center justify-between z-10">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-widest leading-tight">
          {title}
        </h3>
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium tracking-wide border ${config.badgeBg} ${config.badgeText} ${config.badgeBorder} shadow-sm`}>
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${config.dotColor}`} />
          <span>{config.statusLabel}</span>
        </div>
      </div>

      <div className="mt-4 flex items-baseline gap-2 z-10">
        <div className={`text-4xl font-light tracking-tight ${config.valueColor}`}>
          {safeValue}
        </div>
        {subtitle && (
          <span className="text-xs font-medium text-slate-400 tracking-wide">
            {subtitle}
          </span>
        )}
      </div>
    </div>
  );
});

export default MetricCard;
