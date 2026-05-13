import React, { memo } from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  status?: 'normal' | 'warning' | 'critical';
  subtitle?: string;
}

const MetricCard = memo(function MetricCard({ title, value, status = 'normal', subtitle }: MetricCardProps) {
  // Defensive hardening against nullish or empty payload metrics
  const safeValue = value ?? '--';

  // State-of-the-art curated status mapping avoiding generic primary browser defaults
  let config = {
    badgeBg: 'bg-emerald-50 dark:bg-emerald-950/40',
    badgeText: 'text-emerald-600 dark:text-emerald-400',
    badgeBorder: 'border-emerald-200/60 dark:border-emerald-800/60',
    dotColor: 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]',
    statusLabel: 'Normal',
    valueColor: 'text-slate-800 dark:text-slate-100',
    bgGlow: 'from-emerald-500/5 via-transparent to-transparent',
    borderGlow: 'hover:border-emerald-300 dark:hover:border-emerald-700/60',
  };

  if (status === 'warning') {
    config = {
      badgeBg: 'bg-amber-50 dark:bg-amber-950/40',
      badgeText: 'text-amber-600 dark:text-amber-400',
      badgeBorder: 'border-amber-200/60 dark:border-amber-800/60',
      dotColor: 'bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.4)]',
      statusLabel: 'Warning',
      valueColor: 'text-singapodent-accent dark:text-amber-300',
      bgGlow: 'from-amber-500/5 via-transparent to-transparent',
      borderGlow: 'hover:border-amber-300 dark:hover:border-amber-700/60',
    };
  } else if (status === 'critical') {
    config = {
      badgeBg: 'bg-rose-50 dark:bg-rose-950/40',
      badgeText: 'text-rose-600 dark:text-rose-400',
      badgeBorder: 'border-rose-200/60 dark:border-rose-800/60',
      dotColor: 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.4)] animate-pulse',
      statusLabel: 'Critical',
      valueColor: 'text-rose-600 dark:text-rose-400 font-normal',
      bgGlow: 'from-rose-500/5 via-transparent to-transparent',
      borderGlow: 'hover:border-rose-300 dark:hover:border-rose-700/60',
    };
  }

  return (
    <div className={`bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700/60 rounded-xl p-5 flex flex-col relative overflow-hidden transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md ${config.borderGlow} group`}>
      {/* Subtle Premium Top-Down Glassmorphism Gradient Glow */}
      <div className={`absolute inset-0 bg-gradient-to-b ${config.bgGlow} opacity-60 pointer-events-none transition-opacity duration-300 group-hover:opacity-100`} />

      <div className="flex items-center justify-between z-10">
        <h3 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-widest leading-tight">
          {title}
        </h3>
        
        {/* Curated Status Pill Badge replacing simple generic color dots */}
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium tracking-wide border ${config.badgeBg} ${config.badgeText} ${config.badgeBorder} shadow-sm backdrop-blur-sm transition-transform duration-200 group-hover:scale-105`}>
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${config.dotColor}`} />
          <span>{config.statusLabel}</span>
        </div>
      </div>

      <div className="mt-4 flex items-baseline gap-2 z-10">
        <div className={`text-4xl font-light tracking-tight ${config.valueColor} drop-shadow-sm transition-colors duration-200`}>
          {safeValue}
        </div>
        {subtitle && (
          <span className="text-xs font-medium text-slate-400 dark:text-slate-500 tracking-wide">
            {subtitle}
          </span>
        )}
      </div>

      {/* Decorative Bottom Corner Sleek Geometric Overlay Cues */}
      <div className="absolute right-0 bottom-0 translate-x-2 translate-y-2 w-12 h-12 bg-slate-50 dark:bg-slate-700/10 rounded-tl-3xl pointer-events-none transition-all duration-300 group-hover:scale-110" />
    </div>
  );
});

export default MetricCard;
