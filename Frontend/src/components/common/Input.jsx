import { forwardRef } from 'react';
import { cn } from './Button';

export const Input = forwardRef(({ className, label, error, ...props }, ref) => {
  return (
    <div className="w-full">
      {label && <label className="block text-sm font-medium text-gray-300 mb-1.5">{label}</label>}
      <input
        ref={ref}
        className={cn(
          "flex h-11 w-full rounded-xl border border-panelBorder bg-white/5 px-3 py-2 text-sm text-white transition-all",
          "file:border-0 file:bg-transparent file:text-sm file:font-medium",
          "placeholder:text-gray-500",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neon-green/50 focus-visible:border-neon-green/50",
          "disabled:cursor-not-allowed disabled:opacity-50",
          error && "border-red-500/50 focus-visible:ring-red-500/50 focus-visible:border-red-500/50",
          className
        )}
        {...props}
      />
      {error && <p className="mt-1.5 text-sm text-red-500">{error}</p>}
    </div>
  );
});

Input.displayName = 'Input';
