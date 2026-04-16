import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Terminal } from 'lucide-react';

export function BootSequence({ onComplete }) {
  const [logs, setLogs] = useState([]);
  
  const bootLogs = [
    "INITIATING AGRIBOT CORE OS V.1.0...",
    "ESTABLISHING SATELLITE UPLINK...",
    "UPLINK SECURED. PING: 14ms",
    "DECRYPTING FIELD TELEMETRY DATA...",
    "PARSING BIOMASS INDICES...",
    "AWAKENING NEURAL VISION ENGINE...",
    "CONNECTING TO GLOBAL FLEET DRONES...",
    "THREE.JS HARDWARE RENDERING [ONLINE]",
    "SYSTEM BOOT SUCCESSFUL. ENTERING DASHBOARD."
  ];

  useEffect(() => {
    let currentIndex = 0;
    
    const interval = setInterval(() => {
      if (currentIndex < bootLogs.length) {
        setLogs(prev => [...prev, bootLogs[currentIndex]]);
        currentIndex++;
      } else {
        clearInterval(interval);
        setTimeout(onComplete, 800); // 800ms flash before transition
      }
    }, 400); // Fast 400ms per log line
    
    return () => clearInterval(interval);
  }, [onComplete]);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 1.1, filter: "blur(10px)" }}
      transition={{ duration: 0.5 }}
      className="fixed inset-0 z-[100] bg-black flex flex-col justify-center px-8 md:px-32 font-mono"
    >
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-neon-green/10 via-black to-black opacity-40 pointer-events-none" />
      
      <div className="flex items-center text-neon-green mb-8 opacity-80">
        <Terminal className="w-8 h-8 mr-4 animate-pulse" />
        <span className="text-2xl font-bold tracking-widest uppercase">System Boot</span>
      </div>

      <div className="space-y-4">
        {logs.map((log, idx) => (
          <motion.div 
            key={idx}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className={`text-lg tracking-wider ${
              idx === bootLogs.length - 1 
                ? "text-cyan-400 font-bold text-xl drop-shadow-[0_0_10px_rgba(0,255,255,0.8)]" 
                : "text-green-500"
            }`}
          >
            <span className="opacity-50 mr-4">[{new Date().toISOString().substring(11, 23)}]</span>
            {log}
          </motion.div>
        ))}
        {logs.length < bootLogs.length && (
          <motion.div 
            animate={{ opacity: [1, 0, 1] }} 
            transition={{ repeat: Infinity, duration: 0.8 }}
            className="w-4 h-6 bg-green-500 ml-1 inline-block"
          />
        )}
      </div>
    </motion.div>
  );
}
