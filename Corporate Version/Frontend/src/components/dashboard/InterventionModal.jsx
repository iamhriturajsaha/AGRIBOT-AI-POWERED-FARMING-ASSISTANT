import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, Crosshair, Plane, CheckCircle2, AlertTriangle, X } from 'lucide-react';
import { useState, useEffect } from 'react';

export default function InterventionModal({ isOpen, onClose }) {
  const [phase, setPhase] = useState(0); 

  useEffect(() => {
    if (!isOpen) {
      setPhase(0);
      return;
    }
    
    // Simulate high-priority systemic task
    const timers = [
       setTimeout(() => setPhase(1), 2000), // Target Acquired
       setTimeout(() => setPhase(2), 5000), // Deploying Payload
       setTimeout(() => setPhase(3), 8500), // Mission Success
    ];
    return () => timers.forEach(clearTimeout);
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div 
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/50 backdrop-blur-sm p-4 lg:p-0"
      >
        <motion.div 
           initial={{ scale: 0.95, y: 20 }} 
           animate={{ scale: 1, y: 0 }}
           className="w-full max-w-4xl bg-slate-900/40 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl relative flex flex-col"
        >
           {/* Top bar */}
           <div className="bg-red-50 border-b border-red-100 p-4 px-6 flex items-center justify-between">
              <div className="flex items-center text-red-600 font-bold tracking-wide">
                 <ShieldAlert className="w-6 h-6 mr-3" /> EMERGENCY PROTOCOL ACTIVE
              </div>
              <button 
                onClick={onClose} 
                className="text-slate-700 hover:text-slate-900 bg-white/50 border border-slate-300 rounded-lg p-2 hover:bg-slate-100 transition-colors flex items-center shadow-sm font-semibold"
              >
                <X className="w-5 h-5 mr-2" />
                {phase === 3 ? "CLOSE" : "ABORT"}
              </button>
           </div>
           
           <div className="h-[500px] relative flex flex-col items-center justify-center overflow-hidden bg-slate-800/50">
               {/* Map Background Grid */}
               <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'linear-gradient(rgba(15, 23, 42, 0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(15, 23, 42, 0.2) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />

               {/* Cinematic Sequencing */}
               {phase === 0 && (
                  <motion.div initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 1.1, opacity: 0 }} className="text-center z-20">
                     <AlertTriangle className="w-24 h-24 text-red-500 mx-auto mb-6 animate-pulse" />
                     <h2 className="text-3xl font-bold text-white uppercase tracking-wide">Authorizing Override...</h2>
                     <p className="text-slate-400 mt-4 text-lg font-medium">Uplinking to Fleet Command</p>
                  </motion.div>
               )}

               {(phase === 1 || phase === 2) && (
                 <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute inset-0 flex items-center justify-center z-20">
                    {/* Fake Radar Sweep */}
                    <motion.div 
                      animate={{ rotate: 360 }} 
                      transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                      className="absolute w-[400px] h-[400px] rounded-full border border-slate-700"
                      style={{ borderTopColor: '#3b82f6', borderWidth: '3px' }}
                    />
                    
                    {/* The Target Area */}
                    <div className="absolute flex flex-col items-center justify-center">
                       <Crosshair className={`w-48 h-48 ${phase === 1 ? 'text-red-500 animate-pulse' : 'text-blue-500'} transition-colors duration-1000`} />
                       <div className="absolute top-full mt-6 text-center font-semibold tracking-wide text-lg bg-slate-900/40/90 p-4 rounded-xl shadow-sm border border-slate-800">
                          {phase === 1 ? (
                            <span className="text-red-600">TARGET ACQUIRED<br/><span className="text-sm font-normal text-slate-400">Coordinates: 34.0522° N, 118.2437° W</span></span>
                          ) : (
                            <span className="text-blue-600">DEPLOYING COUNTERMEASURES<br/><span className="text-sm font-normal text-slate-400">Distance: 0.00m</span></span>
                          )}
                       </div>
                    </div>

                    {/* The Drone Flying In */}
                    {phase === 2 && (
                       <motion.div 
                         initial={{ x: -300, y: -300, scale: 0 }} 
                         animate={{ x: 0, y: 0, scale: 1 }} 
                         transition={{ duration: 1.5, ease: "easeOut" }}
                         className="absolute"
                       >
                          <Plane className="w-16 h-16 text-slate-200 drop-shadow-lg rotate-45" />
                       </motion.div>
                    )}
                 </motion.div>
               )}

               {phase === 3 && (
                 <motion.div 
                   initial={{ scale: 0.8, opacity: 0 }} 
                   animate={{ scale: 1, opacity: 1 }} 
                   className="text-center z-20 bg-slate-900/40 p-12 rounded-3xl shadow-xl border border-slate-100"
                 >
                    <CheckCircle2 className="w-32 h-32 text-agri-green mx-auto mb-6" />
                    <h2 className="text-4xl font-bold text-white tracking-tight mb-4">Intervention Complete</h2>
                    <p className="text-agri-green mt-2 font-medium text-xl">Pathogen Risk Neutralized.</p>
                 </motion.div>
               )}
           </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
