import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, Crosshair, Plane, CheckCircle2, AlertTriangle } from 'lucide-react';
import { useState, useEffect } from 'react';

export default function InterventionModal({ isOpen, onClose }) {
  const [phase, setPhase] = useState(0); 

  useEffect(() => {
    if (!isOpen) {
      setPhase(0);
      return;
    }
    
    // Simulate high-tension Drone Strike Sequence
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
        className="fixed inset-0 z-[100] flex items-center justify-center bg-black/95 backdrop-blur-md p-4 lg:p-0"
      >
        <motion.div 
           initial={{ scale: 0.9, y: 50, rotateX: 20 }} 
           animate={{ scale: 1, y: 0, rotateX: 0 }}
           className="w-full max-w-5xl bg-gray-950 border border-red-500/50 rounded-3xl overflow-hidden shadow-[0_0_100px_rgba(255,0,0,0.15)] relative"
           style={{ transformPerspective: 1000 }}
        >
           {/* Top bar */}
           <div className="bg-red-950/80 border-b border-red-500/30 p-4 px-6 flex items-center justify-between">
              <div className="flex items-center text-red-500 animate-pulse font-bold tracking-widest text-lg">
                 <ShieldAlert className="w-6 h-6 mr-3" /> AUTONOMOUS INTERVENTION OVERRIDE
              </div>
              <button 
                onClick={onClose} 
                className="text-gray-400 font-bold tracking-widest hover:text-white px-6 py-2 border border-red-500/50 rounded hover:bg-red-500/20 transition-colors"
              >
                {phase === 3 ? "CLOSE PROTOCOL" : "ABORT STRIKE"}
              </button>
           </div>
           
           <div className="h-[600px] relative flex flex-col items-center justify-center overflow-hidden">
               {/* Map Background Grid */}
               <div className="absolute inset-0 opacity-20 pointer-events-none" style={{ backgroundImage: 'linear-gradient(rgba(255,0,0,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(255,0,0,0.2) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
               <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_transparent_0%,_#000_100%)] pointer-events-none z-10" />

               {/* Cinematic Sequencing */}
               {phase === 0 && (
                  <motion.div initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 1.5, opacity: 0 }} className="text-center z-20">
                     <AlertTriangle className="w-32 h-32 text-red-500 mx-auto mb-6 animate-ping" />
                     <h2 className="text-4xl font-black text-white uppercase tracking-widest drop-shadow-[0_0_15px_#f00]">Authenticating...</h2>
                     <p className="text-red-400 mt-4 font-mono text-xl">Uplinking to Fleet Command</p>
                  </motion.div>
               )}

               {(phase === 1 || phase === 2) && (
                 <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute inset-0 flex items-center justify-center z-20">
                    {/* Fake Radar Sweep */}
                    <motion.div 
                      animate={{ rotate: 360 }} 
                      transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                      className="absolute w-[500px] h-[500px] rounded-full border border-red-500/30"
                      style={{ borderTopColor: 'rgba(255,0,0,1)' }}
                    />
                    
                    {/* The Target Area */}
                    <div className="absolute flex flex-col items-center justify-center">
                       <Crosshair className={`w-64 h-64 ${phase === 1 ? 'text-red-500 animate-pulse' : 'text-[#0ff]'} transition-colors duration-1000`} />
                       <div className="absolute top-full mt-4 text-center font-mono font-bold tracking-widest text-lg">
                          {phase === 1 ? (
                            <span className="text-red-500">TARGET LOCK ACQUIRED<br/>Coordinates: 34.0522° N, 118.2437° W</span>
                          ) : (
                            <span className="text-[#0ff]">DEPLOYING FUNGICIDE PAYLOAD<br/>Distance: 0.00m</span>
                          )}
                       </div>
                    </div>

                    {/* The Drone Flying In */}
                    {phase === 2 && (
                       <motion.div 
                         initial={{ x: -400, y: -400, scale: 0 }} 
                         animate={{ x: 0, y: 0, scale: 1 }} 
                         transition={{ duration: 1.5, ease: "easeOut" }}
                         className="absolute"
                       >
                          <Plane className="w-24 h-24 text-white drop-shadow-[0_0_20px_#0ff] rotate-45" />
                       </motion.div>
                    )}
                 </motion.div>
               )}

               {phase === 3 && (
                 <motion.div 
                   initial={{ scale: 0.5, opacity: 0 }} 
                   animate={{ scale: 1, opacity: 1 }} 
                   className="text-center z-20"
                 >
                    <CheckCircle2 className="w-40 h-40 text-neon-green mx-auto mb-8 drop-shadow-[0_0_30px_rgba(0,255,65,0.8)]" />
                    <h2 className="text-5xl font-black text-white uppercase tracking-widest mb-4">Intervention Complete</h2>
                    <p className="text-neon-green mt-2 font-mono text-2xl">Pathogen Spread Neutralized.</p>
                 </motion.div>
               )}
           </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
