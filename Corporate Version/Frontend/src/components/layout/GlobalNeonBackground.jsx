import { motion } from 'framer-motion';

export function GlobalNeonBackground({ children }) {
  return (
    <div className="relative w-full min-h-[100dvh] bg-slate-950 font-sans font-normal overflow-y-auto overflow-x-hidden">
      {/* Cinematic Bright Neon Green Background Glows */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        <motion.div 
          animate={{ scale: [1, 1.2, 1], opacity: [0.6, 1, 0.6] }} 
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-[-5%] right-[-5%] w-[600px] h-[600px] bg-green-500 rounded-full blur-[100px] opacity-80" 
        />
        <motion.div 
          animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0.9, 0.5] }} 
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 2 }}
          className="absolute bottom-[5%] left-[-10%] w-[500px] h-[500px] bg-emerald-500 rounded-full blur-[100px] opacity-70" 
        />
        <motion.div 
          animate={{ x: [0, 50, 0], y: [0, -50, 0], opacity: [0.4, 0.8, 0.4] }} 
          transition={{ duration: 12, repeat: Infinity, ease: "easeInOut", delay: 4 }}
          className="absolute top-[30%] left-[20%] w-[400px] h-[400px] bg-lime-400 rounded-full blur-[90px] opacity-60" 
        />
        <motion.div 
          animate={{ scale: [1, 1.5, 1], opacity: [0.3, 0.7, 0.3] }} 
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut", delay: 6 }}
          className="absolute top-[60%] right-[10%] w-[350px] h-[350px] bg-teal-500 rounded-full blur-[100px] opacity-60" 
        />
        <motion.div 
          animate={{ x: [0, -40, 0], y: [0, 60, 0], opacity: [0.5, 0.9, 0.5] }} 
          transition={{ duration: 9, repeat: Infinity, ease: "easeInOut", delay: 1 }}
          className="absolute bottom-[-15%] right-[20%] w-[450px] h-[450px] bg-agri-green rounded-full blur-[120px] opacity-70" 
        />
      </div>

      {/* Main Content Layer (Above Background) */}
      <div className="relative z-10 w-full min-h-screen">
        {children}
      </div>
    </div>
  );
}
