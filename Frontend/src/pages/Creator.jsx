import { motion } from 'framer-motion';
import { Github, Linkedin, Mail, Terminal, Fingerprint, Code2, Cpu, Database, Zap, ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function Creator() {
  const navigate = useNavigate();

  return (
    <div className="relative flex-1 w-full flex items-stretch p-4 xl:p-8 overflow-hidden font-mono bg-[#09090b]">
       
      {/* Cyberpunk Grid Background */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(0,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none" />
      <div className="absolute inset-x-0 bottom-0 h-[40vh] bg-gradient-to-t from-[#f0f]/10 to-transparent pointer-events-none blur-3xl opacity-50" />
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-[#0ff]/10 rounded-full blur-[120px] pointer-events-none" />

      {/* Main Terminal Card */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="relative z-10 w-full h-full flex flex-col border-2 border-[#0ff]/50 bg-black/80 backdrop-blur-xl shadow-[0_0_50px_rgba(0,255,255,0.15)]"
      >
        {/* Terminal Header */}
        <div className="flex border-b-2 border-[#0ff]/50 bg-[#0ff]/10 px-4 py-3 items-center justify-between">
           <div className="flex space-x-2">
             <div className="w-3 h-3 bg-red-500 rounded-sm animate-pulse" />
             <div className="w-3 h-3 bg-yellow-500 rounded-sm" />
             <div className="w-3 h-3 bg-green-500 rounded-sm" />
           </div>
           <span className="text-[#0ff] text-xs font-bold tracking-widest uppercase">SYS.ADMIN // SYSTEM_OVERRIDE_ACTIVE</span>
        </div>

        <div className="p-8 md:p-12 relative flex-1 flex flex-col justify-center overflow-hidden">
          {/* Cyberpunk Decorative Elements */}
          <div className="absolute top-8 right-8 text-[#0ff]/20 text-9xl font-bold opacity-10 pointer-events-none tracking-tighter">
            DEV
          </div>
          <div className="absolute bottom-0 left-0 w-2 h-full bg-gradient-to-b from-[#f0f] via-[#0ff] to-transparent shadow-[0_0_20px_#0ff]" />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center relative z-10">
            
            {/* Left Col: Identity Matrix */}
            <div>
              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                className="mb-8"
              >
                <div className="flex items-center text-[#f0f] mb-2 text-sm font-bold tracking-widest uppercase">
                  <Fingerprint className="w-4 h-4 mr-2" />
                  Authorized Architect
                </div>
                <h1 className="text-5xl md:text-7xl font-black text-transparent bg-clip-text bg-gradient-to-r from-[#0ff] via-white to-[#f0f] tracking-tighter uppercase glitch-text leading-none mb-2 relative">
                  HRITURAJ SAHA
                  {/* Glitch sub-layer */}
                  <span className="absolute inset-0 translate-x-[2px] translate-y-[-2px] text-[#0ff] opacity-50 mix-blend-screen pointer-events-none" aria-hidden="true">HRITURAJ SAHA</span>
                  <span className="absolute inset-0 translate-x-[-2px] translate-y-[2px] text-[#f0f] opacity-50 mix-blend-screen pointer-events-none" aria-hidden="true">HRITURAJ SAHA</span>
                </h1>
                <p className="text-[#0ff]/70 text-lg md:text-xl font-medium mt-4 border-l-2 border-[#0ff] pl-4 py-1">
                  &gt; Full Stack Engineer & Machine Learning Integrator
                </p>
              </motion.div>

              <div className="space-y-4">
                <motion.a 
                  href="https://mail.google.com/mail/?view=cm&fs=1&to=iamhriturajsaha@gmail.com"
                  target="_blank" rel="noreferrer"
                  whileHover={{ scale: 1.02, x: 10 }}
                  className="flex items-center p-4 bg-[#0ff]/5 border border-[#0ff]/20 hover:border-[#0ff] hover:bg-[#0ff]/10 hover:shadow-[0_0_20px_rgba(0,255,255,0.4)] transition-all group relative overflow-hidden"
                >
                  <div className="absolute inset-y-0 left-0 w-1 bg-[#0ff] group-hover:w-full transition-all duration-300 opacity-20" />
                  <Mail className="w-6 h-6 text-[#0ff] mr-4 relative z-10" />
                  <span className="text-white font-medium tracking-wide relative z-10">iamhriturajsaha@gmail.com</span>
                </motion.a>

                <motion.a 
                  href="https://github.com/iamhriturajsaha" 
                  target="_blank" rel="noreferrer"
                  whileHover={{ scale: 1.02, x: 10 }}
                  className="flex items-center p-4 bg-[#f0f]/5 border border-[#f0f]/20 hover:border-[#f0f] hover:bg-[#f0f]/10 hover:shadow-[0_0_20px_rgba(255,0,255,0.4)] transition-all group relative overflow-hidden"
                >
                  <div className="absolute inset-y-0 left-0 w-1 bg-[#f0f] group-hover:w-full transition-all duration-300 opacity-20" />
                  <Github className="w-6 h-6 text-[#f0f] mr-4 relative z-10" />
                  <span className="text-white font-medium tracking-wide relative z-10">github.com/iamhriturajsaha</span>
                  <Terminal className="w-4 h-4 ml-auto text-[#f0f]/50 group-hover:opacity-100 opacity-0 transition-opacity relative z-10" />
                </motion.a>

                <motion.a 
                  href="https://www.linkedin.com/in/hrituraj-saha-5794b53a0" 
                  target="_blank" rel="noreferrer"
                  whileHover={{ scale: 1.02, x: 10 }}
                  className="flex items-center p-4 bg-[#0ff]/5 border border-[#0ff]/20 hover:border-[#0ff] hover:bg-[#0ff]/10 hover:shadow-[0_0_20px_rgba(0,255,255,0.4)] transition-all group relative overflow-hidden"
                >
                  <div className="absolute inset-y-0 left-0 w-1 bg-[#0ff] group-hover:w-full transition-all duration-300 opacity-20" />
                  <Linkedin className="w-6 h-6 text-[#0ff] mr-4 relative z-10" />
                  <span className="text-white font-medium tracking-wide relative z-10 text-sm md:text-base truncate">linkedin.com/in/hrituraj-saha</span>
                  <Terminal className="w-4 h-4 ml-auto text-[#0ff]/50 group-hover:opacity-100 opacity-0 transition-opacity relative z-10" />
                </motion.a>
              </div>
            </div>

            {/* Right Col: Cyberpunk Stats / "Implant" Matrix */}
            <motion.div 
               initial={{ opacity: 0, lg: 20 }}
               animate={{ opacity: 1, y: 0 }}
               transition={{ delay: 0.4 }}
               className="bg-[#0f0f13] border-2 border-dashed border-[#0ff]/20 p-6 relative"
            >
               <div className="absolute -top-3 -right-3 w-6 h-6 bg-[#0ff] flex items-center justify-center">
                 <Zap className="w-4 h-4 text-black animate-pulse" />
               </div>
               
               <h2 className="text-[#f0f] text-sm uppercase tracking-widest font-bold border-b border-[#f0f]/30 pb-3 mb-6 flex items-center">
                 <Code2 className="w-5 h-5 mr-3" /> Technical Implants & Loadout
               </h2>

               <div className="space-y-6">
                 <div>
                   <div className="flex justify-between text-[#0ff] text-xs font-bold mb-1 uppercase">
                     <span>Neural Network (AI)</span>
                     <span>98.5%</span>
                   </div>
                   <div className="h-2 bg-black border border-[#0ff]/30 w-full overflow-hidden">
                     <div className="h-full bg-[#0ff] w-[98.5%] shadow-[0_0_10px_#0ff]" />
                   </div>
                 </div>

                 <div>
                   <div className="flex justify-between text-[#f0f] text-xs font-bold mb-1 uppercase">
                     <span>Backend Syntax (Django/Python)</span>
                     <span>95.0%</span>
                   </div>
                   <div className="h-2 bg-black border border-[#f0f]/30 w-full overflow-hidden">
                     <div className="h-full bg-[#f0f] w-[95%] shadow-[0_0_10px_#f0f]" />
                   </div>
                 </div>

                 <div>
                   <div className="flex justify-between text-yellow-400 text-xs font-bold mb-1 uppercase">
                     <span>Frontend Cybernetics (React.js)</span>
                     <span>100%</span>
                   </div>
                   <div className="h-2 bg-black border border-yellow-400/30 w-full overflow-hidden">
                     <div className="h-full bg-yellow-400 w-full shadow-[0_0_10px_#facc15]" />
                   </div>
                 </div>

                 <div>
                   <div className="flex justify-between text-orange-500 text-xs font-bold mb-1 uppercase">
                     <span>Cloud Architecture (AWS / Docker)</span>
                     <span>92.3%</span>
                   </div>
                   <div className="h-2 bg-black border border-orange-500/30 w-full overflow-hidden">
                     <div className="h-full bg-orange-500 w-[92.3%] shadow-[0_0_10px_#f97316]" />
                   </div>
                 </div>

                 <div>
                   <div className="flex justify-between text-green-400 text-xs font-bold mb-1 uppercase">
                     <span>Data Vaults (PostgreSQL)</span>
                     <span>96.0%</span>
                   </div>
                   <div className="h-2 bg-black border border-green-400/30 w-full overflow-hidden">
                     <div className="h-full bg-green-400 w-[96%] shadow-[0_0_10px_#4ade80]" />
                   </div>
                 </div>
               </div>

               <div className="mt-8 grid grid-cols-2 gap-4">
                  <div className="bg-[#000] border border-[#0ff]/20 p-3 flex items-center justify-center filter drop-shadow-[0_0_5px_rgba(0,255,255,0.2)]">
                     <Cpu className="w-5 h-5 text-[#0ff] mr-2" />
                     <span className="text-[#0ff] text-xs font-bold tracking-widest">SYS.STABLE</span>
                  </div>
                  <div className="bg-[#000] border border-[#f0f]/20 p-3 flex items-center justify-center filter drop-shadow-[0_0_5px_rgba(255,0,255,0.2)]">
                     <Database className="w-5 h-5 text-[#f0f] mr-2" />
                     <span className="text-[#f0f] text-xs font-bold tracking-widest">DATA.SYNCED</span>
                  </div>
               </div>
            </motion.div>

          </div>

          {/* Return Button */}
          <div className="mt-12 lg:mt-16 flex justify-center relative z-20">
             <motion.button
                onClick={() => navigate(-1)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="group relative inline-flex items-center justify-center px-10 py-4 font-bold text-[#0ff] uppercase tracking-widest bg-black/50 border border-[#0ff] hover:bg-[#0ff] hover:text-black overflow-hidden shadow-[0_0_20px_rgba(0,255,255,0.2)] transition-colors duration-300 backdrop-blur-md"
             >
                <div className="absolute inset-0 w-full h-full bg-white/20 group-hover:translate-x-full transition-transform duration-500 ease-out -skew-x-12 -translate-x-full" />
                <span className="relative flex items-center z-10">
                   <ArrowLeft className="w-5 h-5 mr-3 transition-transform group-hover:-translate-x-2" /> 
                   Return to Operations
                </span>
             </motion.button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
