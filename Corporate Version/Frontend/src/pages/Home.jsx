import { motion } from 'framer-motion';
import { Card, CardContent } from '../components/common/Card';
import { Leaf, Navigation, Cpu, CloudRain, ChevronRight, Activity, Sprout } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Home() {
  const features = [
    {
      title: "Advanced Crop Diagnostics",
      desc: "Upload imagery from your fields. Our machine learning models quickly analyze visual data to identify plant health issues, nutrient deficiencies, and potential diseases.",
      icon: Activity,
      color: "from-agri-lightGreen to-agri-green"
    },
    {
      title: "Geospatial Monitoring",
      desc: "Visualize your farm's performance across different sectors with integrated satellite vegetation indices and soil moisture mappings.",
      icon: Navigation,
      color: "from-green-500 to-emerald-700"
    },
    {
      title: "Environmental Analytics",
      desc: "Comprehensive weather modeling utilizing real-time meteorological data to predict optimal planting, irrigation, and harvesting conditions.",
      icon: CloudRain,
      color: "from-blue-500 to-indigo-600"
    },
    {
      title: "Intelligent Assistant",
      desc: "An integrated AI consultant capable of parsing complex agricultural data to provide actionable advice on crop management strategies.",
      icon: Cpu,
      color: "from-amber-500 to-orange-600"
    }
  ];

  return (
    <div className="relative w-full h-screen font-sans overflow-hidden flex flex-col justify-center">
      {/* Animated Growing Neon Crops (Background Layer - Dual placement) */}
      <div className="absolute inset-0 pointer-events-none z-0">
        
        {/* Left Peripheral Crop */}
        <motion.div
           initial={{ scaleY: 0, opacity: 0 }}
           animate={{ scaleY: 1, opacity: 0.5 }}
           transition={{ duration: 4, ease: "easeOut", delay: 0.3 }}
           className="absolute bottom-0 left-0 lg:left-16 origin-bottom -scale-x-100"
           style={{ height: "65vh" }}
        >
          <svg viewBox="0 0 200 600" className="w-[200px] md:w-[350px] h-full drop-shadow-[0_0_40px_rgba(34,197,94,0.6)]" preserveAspectRatio="xMidYMax meet">
             <defs>
               <linearGradient id="stemGradient" x1="0" y1="1" x2="0" y2="0">
                 <stop offset="0%" stopColor="#052e16" />
                 <stop offset="30%" stopColor="#166534" />
                 <stop offset="70%" stopColor="#22c55e" />
                 <stop offset="100%" stopColor="#86efac" />
               </linearGradient>
             </defs>
             <path d="M100 600 C95 450 105 250 100 50" fill="none" stroke="url(#stemGradient)" strokeWidth="6" strokeLinecap="round" />
             <path d="M100 500 Q20 450 40 320 Q60 410 100 480" fill="rgba(34, 197, 94, 0.4)" stroke="#4ade80" strokeWidth="2" />
             <path d="M100 450 Q180 400 160 270 Q140 360 100 430" fill="rgba(34, 197, 94, 0.4)" stroke="#4ade80" strokeWidth="2" />
             <path d="M100 350 Q30 300 50 180 Q70 260 100 330" fill="rgba(34, 197, 94, 0.5)" stroke="#86efac" strokeWidth="2" />
             <path d="M100 300 Q170 250 150 130 Q130 210 100 280" fill="rgba(34, 197, 94, 0.5)" stroke="#86efac" strokeWidth="2" />
             <path d="M100 50 Q80 10 100 0 Q120 10 100 50" fill="rgba(167, 243, 208, 0.8)" stroke="#bef264" strokeWidth="3" />
             <circle cx="100" cy="50" r="8" fill="rgba(217, 249, 157, 0.8)" />
          </svg>
        </motion.div>

        {/* Right Peripheral Crop */}
        <motion.div
           initial={{ scaleY: 0, opacity: 0 }}
           animate={{ scaleY: 1, opacity: 0.5 }}
           transition={{ duration: 4, ease: "easeOut", delay: 0.5 }}
           className="absolute bottom-0 right-0 lg:right-16 origin-bottom"
           style={{ height: "65vh" }}
        >
          <svg viewBox="0 0 200 600" className="w-[200px] md:w-[350px] h-full drop-shadow-[0_0_40px_rgba(34,197,94,0.6)]" preserveAspectRatio="xMidYMax meet">
             <path d="M100 600 C95 450 105 250 100 50" fill="none" stroke="url(#stemGradient)" strokeWidth="6" strokeLinecap="round" />
             <path d="M100 500 Q20 450 40 320 Q60 410 100 480" fill="rgba(34, 197, 94, 0.4)" stroke="#4ade80" strokeWidth="2" />
             <path d="M100 450 Q180 400 160 270 Q140 360 100 430" fill="rgba(34, 197, 94, 0.4)" stroke="#4ade80" strokeWidth="2" />
             <path d="M100 350 Q30 300 50 180 Q70 260 100 330" fill="rgba(34, 197, 94, 0.5)" stroke="#86efac" strokeWidth="2" />
             <path d="M100 300 Q170 250 150 130 Q130 210 100 280" fill="rgba(34, 197, 94, 0.5)" stroke="#86efac" strokeWidth="2" />
             <path d="M100 50 Q80 10 100 0 Q120 10 100 50" fill="rgba(167, 243, 208, 0.8)" stroke="#bef264" strokeWidth="3" />
             <circle cx="100" cy="50" r="8" fill="rgba(217, 249, 157, 0.8)" />
          </svg>
        </motion.div>

      </div>
      <div className="relative z-10 w-full max-w-7xl mx-auto px-6 xl:px-12 py-8 mt-16">
        <div className="flex flex-col lg:flex-row items-center gap-8 lg:gap-16">
          
          {/* Left Column: Title & Overview */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="flex-1 text-center lg:text-left"
          >
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-agri-green/10 text-agri-green font-semibold tracking-wide text-sm mb-8">
              <Sprout className="w-5 h-5 mr-2" />
              Enterprise Farm Management V.1.0
            </div>

            <h1 className="text-5xl lg:text-7xl font-bold text-white tracking-tight mb-6 leading-tight">
              Smarter Agriculture with <span className="text-transparent bg-clip-text bg-gradient-to-r from-agri-green to-emerald-400">AgriBot</span>
            </h1>

            <p className="text-base lg:text-lg text-slate-300 mb-8 leading-relaxed max-w-2xl mx-auto lg:mx-0">
              Transform your agricultural operations with data-driven insights. AgriBot combines advanced AI diagnostics, real-time satellite telemetry, and environmental analytics into a unified command center.
            </p>

            <Link to="/dashboard">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="group relative inline-flex items-center justify-center px-8 py-4 font-semibold text-white bg-agri-green hover:bg-agri-green/90 rounded-xl overflow-hidden shadow-sm transition-colors"
                type="button"
              >
                <span className="relative flex items-center z-10 text-lg">
                  Access Platform <ChevronRight className="ml-2 w-5 h-5 transition-transform group-hover:translate-x-1" />
                </span>
              </motion.button>
            </Link>
          </motion.div>

           {/* Right Column: Key Metrics / Features Image placeholder or modern geometric art */}
          <motion.div
             initial={{ opacity: 0, lg: 30 }}
             animate={{ opacity: 1, lg: 0 }}
             transition={{ duration: 0.6, delay: 0.2 }}
             className="flex-1 w-full relative hidden lg:block"
          >
             <div className="relative w-full aspect-square rounded-3xl bg-slate-900/40 backdrop-blur-3xl border border-slate-700 shadow-[0_0_50px_rgba(34,197,94,0.1)] p-6 lg:p-8 overflow-hidden flex items-center justify-center">
                {/* Decorative Agricultural Animations Grid */}
                <div className="grid grid-cols-2 gap-4 w-full h-full">
                   
                   {/* Box 1: Smart AI Core */}
                   <div className="bg-slate-950/80 rounded-2xl shadow-md border border-slate-800 flex flex-col items-center justify-center relative overflow-hidden group">
                     <motion.div 
                       animate={{ rotate: 360 }} 
                       transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                       className="absolute inset-0 w-full h-full border-[10px] border-dashed border-agri-green/30 rounded-full scale-[1.5]"
                     />
                     <Cpu className="w-10 h-10 text-slate-200 z-10" />
                     <span className="text-[10px] font-bold text-slate-400 mt-2 z-10 tracking-widest uppercase">AI Core</span>
                   </div>

                   {/* Box 2: Health Monitor */}
                   <div className="bg-gradient-to-br from-slate-900 to-agri-green/10 rounded-2xl border border-agri-green/30 flex flex-col items-center justify-center shadow-inner">
                     <motion.div 
                       animate={{ y: [0, -10, 0] }} 
                       transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                     >
                       <Leaf className="w-12 h-12 text-agri-green fill-agri-green/40" />
                     </motion.div>
                     <span className="text-[10px] font-bold text-agri-green mt-2 tracking-widest uppercase">Optimal Crop</span>
                   </div>

                   {/* Box 3: Weather Tracking */}
                   <div className="bg-gradient-to-br from-slate-900 to-blue-900/30 rounded-2xl border border-blue-500/30 flex flex-col items-center justify-center shadow-inner relative overflow-hidden">
                     <motion.div
                       animate={{ y: [-20, 20], opacity: [0, 1, 0] }}
                       transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                       className="absolute inset-0 flex items-center justify-center"
                     >
                        <div className="w-0.5 h-full bg-blue-400/50 rotate-45 transform translate-x-4"></div>
                        <div className="w-0.5 h-full bg-blue-400/50 rotate-45 transform -translate-x-4"></div>
                     </motion.div>
                     <CloudRain className="w-12 h-12 text-blue-400 z-10 fill-blue-500/20" />
                     <span className="text-[10px] font-bold text-blue-400 mt-2 z-10 tracking-widest uppercase">Climate Sync</span>
                   </div>

                   {/* Box 4: Telemetry Pulse */}
                   <div className="bg-slate-950/80 rounded-2xl shadow-md border border-slate-800 flex flex-col items-center justify-center relative overflow-hidden">
                     <motion.div 
                       animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0] }} 
                       transition={{ duration: 2, repeat: Infinity, ease: "easeOut" }}
                       className="absolute w-20 h-20 bg-amber-400 rounded-full"
                     />
                     <Activity className="w-10 h-10 text-amber-500 z-10" />
                     <span className="text-[10px] font-bold text-slate-400 mt-2 z-10 tracking-widest uppercase">Telemetry</span>
                   </div>
                   
                </div>
             </div>
          </motion.div>
        </div>

        {/* Bottom Feature Grid */}
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-8 lg:mt-12"
        >
          {features.map((feature, i) => (
            <Card key={i} className="!bg-slate-900/40 !backdrop-blur-3xl !border-slate-800 shadow-xl hover:shadow-2xl transition-shadow">
              <CardContent className="p-6 md:p-8 flex flex-col h-full relative z-10">
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(34,197,94,0.3)]`}>
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed font-medium">
                  {feature.desc}
                </p>
              </CardContent>
            </Card>
          ))}
        </motion.div>
      </div>
    </div>
  );
}
