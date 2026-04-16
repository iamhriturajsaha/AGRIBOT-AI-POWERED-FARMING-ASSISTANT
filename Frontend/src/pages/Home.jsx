import { motion } from 'framer-motion';
import { Card, CardContent } from '../components/common/Card';
import { Leaf, Navigation, Cpu, CloudRain, ChevronRight, Activity, Gamepad2, Tractor, Sprout } from 'lucide-react';
import { Link } from 'react-router-dom';

const PixelSprout = ({ className }) => (
  <svg viewBox="0 0 24 24" className={className} fill="currentColor" style={{ shapeRendering: 'crispEdges' }}>
    {/* Pot */}
    <g fill="#94a3b8">
      <rect x="6" y="16" width="12" height="6" />
      <rect x="7" y="22" width="10" height="1" />
      <rect x="5" y="15" width="14" height="2" fill="#eab308" /> {/* Pot Rim */}
    </g>
    <rect x="8" y="18" width="8" height="1" fill="#cbd5e1" opacity="0.3" /> {/* Pot Detail */}

    {/* Soil */}
    <g fill="#451a03">
      <rect x="8" y="13" width="8" height="2" />
      <rect x="7" y="14" width="10" height="1" />
    </g>

    {/* Stem */}
    <g fill="#16a34a">
      <rect x="11" y="8" width="2" height="6" />
      <rect x="11" y="13" width="2" height="1" fill="#14532d" /> {/* Stem Shadow */}
    </g>

    {/* Right Leaf (Higher) */}
    <g fill="#22c55e">
      <rect x="13" y="4" width="4" height="4" />
      <rect x="14" y="3" width="4" height="4" />
      <rect x="15" y="2" width="2" height="1" fill="#4ade80" /> {/* Highlight */}
      <rect x="14" y="5" width="2" height="1" fill="#15803d" /> {/* Shadow */}
    </g>

    {/* Left Leaf (Lower) */}
    <g fill="#22c55e">
      <rect x="7" y="9" width="4" height="3" />
      <rect x="6" y="10" width="4" height="2" />
      <rect x="7" y="10" width="2" height="1" fill="#4ade80" /> {/* Highlight */}
      <rect x="9" y="11" width="2" height="1" fill="#15803d" /> {/* Shadow */}
    </g>
  </svg>
);

const PixelTractor = ({ className }) => (
  <svg viewBox="0 0 40 24" className={className} fill="currentColor" style={{ shapeRendering: 'crispEdges' }}>
    {/* Rear Body / Hitch */}
    <rect x="5" y="14" width="4" height="4" fill="#14532d" />
    <rect x="5" y="15" width="2" height="2" fill="#064e3b" />

    {/* Main Green Body */}
    <g fill="#15803d">
      <rect x="9" y="10" width="18" height="8" />
      <rect x="27" y="11" width="6" height="6" />
      <rect x="33" y="13" width="3" height="3" />
    </g>
    {/* Highlights and Texture */}
    <g fill="#22c55e">
      <rect x="10" y="11" width="16" height="2" />
      <rect x="27" y="12" width="5" height="1" />
      <rect x="20" y="13" width="12" height="2" />
      <rect x="27" y="11" width="3" height="1" fill="#4ade80" />
    </g>
    {/* Yellow Detail Line */}
    <rect x="26" y="12" width="6" height="1" fill="#eab308" />

    {/* Cabin Structure */}
    <rect x="12" y="5" width="12" height="1" fill="#064e3b" /> {/* Roof */}
    <g fill="#1e293b"> {/* Pillars */}
      <rect x="12" y="6" width="1" height="5" />
      <rect x="23" y="6" width="1" height="5" />
      <rect x="18" y="6" width="1" height="5" />
    </g>
    
    {/* Glass Sections */}
    <g fill="#cbd5e1" opacity="0.5">
      <rect x="13" y="6" width="5" height="4" />
      <rect x="19" y="6" width="4" height="4" />
    </g>
    <rect x="13" y="7" width="3" height="1" fill="#f8fafc" opacity="0.8" /> {/* Glint */}

    {/* Steps (Emerald Green) */}
    <g fill="#15803d">
      <rect x="17" y="16" width="1" height="5" />
      <rect x="20" y="16" width="1" height="5" />
      <rect x="17" y="18" width="4" height="1" />
      <rect x="17" y="20" width="4" height="1" />
    </g>

    {/* Back Wheel (Large 4-Tier) */}
    <g>
      <rect x="3" y="10" width="12" height="12" fill="#1e293b" /> {/* Tire */}
      <rect x="4" y="11" width="10" height="10" fill="#451a03" /> {/* Inner Brown */}
      <rect x="5" y="12" width="8" height="8" fill="#eab308" />   {/* Golden Rim */}
      <rect x="7" y="14" width="4" height="4" fill="#facc15" />   {/* Neon Center */}
    </g>

    {/* Front Wheel (Small 4-Tier) */}
    <g>
      <rect x="26" y="15" width="8" height="8" fill="#1e293b" />  {/* Tire */}
      <rect x="27" y="16" width="6" height="6" fill="#451a03" />  {/* Inner Brown */}
      <rect x="28" y="17" width="4" height="4" fill="#eab308" />  {/* Golden Rim */}
      <rect x="29" y="18" width="2" height="2" fill="#facc15" />  {/* Neon Center */}
    </g>
  </svg>
);

export default function Home() {
  const features = [
    {
      title: "Neural Vision Diagnostics",
      desc: "Upload drone captures. Our CNN instantly rips through cellular data to detect aggressive pathogen outbreaks in real-time.",
      icon: Activity,
      color: "from-pink-500 to-rose-400"
    },
    {
      title: "Autonomous Fleet Tracking",
      desc: "Monitor airborne scout and sprayer drones across sectors with live cybernetic telemetry.",
      icon: Navigation,
      color: "from-cyan-400 to-blue-500"
    },
    {
      title: "Bio-Threat Environment Matrix",
      desc: "Live Open-Meteo satellite feed scanning for massive UV indices and fungal-spawning humidity spikes.",
      icon: CloudRain,
      color: "from-purple-500 to-indigo-500"
    },
    {
      title: "AI Voice Copilot",
      desc: "An integrated generative LLM that remembers your images and gives lethal precision advice on crop salvation.",
      icon: Cpu,
      color: "from-orange-400 to-pink-500"
    }
  ];

  return (
    <div className="relative w-full h-[100dvh] max-h-[100dvh] overflow-hidden flex flex-col items-center justify-center px-6 xl:px-12 pt-28 lg:pt-36 pb-32 lg:pb-40 bg-transparent font-sans">

      {/* GTA 6 Vice City Sunset / Cyberpunk Ag Background */}
      <div className="absolute inset-0 z-0">
        {/* Vice City Sky Gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-[#0a0a0f] via-[#2a0845] to-[#f42b8e] opacity-60 mix-blend-screen" />

        {/* Ambient Drifting Glow Orbs */}
        <motion.div
          animate={{ scale: [1, 1.4, 1], opacity: [0.5, 0.9, 0.5], x: [0, 60, 0], y: [0, -40, 0] }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-[10%] left-[15%] w-[400px] h-[400px] bg-[#0ff] rounded-full blur-[100px] opacity-70 pointer-events-none"
        />
        <motion.div
          animate={{ scale: [1, 1.6, 1], opacity: [0.4, 0.8, 0.4], x: [0, -80, 0], y: [0, 70, 0] }}
          transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
          className="absolute bottom-[20%] right-[10%] w-[500px] h-[500px] bg-[#f0f] rounded-full blur-[120px] opacity-60 pointer-events-none"
        />
        <motion.div
          animate={{ scale: [1, 1.3, 1], opacity: [0.6, 1, 0.6], x: [0, 40, 0], y: [0, 40, 0] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-[40%] right-[30%] w-[300px] h-[300px] bg-amber-500 rounded-full blur-[80px] opacity-80 pointer-events-none"
        />

        {/* Sun / Neon Orb */}
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          className="absolute bottom-[-20%] right-[-10%] w-[80vw] h-[80vw] max-w-[800px] max-h-[800px] bg-gradient-to-t from-[#f42b8e] to-[#ffb703] rounded-full blur-[60px] opacity-60 pointer-events-none"
        />



        {/* Retro Grid Floor */}
        <div className="absolute bottom-0 inset-x-0 h-[40vh] bg-[linear-gradient(rgba(244,43,142,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,255,0.05)_1px,transparent_1px)] bg-[size:60px_40px] [transform:perspective(500px)_rotateX(60deg)] [transform-origin:bottom]" />
      </div>

      <div className="relative z-10 w-full max-w-[90rem] mx-auto flex flex-col lg:flex-row items-center gap-12 lg:gap-24 mb-0">

        {/* Left Column: Title & Overview */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className="flex-1 text-center lg:text-left"
        >
          <div className="inline-flex items-center px-4 py-2 lg:px-5 lg:py-2.5 rounded-full bg-black/40 border border-pink-500/30 text-pink-400 font-bold tracking-widest text-sm uppercase mb-8 backdrop-blur-md">
            <Gamepad2 className="w-5 h-5 mr-2" /> V.1.0 // The Neon Harvest
          </div>

          <h1 className="text-8xl lg:text-[10rem] font-black text-transparent bg-clip-text bg-gradient-to-br from-[#0ff] via-white to-[#f42b8e] tracking-tighter mb-6 italic drop-shadow-[0_0_30px_rgba(244,43,142,0.6)] leading-none">
            AgriBot
          </h1>

          <p className="text-xl lg:text-3xl text-gray-300 font-medium mb-10 leading-relaxed max-w-2xl mx-auto lg:mx-0">
            Welcome to the ultimate agricultural command center. High-octane AI diagnostics, real-time satellite telemetry, and autonomous fleet tracking.
            <span className="block mt-4 text-pink-400 font-bold italic">The future of farming is neon.</span>
          </p>

          <Link to="/dashboard">
            <motion.button
              whileHover={{ scale: 1.05, textShadow: "0px 0px 12px rgb(0,255,255)" }}
              whileTap={{ scale: 0.95 }}
              className="group relative inline-flex items-center justify-center px-8 py-4 lg:px-10 lg:py-5 font-bold text-black uppercase tracking-widest bg-gradient-to-r from-[#0ff] to-[#f42b8e] rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(244,43,142,0.7)]"
            >
              <div className="absolute inset-0 w-full h-full bg-white/20 group-hover:translate-x-full transition-transform duration-500 ease-out -skew-x-12 -translate-x-full" />
              <span className="relative flex items-center z-10 text-lg lg:text-2xl">
                Enter Operations <ChevronRight className="ml-3 w-6 h-6 lg:w-8 lg:h-8" />
              </span>
            </motion.button>
          </Link>
        </motion.div>

        {/* Right Column: Features Grid */}
        <motion.div
          initial={{ y: 50 }}
          animate={{ y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="flex-1 w-full grid grid-cols-1 md:grid-cols-2 gap-5 lg:gap-8"
        >
          {features.map((feature, i) => (
            <motion.div
              key={i}
              whileHover={{ y: -5, scale: 1.02 }}
              className="relative group"
            >
              <div className={`absolute -inset-0.5 bg-gradient-to-r ${feature.color} rounded-3xl blur opacity-30 group-hover:opacity-70 transition duration-500`}></div>
              <Card className="relative h-full bg-black/60 backdrop-blur-xl border border-white/5 rounded-3xl overflow-hidden">
                <CardContent className="p-6 md:p-10 flex flex-col h-full">
                  <div className={`w-14 h-14 lg:w-16 lg:h-16 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-5 shadow-xl shadow-black/50`}>
                    <feature.icon className="w-7 h-7 lg:w-8 lg:h-8 text-white" />
                  </div>
                  <h3 className="text-xl lg:text-2xl font-bold text-white mb-3 italic tracking-wide">{feature.title}</h3>
                  <p className="text-sm lg:text-lg text-gray-300 leading-relaxed font-medium">
                    {feature.desc}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Full-width Infinite Tractor Harvest Animation seamlessly integrated into the environment */}
      <div className="absolute bottom-0 left-0 w-full h-[100px] lg:h-[140px] overflow-hidden pointer-events-none z-50">

        {/* Stationary Crops with phase-locked regrowth math spanning the entire width */}
        <div className="absolute bottom-4 left-0 w-full flex justify-between px-2 lg:px-4">
          {[...Array(60)].map((_, i) => {
            const cropVw = (i / 59) * 100;
            // Tractor travels 140vw in 15 seconds. Offset blade reach by 0.3s.
            const hitTime = (((cropVw + 20) / 140) * 15) - 0.3;
            return (
              <motion.div
                key={i}
                initial={{ opacity: 1, scale: 1 }}
                animate={{ opacity: [1, 0, 0, 1, 1], scale: [1, 0, 0, 1, 1] }}
                transition={{
                  duration: 15,
                  repeat: Infinity,
                  ease: "linear",
                  times: [0, 0.05, 0.6, 0.8, 1],
                  delay: Math.max(0, hitTime)
                }}
                className="flex items-center justify-center shrink-0 w-6 h-6 lg:w-8 lg:h-8"
              >
                <PixelSprout className="w-6 h-6 lg:w-8 lg:h-8 text-green-400 drop-shadow-[0_0_12px_rgba(74,222,128,0.8)]" />
              </motion.div>
            );
          })}
        </div>

        {/* Moving Tractor */}
        <motion.div
          initial={{ x: "-20vw" }}
          animate={{ x: "120vw" }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
          className="absolute bottom-4 left-0 z-50 flex items-end"
        >
          <motion.div
            animate={{ y: [0, -3, 0] }}
            transition={{ duration: 0.2, repeat: Infinity }}
            className="relative z-50"
          >
            <PixelTractor className="w-14 h-14 lg:w-20 lg:h-20 text-yellow-400 drop-shadow-[0_0_20px_rgba(250,204,21,1)]" />
            {/* Neon Laser Blade pulling forward */}
            <div className="absolute right-[-25px] bottom-2 w-10 h-[3px] lg:h-[4px] bg-cyan-300 shadow-[0_0_15px_#0ff] rounded-full" />
            <div className="absolute right-[-15px] bottom-4 lg:bottom-6 w-6 lg:w-8 h-[3px] lg:h-[4px] bg-cyan-300 shadow-[0_0_15px_#0ff] rounded-full" />
          </motion.div>
        </motion.div>
      </div>

    </div>
  );
}
