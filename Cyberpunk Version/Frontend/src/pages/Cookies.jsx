import React from 'react';
import { Cookie, ShieldCheck, Settings, Info, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const Cookies = () => {
  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <Link to="/home" className="inline-flex items-center text-neon-pink hover:text-white transition-colors mb-8 group uppercase tracking-widest font-black italic text-xs">
        <ArrowLeft className="w-4 h-4 mr-2 transform group-hover:-translate-x-1 transition-transform" />
        // Return to Operations
      </Link>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-black/60 border border-white/5 rounded-3xl p-8 md:p-12 backdrop-blur-xl shadow-[0_0_50px_rgba(255,183,3,0.05)] relative overflow-hidden"
      >
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-neon-pink via-transparent to-neon-green opacity-50" />
        
        <div className="flex items-center space-x-6 mb-12">
          <div className="w-16 h-16 bg-black/40 rounded-2xl flex items-center justify-center border border-neon-pink/30 shadow-[0_0_15px_rgba(255,183,3,0.2)]">
            <Cookie className="w-8 h-8 text-neon-pink" />
          </div>
          <div>
            <h1 className="text-4xl font-display font-black text-white italic tracking-tighter uppercase">Cookies // Cache</h1>
            <p className="text-neon-green text-xs font-bold tracking-[0.2em] mt-2">LAST SYNC: APRIL 16, 2026</p>
          </div>
        </div>

        <div className="prose prose-invert prose-slate max-w-none space-y-10 text-gray-400">
          <section>
            <h2 className="text-xl font-black text-white flex items-center mb-4 italic uppercase tracking-widest">
              <Info className="w-5 h-5 mr-3 text-neon-pink" /> 01. Neural Markers
            </h2>
            <p className="font-medium">
              We use small temporal data nodes, known as cookies, to maintain your neural uplink. These markers allow the system to remember your tactical preferences and session state.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-black text-white flex items-center mb-4 italic uppercase tracking-widest">
              <ShieldCheck className="w-5 h-5 mr-3 text-neon-pink" /> 02. Data Types
            </h2>
            <div className="space-y-6 mt-6">
              <div className="p-6 bg-black/40 rounded-2xl border border-white/10 group hover:border-neon-pink/30 transition-all duration-300">
                <h3 className="text-lg font-black text-white mb-2 italic uppercase tracking-widest">Hard-Code Necessary</h3>
                <p className="text-sm text-gray-500 font-medium tracking-tight">Essential for authentication and account security. Without these nodes, the core platform cannot bridge your connection.</p>
              </div>
              <div className="p-6 bg-black/40 rounded-2xl border border-white/10 group hover:border-neon-pink/30 transition-all duration-300">
                <h3 className="text-lg font-black text-white mb-2 italic uppercase tracking-widest">Preference Sync</h3>
                <p className="text-sm text-gray-500 font-medium tracking-tight">Preserves your custom UI settings, dashboard configurations, and language matrices.</p>
              </div>
              <div className="p-6 bg-black/40 rounded-2xl border border-white/10 group hover:border-neon-pink/30 transition-all duration-300">
                <h3 className="text-lg font-black text-white mb-2 italic uppercase tracking-widest">Analytical Stream</h3>
                <p className="text-sm text-gray-500 font-medium tracking-tight">Anonymized telemetry used to optimize neural processing speeds and diagnostic accuracy.</p>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-xl font-black text-white flex items-center mb-4 italic uppercase tracking-widest">
              <Settings className="w-5 h-5 mr-3 text-neon-pink" /> 03. Tactical Overrides
            </h2>
            <p className="font-medium">
              You can override and purge all cookies through your browser interface. Note that disabling essential cookies will destabilize your access to AgriBot Core.
            </p>
          </section>
        </div>
      </motion.div>
    </div>
  );
};

export default Cookies;
