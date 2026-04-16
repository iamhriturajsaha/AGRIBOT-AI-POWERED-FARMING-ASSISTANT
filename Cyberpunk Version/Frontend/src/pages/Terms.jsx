import React from 'react';
import { FileText, AlertTriangle, Scale, UserCheck, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const Terms = () => {
  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <Link to="/home" className="inline-flex items-center text-neon-blue hover:text-white transition-colors mb-8 group uppercase tracking-widest font-black italic text-xs">
        <ArrowLeft className="w-4 h-4 mr-2 transform group-hover:-translate-x-1 transition-transform" />
        // Return to Operations
      </Link>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-black/60 border border-white/5 rounded-3xl p-8 md:p-12 backdrop-blur-xl shadow-[0_0_50px_rgba(244,43,142,0.05)] relative overflow-hidden"
      >
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-neon-blue via-transparent to-neon-pink opacity-50" />
        
        <div className="flex items-center space-x-6 mb-12">
          <div className="w-16 h-16 bg-black/40 rounded-2xl flex items-center justify-center border border-neon-blue/30 shadow-[0_0_15px_rgba(244,43,142,0.2)]">
            <FileText className="w-8 h-8 text-neon-blue" />
          </div>
          <div>
            <h1 className="text-4xl font-display font-black text-white italic tracking-tighter uppercase">Terms // Directive</h1>
            <p className="text-neon-pink text-xs font-bold tracking-[0.2em] mt-2">LAST SYNC: APRIL 16, 2026</p>
          </div>
        </div>

        <div className="prose prose-invert prose-slate max-w-none space-y-10 text-gray-400">
          <section>
            <h2 className="text-xl font-black text-white flex items-center mb-4 italic uppercase tracking-widest">
              <Scale className="w-5 h-5 mr-3 text-neon-blue" /> 01. Operational Consent
            </h2>
            <p className="font-medium leading-relaxed">
              By initiating an uplink with AgriBot Core, you enter into a binding directive. Failure to comply with operational standards results in immediate termination of the neural interface.
            </p>
          </section>

          <section className="bg-neon-pink/5 border border-neon-pink/20 p-8 rounded-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 p-2 opacity-10">
              <AlertTriangle className="w-20 h-20 text-neon-pink" />
            </div>
            <h2 className="text-xl font-black text-neon-pink flex items-center mb-4 italic uppercase tracking-widest">
              <AlertTriangle className="w-5 h-5 mr-3" /> 02. Diagnostic Disclaimer
            </h2>
            <p className="text-white font-bold italic leading-relaxed">
              AI-GENERATED CROP DIAGNOSTICS ARE FOR DECISION SUPPORT ONLY. THEY ARE NOT A REPLACEMENT FOR FIELD INSPECTION OR PROFESSIONAL BIO-ANALYTICS. AGRIBOT CORE DISCLAIMS ALL LIABILITY FOR CROP LOSS RESULTING FROM NEURAL ERRORS.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-black text-white flex items-center mb-4 italic uppercase tracking-widest">
              <UserCheck className="w-5 h-5 mr-3 text-neon-blue" /> 03. Tactical Responsibility
            </h2>
            <p>The Operator is responsible for:</p>
            <ul className="list-none pl-4 space-y-4 mt-6 border-l border-white/10">
              <li className="flex items-center">
                <span className="text-neon-blue mr-3 font-black">&gt;&gt;</span>
                <span>Maintaining sensor integrity and image resolution for analysis.</span>
              </li>
              <li className="flex items-center">
                <span className="text-neon-blue mr-3 font-black">&gt;&gt;</span>
                <span>Securing tactical login credentials across all interface nodes.</span>
              </li>
              <li className="flex items-center">
                <span className="text-neon-blue mr-3 font-black">&gt;&gt;</span>
                <span>Compliance with local bio-security and agricultural laws.</span>
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-black text-white mb-6 italic uppercase tracking-widest">04. Force Majeure</h2>
            <p className="font-medium">
              AgriBot Core is not liable for system outages caused by massive solar flares, orbital interference, or global network destabilization.
            </p>
          </section>
        </div>
      </motion.div>
    </div>
  );
};

export default Terms;
