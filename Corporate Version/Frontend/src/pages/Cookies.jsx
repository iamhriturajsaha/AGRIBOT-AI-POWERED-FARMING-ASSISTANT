import React from 'react';
import { Cookie, ShieldCheck, Settings, Info, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const Cookies = () => {
  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <Link to="/home" className="inline-flex items-center text-agri-lightGreen hover:text-white transition-colors mb-8 group">
        <ArrowLeft className="w-4 h-4 mr-2 transform group-hover:-translate-x-1 transition-transform" />
        Back to Home
      </Link>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-slate-900/40 border border-slate-800 rounded-3xl p-8 md:p-12 backdrop-blur-xl shadow-2xl"
      >
        <div className="flex items-center space-x-4 mb-8">
          <div className="w-12 h-12 bg-agri-green/20 rounded-2xl flex items-center justify-center border border-agri-green/30">
            <Cookie className="w-6 h-6 text-agri-lightGreen" />
          </div>
          <div>
            <h1 className="text-3xl font-display font-bold text-white">Cookie Policy</h1>
            <p className="text-slate-400 text-sm">Last updated: April 16, 2026</p>
          </div>
        </div>

        <div className="prose prose-invert prose-slate max-w-none space-y-8 text-slate-300">
          <section>
            <h2 className="text-xl font-bold text-white flex items-center mb-4">
              <Info className="w-5 h-5 mr-3 text-agri-green" /> 1. What are Cookies?
            </h2>
            <p>
              Cookies are small text files that are stored on your device when you visit a website. They help the platform remember your preferences, keep you logged in, and provide a smoother user experience.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white flex items-center mb-4">
              <ShieldCheck className="w-5 h-5 mr-3 text-agri-green" /> 2. How We Use Cookies
            </h2>
            <div className="space-y-4">
              <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                <h3 className="text-lg font-bold text-white mb-2">Essential Cookies</h3>
                <p className="text-sm text-slate-400">These are required for the platform to function. They handle authentication and security sessions. Without them, you cannot use our services.</p>
              </div>
              <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                <h3 className="text-lg font-bold text-white mb-2">Preference Cookies</h3>
                <p className="text-sm text-slate-400">These remember your settings like UI preferences and dashboard layout selections.</p>
              </div>
              <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                <h3 className="text-lg font-bold text-white mb-2">Analytical Cookies</h3>
                <p className="text-sm text-slate-400">We use these anonymized cookies to understand how users interact with AgriBot, helping us optimize the dashboard and diagnostic flows.</p>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white flex items-center mb-4">
              <Settings className="w-5 h-5 mr-3 text-agri-green" /> 3. Managing Cookies
            </h2>
            <p>
              You can control and manage cookies through your browser settings. Most browsers allow you to block or delete cookies, though this may impact the functionality of AgriBot.
            </p>
            <p className="mt-4">
              For more information on how to manage cookies, visit <a href="https://www.aboutcookies.org" target="_blank" rel="noopener noreferrer" className="text-agri-lightGreen hover:underline">aboutcookies.org</a>.
            </p>
          </section>
        </div>
      </motion.div>
    </div>
  );
};

export default Cookies;
