import React from 'react';
import { FileText, AlertTriangle, Scale, UserCheck, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const Terms = () => {
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
            <FileText className="w-6 h-6 text-agri-lightGreen" />
          </div>
          <div>
            <h1 className="text-3xl font-display font-bold text-white">Terms of Service</h1>
            <p className="text-slate-400 text-sm">Last updated: April 16, 2026</p>
          </div>
        </div>

        <div className="prose prose-invert prose-slate max-w-none space-y-8 text-slate-300">
          <section>
            <h2 className="text-xl font-bold text-white flex items-center mb-4">
              <Scale className="w-5 h-5 mr-3 text-agri-green" /> 1. Acceptance of Terms
            </h2>
            <p>
              By accessing or using the AgriBot platform, you agree to be bound by these Terms of Service. If you do not agree to these terms, please do not use our services.
            </p>
          </section>

          <section className="bg-amber-900/20 border border-amber-900/50 p-6 rounded-2xl">
            <h2 className="text-xl font-bold text-amber-500 flex items-center mb-4">
              <AlertTriangle className="w-5 h-5 mr-3" /> 2. AI Diagnostic Disclaimer
            </h2>
            <p className="text-amber-100/80 leading-relaxed font-medium">
              AgriBot provides AI-generated crop diagnostics for <strong>decision support purposes only</strong>. Our assessments are based on machine learning models and are not a replacement for professional agronomic advice, on-site inspections, or laboratory testing. AgriBot does not guarantee 100% accuracy in its predictions.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white flex items-center mb-4">
              <UserCheck className="w-5 h-5 mr-3 text-agri-green" /> 3. User Responsibilities
            </h2>
            <p>You are responsible for:</p>
            <ul className="list-disc pl-6 space-y-2 mt-2">
              <li>Providing clear and accurate images for diagnostic analysis.</li>
              <li>Maintaining the confidentiality of your account credentials.</li>
              <li>Ensuring that your use of the platform complies with local agricultural regulations.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-4">4. Limitation of Liability</h2>
            <p>
              To the maximum extent permitted by law, AgriBot and its developers shall not be liable for any crop loss, financial loss, or damages resulting from the use of or reliance upon AI diagnostic results. Final decisions regarding crop treatment, fertilizer application, or harvesting should be made using multiple sources of information.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-white mb-4">5. Modifications to Service</h2>
            <p>
              We reserve the right to modify, suspend, or discontinue any part of the service at any time. We will notify users of significant changes to these terms via email or platform notifications.
            </p>
          </section>
        </div>
      </motion.div>
    </div>
  );
};

export default Terms;
