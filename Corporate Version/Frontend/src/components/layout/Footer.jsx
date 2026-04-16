import React from 'react';
import { Link } from 'react-router-dom';
import { Github, Globe, Leaf, ExternalLink, Mail, Shield, BookOpen } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const footerSections = [
    {
      title: "Quick Links",
      links: [
        { name: "FAO Official", url: "https://www.fao.org", icon: <Globe className="w-4 h-4" /> },
        { name: "USDA Portal", url: "https://www.usda.gov", icon: <Shield className="w-4 h-4" /> },
        { name: "PlantVillage", url: "https://plantvillage.psu.edu", icon: <Leaf className="w-4 h-4" /> },
      ]
    },
    {
      title: "Resources",
      links: [
        { name: "EPPO Database", url: "https://gd.eppo.int", icon: <BookOpen className="w-4 h-4" /> },
        { name: "SARE Resource", url: "https://www.sare.org", icon: <ExternalLink className="w-4 h-4" /> },
        { name: "AgriResearch", url: "https://www.cgiar.org", icon: <Globe className="w-4 h-4" /> },
      ]
    },
    {
      title: "Developer",
      links: [
        { name: "GitHub Profile", url: "https://github.com/iamhriturajsaha", icon: <Github className="w-4 h-4" /> },
        { name: "Support Project", url: "https://github.com/iamhriturajsaha/AGRIBOT-AI-POWERED-FARMING-ASSISTANT", icon: <ExternalLink className="w-4 h-4" /> },
      ]
    }
  ];

  return (
    <footer className="mt-16 border-t border-slate-800/50 pt-12 pb-8 px-4">
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-12">
        {/* Brand Section */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-agri-green/20 rounded-xl flex items-center justify-center border border-agri-green/30 shadow-[0_0_15px_rgba(22,101,52,0.2)]">
              <Leaf className="w-6 h-6 text-agri-lightGreen" />
            </div>
            <span className="text-xl font-display font-bold text-white tracking-tight">Agri<span className="text-agri-lightGreen">Bot</span></span>
          </div>
          <p className="text-slate-400 text-sm leading-relaxed max-w-xs">
            Empowering modern agriculture with AI-driven plant health diagnostics and precision crop analysis.
          </p>
          <div className="flex space-x-4">
            <a 
              href="https://github.com/iamhriturajsaha" 
              target="_blank" 
              rel="noopener noreferrer"
              className="w-9 h-9 flex items-center justify-center rounded-lg bg-slate-800/50 text-slate-400 hover:text-white hover:bg-slate-700 transition-all duration-300 border border-slate-700/50"
            >
              <Github className="w-5 h-5" />
            </a>
            <a 
              href="https://mail.google.com/mail/?view=cm&fs=1&to=iamhriturajsaha@gmail.com" 
              target="_blank"
              rel="noopener noreferrer"
              className="w-9 h-9 flex items-center justify-center rounded-lg bg-slate-800/50 text-slate-400 hover:text-white hover:bg-slate-700 transition-all duration-300 border border-slate-700/50"
            >
              <Mail className="w-5 h-5" />
            </a>
          </div>
        </div>

        {/* Links Sections */}
        {footerSections.map((section, idx) => (
          <div key={idx} className="space-y-6">
            <h4 className="text-white font-bold text-sm uppercase tracking-widest">{section.title}</h4>
            <ul className="space-y-4">
              {section.links.map((link, lIdx) => (
                <li key={lIdx}>
                  <a 
                    href={link.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="group flex items-center text-slate-400 hover:text-agri-lightGreen transition-colors duration-300 text-sm"
                  >
                    <span className="mr-3 p-1.5 rounded-md bg-slate-800/30 border border-slate-700/50 group-hover:border-agri-green/30 group-hover:bg-agri-green/10 transition-all duration-300">
                      {link.icon}
                    </span>
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="max-w-6xl mx-auto pt-8 border-t border-slate-800/50 flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0 text-xs text-slate-500 font-medium">
        <p>© {currentYear} AgriBot AI. All rights reserved.</p>
        <div className="flex items-center space-x-6">
          <Link to="/privacy" className="hover:text-agri-lightGreen transition-colors">Privacy Policy</Link>
          <Link to="/terms" className="hover:text-agri-lightGreen transition-colors">Terms of Service</Link>
          <Link to="/cookies" className="hover:text-agri-lightGreen transition-colors">Cookies</Link>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
