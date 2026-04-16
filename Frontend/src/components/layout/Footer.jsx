import React from 'react';
import { Link } from 'react-router-dom';
import { Github, Globe, Leaf, ExternalLink, Mail, Shield, BookOpen, Cpu } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const footerSections = [
    {
      title: "Neural Links",
      links: [
        { name: "FAO Hub", url: "https://www.fao.org", icon: <Globe className="w-4 h-4" /> },
        { name: "USDA Uplink", url: "https://www.usda.gov", icon: <Shield className="w-4 h-4" /> },
        { name: "PlantVillage", url: "https://plantvillage.psu.edu", icon: <Leaf className="w-4 h-4" /> },
      ]
    },
    {
      title: "Data Nodes",
      links: [
        { name: "EPPO Global", url: "https://gd.eppo.int", icon: <BookOpen className="w-4 h-4" /> },
        { name: "SARE Systems", url: "https://www.sare.org", icon: <ExternalLink className="w-4 h-4" /> },
        { name: "AgriResearch", url: "https://www.cgiar.org", icon: <Globe className="w-4 h-4" /> },
      ]
    },
    {
      title: "The Architect",
      links: [
        { name: "GitHub ID", url: "https://github.com/iamhriturajsaha", icon: <Github className="w-4 h-4" /> },
        { name: "Primary Repo", url: "https://github.com/iamhriturajsaha/AGRIBOT-AI-POWERED-FARMING-ASSISTANT", icon: <Cpu className="w-4 h-4" /> },
      ]
    }
  ];

  return (
    <footer className="relative mt-20 border-t border-white/5 pt-16 pb-12 px-6 overflow-hidden">
      {/* Cyberpunk Background Accents */}
      <div className="absolute top-0 left-1/4 w-1/2 h-px bg-gradient-to-r from-transparent via-neon-green/30 to-transparent" />
      <div className="absolute top-0 right-1/4 w-px h-20 bg-gradient-to-b from-neon-blue/20 to-transparent" />

      <div className="max-w-[1400px] mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-16 relative z-10 text-center md:text-left">
        {/* Brand Section */}
        <div className="space-y-6 flex flex-col items-center md:items-start">
          <div className="flex items-center space-x-3 group">
            <div className="w-12 h-12 bg-black/40 rounded-xl flex items-center justify-center border border-white/10 group-hover:border-neon-green/50 shadow-[0_0_20px_rgba(0,255,255,0.1)] transition-all duration-500">
              <Leaf className="w-7 h-7 text-neon-green" />
            </div>
            <span className="text-2xl font-display font-black text-white italic tracking-tighter">
              Agri<span className="text-neon-green hover:text-neon-blue transition-colors duration-500">Bot</span>
            </span>
          </div>
          <p className="text-gray-400 text-sm leading-relaxed max-w-xs font-medium italic">
            Synthetic diagnostics for the modern bio-frontier. Processing crop data at lethal precision.
          </p>
          <div className="flex space-x-4">
            <a 
              href="https://github.com/iamhriturajsaha" 
              target="_blank" 
              rel="noopener noreferrer"
              className="w-10 h-10 flex items-center justify-center rounded-xl bg-white/5 text-gray-400 hover:text-neon-green hover:bg-white/10 transition-all duration-500 border border-white/5 hover:border-neon-green/30 shadow-lg"
            >
              <Github className="w-5 h-5" />
            </a>
            <a 
              href="https://mail.google.com/mail/?view=cm&fs=1&to=iamhriturajsaha@gmail.com" 
              target="_blank"
              rel="noopener noreferrer"
              className="w-10 h-10 flex items-center justify-center rounded-xl bg-white/5 text-gray-400 hover:text-neon-blue hover:bg-white/10 transition-all duration-500 border border-white/5 hover:border-neon-blue/30 shadow-lg"
            >
              <Mail className="w-5 h-5" />
            </a>
          </div>
        </div>

        {/* Links Sections */}
        {footerSections.map((section, idx) => (
          <div key={idx} className="space-y-8">
            <h4 className="text-white font-black text-xs uppercase tracking-[0.3em] italic opacity-80 border-l-2 border-neon-blue pl-4">{section.title}</h4>
            <ul className="space-y-5">
              {section.links.map((link, lIdx) => (
                <li key={lIdx}>
                  <a 
                    href={link.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="group flex items-center justify-center md:justify-start text-gray-400 hover:text-white transition-all duration-300 text-sm font-bold uppercase tracking-wider"
                  >
                    <span className="mr-4 p-2 rounded-lg bg-black/40 border border-white/5 group-hover:border-neon-green/30 group-hover:bg-neon-green/5 transition-all duration-500 group-hover:shadow-[0_0_15px_rgba(0,255,255,0.2)]">
                      {React.cloneElement(link.icon, { className: 'w-4 h-4 group-hover:text-neon-green transition-colors' })}
                    </span>
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="max-w-[1400px] mx-auto pt-10 border-t border-white/5 flex flex-col md:flex-row justify-between items-center space-y-6 md:space-y-0 text-[10px] text-gray-500 font-black uppercase tracking-[0.2em]">
        <p className="flex items-center">
          <span className="w-1.5 h-1.5 bg-neon-green rounded-full mr-2 animate-pulse shadow-[0_0_8px_#0ff]" />
          © {currentYear} AGRIBOT CORE. ALL RIGHTS RESERVED.
        </p>
        <div className="flex items-center space-x-8">
          <Link to="/privacy" className="hover:text-neon-green transition-colors duration-300">Privacy // Protocol</Link>
          <Link to="/terms" className="hover:text-neon-green transition-colors duration-300">Terms // Directive</Link>
          <Link to="/cookies" className="hover:text-neon-green transition-colors duration-300">Cookies // Cache</Link>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
