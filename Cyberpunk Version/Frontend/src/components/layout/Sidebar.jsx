import { NavLink, useNavigate } from 'react-router-dom';
import { Home, Leaf, MessageSquare, LogOut, Sprout, HelpCircle, Clock, User, Terminal, Activity, Music, Volume2, VolumeX, TrendingUp, Map, TreePine, Network } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { cn } from '../common/Button';
import { useState, useRef, useEffect } from 'react';

export function Sidebar({ isHomeRoute }) {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const toggleMusic = () => {
    if (!audioRef.current) return;
    
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play().catch(e => console.log('Playback prevented', e));
    }
    setIsPlaying(!isPlaying);
  };

  const handleAudioEnded = () => {
    if (audioRef.current && isPlaying) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(e => console.log('Loop prevented', e));
    }
  };

  const navItems = [
    { name: 'Overview', path: '/home', icon: Home },
    { name: 'Command Center', path: '/dashboard', icon: Activity },
    { name: 'Prediction', path: '/prediction', icon: Leaf },
    { name: 'AI Chat', path: '/chat', icon: MessageSquare },
    { name: 'History', path: '/history', icon: Clock },
    { name: 'FAQ', path: '/faq', icon: HelpCircle },
    { name: 'Profile', path: '/profile', icon: User },
    { name: 'Sys.Admin', path: '/developer', icon: Terminal },
  ];

  return (
    <>
      {/* Hidden Audio Player preserved globally across layout shifts */}
      <audio ref={audioRef} src="/synthwave.mp3" loop onEnded={handleAudioEnded} />

      <aside className={isHomeRoute
        ? "flex items-center space-x-1 lg:space-x-2 bg-black/60 backdrop-blur-xl border border-white/10 rounded-full px-4 py-2 lg:px-5 lg:py-2.5 shadow-[0_0_20px_rgba(0,0,0,0.8)] scale-90 sm:scale-100"
        : "w-64 flex flex-col hidden md:flex h-screen sticky top-0 bg-transparent"
      }>
        {isHomeRoute ? (
          <>
            <Sprout className="w-5 h-5 lg:w-6 lg:h-6 text-neon-green mr-2 lg:mr-4 animate-pulse" />
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  title={item.name}
                  className={({ isActive }) => cn(
                    "p-2 lg:p-2.5 rounded-full transition-all duration-300 group shadow-none",
                    isActive
                      ? "bg-neon-blue/20 text-white shadow-[inset_0_0_15px_rgba(0,195,255,0.4)]"
                      : "text-gray-400 hover:text-white hover:bg-white/5"
                  )}
                >
                  <Icon className="w-4 h-4 lg:w-5 lg:h-5 transition-transform group-hover:scale-110 group-hover:drop-shadow-[0_0_8px_currentColor]" />
                </NavLink>
              );
            })}

            <div className="w-px h-5 lg:h-6 bg-white/20 mx-1 lg:mx-2" />

            <button
              onClick={toggleMusic}
              title="Toggle BGM"
              className={cn(
                "p-2 lg:p-2.5 rounded-full transition-all duration-300 group shadow-none",
                isPlaying
                  ? "text-cyan-400 shadow-[inset_0_0_15px_rgba(0,255,255,0.4)] bg-cyan-500/10"
                  : "text-gray-400 hover:text-white hover:bg-white/5"
              )}
            >
              {isPlaying ? <Music className="w-4 h-4 lg:w-5 lg:h-5 animate-pulse drop-shadow-[0_0_5px_currentColor]" /> : <VolumeX className="w-4 h-4 lg:w-5 lg:h-5 transition-transform group-hover:scale-110" />}
            </button>

            <button
              onClick={handleLogout}
              title="System Exit"
              className="p-2 lg:p-2.5 text-red-500 hover:bg-red-500/10 rounded-full transition-all duration-300 group"
            >
              <LogOut className="w-4 h-4 lg:w-5 lg:h-5 transition-transform group-hover:scale-110 group-hover:drop-shadow-[0_0_8px_currentColor]" />
            </button>
          </>
        ) : (
          <>
            <div className="h-20 flex items-center px-8 border-b border-transparent">
              <Sprout className="w-8 h-8 text-neon-green mr-3" />
              <span className="text-3xl font-black italic tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-[#0ff] to-[#f42b8e] drop-shadow-[0_0_10px_rgba(244,43,142,0.3)]">
                AgriBot
              </span>
            </div>

            <div className="flex-1 px-4 py-6 space-y-2">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) => cn(
                      "flex items-center px-6 py-4 transition-all duration-300 group font-bold tracking-wide",
                      isActive
                        ? "bg-gradient-to-r from-neon-blue/20 to-transparent border-l-4 border-neon-blue text-white shadow-[inset_20px_0_20px_-20px_rgba(244,43,142,0.5)]"
                        : "border-l-4 border-transparent text-gray-400 hover:text-white hover:bg-white/5"
                    )}
                  >
                    <Icon className="w-5 h-5 mr-3 transition-transform group-hover:scale-110" />
                    {item.name}
                  </NavLink>
                );
              })}
            </div>

            <div className="p-6 border-t border-transparent mt-auto space-y-4">
              {/* BGM Toggle Button */}
              <button
                onClick={toggleMusic}
                className={`flex w-full items-center justify-between px-4 py-3 font-bold tracking-wider transition-all duration-300 rounded-xl border ${isPlaying
                    ? "bg-cyan-500/10 border-cyan-400/50 text-cyan-400 shadow-[inset_0_0_15px_rgba(0,255,255,0.2)]"
                    : "bg-black/40 border-white/10 text-gray-500 hover:text-gray-300 hover:border-white/20"
                  }`}
              >
                <div className="flex items-center text-sm">
                  <Music className={`w-4 h-4 mr-3 ${isPlaying ? "animate-pulse drop-shadow-[0_0_5px_#0ff]" : ""}`} />
                  <span>SYNTHWAVE BGM</span>
                </div>
                {isPlaying ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
              </button>

              {/* Logout Button */}
              <button
                onClick={handleLogout}
                className="flex w-full items-center justify-center px-4 py-3 text-red-500 text-sm font-bold tracking-wider transition-all duration-200 rounded-xl border border-transparent hover:bg-red-500/10 hover:border-red-500/30 hover:shadow-[0_0_15px_rgba(239,68,68,0.2)] group uppercase"
              >
                <LogOut className="w-4 h-4 mr-3 transition-transform group-hover:-translate-x-1" />
                System Exit
              </button>
            </div>
          </>
        )}
      </aside>
    </>
  );
}
