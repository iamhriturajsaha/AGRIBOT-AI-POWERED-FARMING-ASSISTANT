import { NavLink, useNavigate } from 'react-router-dom';
import { Home, Leaf, MessageSquare, LogOut, Sprout, HelpCircle, Clock, User, Phone, Activity } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { cn } from '../common/Button';

export function Sidebar({ isHomeRoute }) {
  const { logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    { name: 'Overview', path: '/home', icon: Home },
    { name: 'Dashboard', path: '/dashboard', icon: Activity },
    { name: 'Prediction', path: '/prediction', icon: Leaf },
    { name: 'AI Assistant', path: '/chat', icon: MessageSquare },
    { name: 'History', path: '/history', icon: Clock },
    { name: 'FAQ', path: '/faq', icon: HelpCircle },
    { name: 'Profile', path: '/profile', icon: User },
    { name: 'Contact', path: '/contact', icon: Phone },
  ];

  return (
    <aside className={isHomeRoute
      ? "flex items-center space-x-1 lg:space-x-2 bg-slate-900/40/90 backdrop-blur-md border border-slate-800 rounded-full px-4 py-2 lg:px-5 lg:py-2.5 shadow-sm scale-90 sm:scale-100"
      : "w-64 flex flex-col hidden md:flex h-screen sticky top-0 bg-slate-900/40 border-r border-slate-800 shadow-sm"
    }>
      {isHomeRoute ? (
        <>
          <Sprout className="w-5 h-5 lg:w-6 lg:h-6 text-agri-green mr-2 lg:mr-4" />
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
                    ? "bg-agri-green/10 text-agri-green"
                    : "text-slate-400 hover:text-agri-green hover:bg-slate-800/50"
                )}
              >
                <Icon className="w-4 h-4 lg:w-5 lg:h-5 transition-transform group-hover:scale-110" />
              </NavLink>
            );
          })}

          <div className="w-px h-5 lg:h-6 bg-slate-200 mx-1 lg:mx-2" />

          <button
            onClick={handleLogout}
            title="Logout"
            className="p-2 lg:p-2.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-full transition-all duration-300 group"
          >
            <LogOut className="w-4 h-4 lg:w-5 lg:h-5 transition-transform group-hover:scale-110" />
          </button>
        </>
      ) : (
        <>
          <div className="h-20 flex items-center px-8 border-b border-slate-800">
            <Sprout className="w-8 h-8 text-agri-green mr-3" />
            <span className="text-2xl font-bold tracking-tight text-white">
              AgriBot
            </span>
          </div>

          <div className="flex-1 px-4 py-6 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => cn(
                    "flex items-center px-4 py-3 transition-all duration-200 rounded-xl font-medium tracking-wide",
                    isActive
                      ? "bg-agri-green/10 text-agri-green"
                      : "text-slate-400 hover:text-white hover:bg-slate-800/50"
                  )}
                >
                  {({ isActive }) => (
                    <>
                      <Icon className={cn(
                        "w-5 h-5 mr-3 transition-transform group-hover:scale-110",
                        isActive ? "text-agri-green" : "text-slate-400 group-hover:text-slate-400"
                      )} />
                      {item.name}
                    </>
                  )}
                </NavLink>
              );
            })}
          </div>

          <div className="p-4 border-t border-slate-800 mt-auto">
            <button
              onClick={handleLogout}
              className="flex w-full items-center justify-center px-4 py-3 text-slate-400 text-sm font-medium transition-all duration-200 rounded-xl hover:bg-red-50 hover:text-red-700 group"
            >
              <LogOut className="w-4 h-4 mr-2 transition-transform group-hover:-translate-x-1" />
              Sign Out
            </button>
          </div>
        </>
      )}
    </aside>
  );
}
