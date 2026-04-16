import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { motion } from 'framer-motion';
import { useEffect, useRef } from 'react';
import GlassBreakEffect from '../common/GlassBreakEffect';
import Footer from './Footer';

export function AppLayout() {
  const location = useLocation();
  const mainRef = useRef(null);
  const isHomeRoute = location.pathname === '/home';
  const isCreatorRoute = location.pathname === '/developer';
  const isFullScreenRoute = isHomeRoute || isCreatorRoute;

  useEffect(() => {
    if (mainRef.current) {
      mainRef.current.scrollTo(0, 0);
    }
  }, [location.pathname]);

  return (
    <div className="flex h-screen w-full relative overflow-hidden bg-[#0a0a0f]">
      {/* 3D Glass Breaking Effect */}
      <GlassBreakEffect />

      {/* Background ambient glow effect using mix-blend-mode over the Unsplash image */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-neon-green/10 blur-[150px] mix-blend-screen pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-neon-blue/10 blur-[150px] mix-blend-screen pointer-events-none" />

      <div className={`fixed z-50 print:hidden 
        ${isCreatorRoute ? 'opacity-0 pointer-events-none scale-95' : 'opacity-100 pointer-events-auto scale-100'}
        ${isHomeRoute ? 'top-8 left-1/2 -translate-x-1/2' : 'left-0 top-0 h-full'}
      `}>
        <Sidebar isHomeRoute={isHomeRoute} />
      </div>

      <main ref={mainRef} className="flex-1 relative z-10 w-full h-full overflow-y-auto">
        {/* Mobile Header would go here */}
        <div className={isFullScreenRoute ? "w-full h-full" : "py-6 md:py-10 px-4 md:px-8 w-full max-w-[1400px] mx-auto"}>
          <motion.div
            className={isFullScreenRoute ? "w-full min-h-[100vh] flex flex-col" : ""}
            initial={{ y: 10 }}
            animate={{ y: 0 }}
            exit={{ y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <Outlet />
          </motion.div>
        </div>
        {!isFullScreenRoute && <Footer />}
      </main>
    </div>
  );
}
