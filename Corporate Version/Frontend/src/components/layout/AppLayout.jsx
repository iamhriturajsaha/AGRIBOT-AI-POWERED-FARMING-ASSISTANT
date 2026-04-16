import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { motion } from 'framer-motion';
import { useEffect, useRef } from 'react';
import Footer from './Footer';

export function AppLayout() {
  const location = useLocation();
  const mainRef = useRef(null);
  const isHomeRoute = location.pathname === '/home';
  
  // All pages now use the centralized full screen layout visually (no left sidebar)
  // We still provide standard padding for non-home pages
  const isFullScreenRoute = isHomeRoute;

  useEffect(() => {
    if (mainRef.current) {
      mainRef.current.scrollTo(0, 0);
    }
  }, [location.pathname]);

  return (
    <div className="flex h-screen w-full relative overflow-hidden bg-transparent">
      <div className="fixed z-50 print:hidden top-8 left-1/2 -translate-x-1/2">
        <Sidebar isHomeRoute={true} />
      </div>

      <main ref={mainRef} className="flex-1 relative z-10 w-full h-full overflow-y-auto overflow-x-hidden">
        <div className={isFullScreenRoute ? "w-full h-full" : "pt-24 pb-10 px-4 md:px-8 mx-auto xl:max-w-7xl"}>
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
        {!isHomeRoute && <Footer />}
      </main>
    </div>
  );
}
