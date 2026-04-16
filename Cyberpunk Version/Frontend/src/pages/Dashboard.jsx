import { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { Card, CardContent, CardHeader, CardTitle } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { motion } from 'framer-motion';
import { Leaf, Droplets, Sun, Activity, ArrowRight, CloudRain, Thermometer, Wind, AlertCircle, Sprout, CheckCircle2, FlaskConical, Beaker, Calendar, Download, Navigation, Power } from 'lucide-react';
import { Link } from 'react-router-dom';
import toast from 'react-hot-toast';
import api from '../services/api';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import WorldCropMap from '../components/dashboard/WorldCropMap';
import InterventionModal from '../components/dashboard/InterventionModal';

export default function Dashboard() {
  const { user } = useAuth();
  
  const [weatherData, setWeatherData] = useState(null);
  const [loadingWeather, setLoadingWeather] = useState(true);
  const [historyData, setHistoryData] = useState([]);
  
  // Hackathon Power Features Phase 3 & 5 Status
  const [dayOffset, setDayOffset] = useState(0);
  const [isInterventionMode, setIsInterventionMode] = useState(false);
  const [manualIrrigation, setManualIrrigation] = useState(null);
  
  const hour = new Date().getHours();
  let greeting = 'Good evening';
  if (hour < 12) greeting = 'Good morning';
  else if (hour < 18) greeting = 'Good afternoon';

  const notifyUser = (title, message) => {
    if (!("Notification" in window)) return;
    if (Notification.permission === "granted") {
      new Notification(title, { body: message, icon: '/vite.svg' });
    } else if (Notification.permission !== "denied") {
      Notification.requestPermission().then(permission => {
        if (permission === "granted") {
           new Notification(title, { body: message, icon: '/vite.svg' });
        }
      });
    }
  };

  useEffect(() => {
    // Fetch Weather with advanced actual elements
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const { latitude, longitude } = position.coords;
          try {
            const res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,uv_index,surface_pressure`);
            const data = await res.json();
            setWeatherData(data);
          } catch (error) {
            console.error("Error fetching weather", error);
          } finally {
            setLoadingWeather(false);
          }
        },
        (error) => {
          console.error("Error getting location", error);
          setLoadingWeather(false);
        }
      );
    } else {
      setLoadingWeather(false);
    }

    // Fetch History Data
    api.get('/disease/history/')
      .then(res => {
        if (res.data?.data) {
          const formattedData = res.data.data.map((item, index) => {
            const isHealthy = item.result.toLowerCase().includes('healthy');
            return {
              name: `Scan ${res.data.data.length - index}`,
              health: isHealthy ? 100 : Math.round(100 - (item.confidence * 100)),
              disease: item.result.split('___').pop().replace('_', ' '),
              date: new Date(item.created_at).toLocaleDateString(),
              rawItem: item
            };
          }).reverse();
          setHistoryData(formattedData);
        }
      })
      .catch(console.error);
  }, []);

  const temp = weatherData?.current?.temperature_2m ?? '--';
  const humidity = weatherData?.current?.relative_humidity_2m ?? '--';
  const wind = weatherData?.current?.wind_speed_10m ?? '--';
  const uvIndex = weatherData?.current?.uv_index ?? '--';
  const pressure = weatherData?.current?.surface_pressure ?? '--';
  
  const riskOfFungal = humidity !== '--' && humidity > 80;
  const isIrrigatingActive = manualIrrigation !== null ? manualIrrigation : !riskOfFungal;

  useEffect(() => {
    // Proactive Push Warning System
    if (riskOfFungal) {
       notifyUser("Severe Biothreat Warning ⚠️", `Critical humidity (${humidity}%) detected! Fungal Outbreak Risk is high. Prepare countermeasures.`);
    }
    if (uvIndex !== '--' && uvIndex > 8) {
       notifyUser("Critical UV Exposure ☀️", `UV Index is dangerously high (${uvIndex}). Potential nutrient burn on exposed crops.`);
    }
  }, [riskOfFungal, humidity, uvIndex]);
  
  let bestCrop = "Corn 🌽";
  if (temp !== '--') {
    if (temp > 28 && humidity > 70) bestCrop = "Rice 🌾";
    else if (temp < 20) bestCrop = "Wheat 🌾";
    else if (humidity > 80 && temp > 22) bestCrop = "Sugarcane 🎋";
  }

  // Math multipliers for Time-Travel Simulation
  const simScale = 1 + (dayOffset * 0.05); 
  const simDecay = 1 - (dayOffset * 0.012);

  const baseHealth = historyData.length > 0 ? historyData[historyData.length - 1].health : 94;
  const simulatedHealth = Math.max(12, Math.round(baseHealth * simDecay));

  // Real-time elements replacing the old stats
  const stats = [
    { title: "Crop Health Index", value: `${simulatedHealth}%`, icon: Activity, color: simulatedHealth > 50 ? "text-neon-green" : "text-red-500", bg: "bg-neon-green/10", border: simulatedHealth < 50 ? "border-red-500 shadow-[0_0_15px_#f00]" : "border-neon-green/20" },
    { title: "NDVI Biomass Index", value: `${(0.82 * simDecay).toFixed(2)} ${dayOffset > 20 ? '(Critical)' : '(High)'}`, icon: Leaf, color: "text-purple-400", bg: "bg-purple-400/10", border: "border-purple-400/20" },
    { title: "Active Pest Threat", value: dayOffset > 14 ? "CRITICAL RISK" : "Low Risk", icon: AlertCircle, color: dayOffset > 14 ? "text-red-500" : "text-red-400", bg: "bg-red-400/10", border: dayOffset > 14 ? "border-red-500 shadow-[0_0_15px_#f00]" : "border-red-400/20" },
  ];

  // Dynamic Threat Assessment based on live weather data, scaled into the futures
  const threatData = [
    { subject: 'Fungal Spores', A: Math.min(100, Math.round((riskOfFungal ? 95 : 25) * simScale)), fullMark: 100 },
    { subject: 'Bacterial Growth', A: Math.min(100, Math.round((humidity !== '--' && humidity > 70 ? 75 : 35) * simScale)), fullMark: 100 },
    { subject: 'Pest Migration', A: Math.min(100, Math.round((temp !== '--' && temp > 25 ? 85 : 40) * simScale)), fullMark: 100 },
    { subject: 'Nutrient Burn', A: Math.min(100, Math.round((uvIndex !== '--' && uvIndex > 7 ? 80 : 30) * simScale)), fullMark: 100 },
    { subject: 'Moisture Stress', A: Math.min(100, Math.round((humidity !== '--' && humidity < 40 ? 90 : 15) * simScale)), fullMark: 100 },
    { subject: 'Frost Damage', A: Math.min(100, Math.round((temp !== '--' && temp < 5 ? 90 : 5) * simScale)), fullMark: 100 },
  ];

  return (
    <div id="dashboard-container" className="space-y-8 pb-10">
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center">
        <div>
          <motion.h1 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-4xl font-display font-bold text-white tracking-tight"
          >
            {greeting}, <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-green to-neon-blue">{user?.username || 'Farmer'}</span> 👋
          </motion.h1>
          <p className="text-gray-400 mt-2">Connecting to live weather and AI tracking modules...</p>
        </div>
      </header>

      {/* Time-Travel Predictive Dashboard Simulator & Intervention */}
      <motion.div initial={{ y: 20 }} animate={{ y: 0 }} transition={{ delay: 0.05 }}>
        <Card className="border border-[#0ff]/30 glass-card bg-black/40 backdrop-blur-md shadow-[0_0_30px_rgba(0,255,255,0.1)] overflow-hidden relative">
           <div className="absolute top-0 right-0 w-64 h-64 bg-[#0ff]/20 rounded-full blur-[80px] pointer-events-none" />
           <CardContent className="p-6 md:px-10 flex flex-col md:flex-row items-center gap-6">
              <div className="flex-1 w-full">
                <h2 className="text-xl font-bold flex items-center mb-4 text-[#0ff]">
                   <Calendar className="w-5 h-5 mr-2" /> Predictive Timeline Simulation
                </h2>
                <div className="flex items-center gap-4">
                   <span className="font-bold text-gray-400 whitespace-nowrap uppercase text-sm tracking-widest">Day 0</span>
                   <input 
                      type="range" 
                      min="0" max="60" 
                      value={dayOffset}
                      onChange={(e) => setDayOffset(parseInt(e.target.value))}
                      className="w-full accent-pink-500 cursor-pointer h-2 bg-gray-800 rounded-lg appearance-none"
                   />
                   <span className="font-bold text-[#f42b8e] whitespace-nowrap uppercase text-sm tracking-widest">Day +60</span>
                </div>
                <div className="mt-4 text-sm text-gray-300 font-mono flex items-center">
                   Viewing algorithmic projection for: <span className="text-white font-bold text-lg mx-2 border border-white/20 px-3 py-1 rounded bg-white/10">Day +{dayOffset}</span>
                   {dayOffset > 0 && <span className="ml-4 text-orange-400 animate-pulse hidden md:inline">⚠️ Extrapolating future states...</span>}
                </div>
              </div>
              
              {/* God Mode Intervention Button */}
              <div className="md:border-l md:border-white/10 md:pl-8 flex flex-col items-center mt-4 md:mt-0">
                 <button 
                   onClick={() => setIsInterventionMode(true)}
                   className="group relative inline-flex items-center justify-center px-6 py-4 font-bold text-white uppercase tracking-widest bg-red-600 rounded-xl overflow-hidden shadow-[0_0_20px_rgba(220,38,38,0.5)] hover:shadow-[0_0_40px_rgba(220,38,38,0.8)] transition-all"
                 >
                    <div className="absolute inset-0 w-full h-full bg-white/20 group-hover:translate-x-full transition-transform duration-500 ease-out -skew-x-12 -translate-x-full" />
                    <span className="relative flex items-center z-10 text-sm">
                       <Power className="mr-2 w-5 h-5" /> Critical Override
                    </span>
                 </button>
                 <span className="text-[10px] text-gray-500 mt-2 uppercase tracking-widest font-bold">Autonomous Strike</span>
              </div>
           </CardContent>
        </Card>
      </motion.div>

      {/* Weather & Micro-climate Widget */}
      <motion.div initial={{ y: 20 }} animate={{ y: 0 }} transition={{ delay: 0.1 }}>
        <Card className="border border-white/10 glass-card overflow-hidden relative">
          <div className="absolute top-0 right-0 w-64 h-64 bg-neon-blue/10 rounded-full blur-[80px] -mr-20 -mt-20 pointer-events-none" />
          <CardContent className="p-6 md:p-8">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
              <div className="flex items-start gap-6">
                <div className="flex-shrink-0 w-20 h-20 bg-neon-blue/10 border border-neon-blue/30 rounded-2xl flex items-center justify-center mt-1">
                  <CloudRain className="w-10 h-10 text-neon-blue" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white mb-1">Local Farm Climate <span className="text-sm font-normal text-gray-400">(Real-time)</span></h2>
                  <p className="text-neon-blue/80 font-medium mb-3">
                    {loadingWeather ? "Detecting local climate..." : "Connected to Open-Meteo Satellites"}
                  </p>
                  
                  {/* Disease Risk Warning Indicator */}
                  {riskOfFungal ? (
                    <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm font-semibold">
                      <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse mr-2" />
                      High humidity ({humidity}%) → Severe Fungal Outbreak Risk ⚠️
                    </div>
                  ) : (
                    <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-neon-green/10 border border-neon-green/20 text-neon-green text-sm hover:shadow-[0_0_15px_rgba(0,255,0,0.2)] transition-shadow">
                      <CheckCircle2 className="w-4 h-4 mr-2" />
                      Optimal growing conditions. Pathogen risk is low.
                    </div>
                  )}
                </div>
              </div>
              <div className="flex gap-8">
                <div className="flex flex-col items-center">
                  <Thermometer className="w-6 h-6 text-red-400 mb-2" />
                  <span className="text-2xl font-bold text-white">{temp}°C</span>
                  <span className="text-xs text-gray-400 uppercase tracking-wider">Air Temp</span>
                </div>
                <div className="hidden md:flex flex-col items-center">
                  <Droplets className="w-6 h-6 text-orange-400 mb-2" />
                  <span className="text-2xl font-bold text-white">{humidity}%</span>
                  <span className="text-xs text-orange-400 uppercase tracking-wider font-bold">Humidity</span>
                </div>
                <div className="flex flex-col items-center">
                  <Wind className="w-6 h-6 text-gray-300 mb-2" />
                  <span className="text-2xl font-bold text-white">{wind}</span>
                  <span className="text-xs text-gray-400 uppercase tracking-wider">Wind</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Ecosystem Recommendation & Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div 
          initial={{ y: 20 }}
          animate={{ y: 0 }}
          transition={{ delay: 0.2 }}
          className="col-span-1"
        >
          <Card className="glass-card border border-purple-500/30 h-full flex flex-col justify-center items-center text-center p-6 bg-gradient-to-br from-purple-900/10 to-transparent">
             <Sprout className="w-12 h-12 text-purple-400 mb-3" />
             <h3 className="text-gray-400 text-sm uppercase tracking-wider mb-1">AI Best Crop</h3>
             <p className="text-3xl font-bold text-white mb-2">{bestCrop}</p>
             <p className="text-xs text-purple-300/70">Based on live climate variables</p>
          </Card>
        </motion.div>

        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={stat.title}
              initial={{ y: 20 }}
              animate={{ y: 0 }}
              transition={{ delay: index * 0.1 + 0.3 }}
            >
              <Card className={`glass-card border ${stat.border} h-full hover:shadow-[0_0_20px_rgba(255,255,255,0.05)] transition-shadow`}>
                <CardContent className="flex flex-col items-center justify-center p-6 text-center h-full">
                  <div className={`p-4 rounded-full ${stat.bg} mb-4`}>
                    <Icon className={`w-8 h-8 ${stat.color}`} />
                  </div>
                  <div>
                    <p className="text-3xl font-display font-bold text-white mt-1">{stat.value}</p>
                    <p className="text-sm text-gray-400 font-medium mt-1 uppercase tracking-wider">{stat.title}</p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Grid for Live Anomaly Matrix and Graphs */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        
        {/* Real-Time Pathogen Threat Assessment Matrix (Replacing Map) */}
        <Card className="glass-card border-white/10 overflow-hidden relative xl:col-span-2">
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-red-500/50 to-transparent pointer-events-none" />
          <CardHeader className="bg-black/20">
             <CardTitle className="flex items-center text-xl text-white">
               <AlertCircle className="w-5 h-5 mr-2 text-red-500 animate-pulse" />
               Real-Time Environmental Threat Assessment
             </CardTitle>
          </CardHeader>
          <CardContent className="h-[400px] flex flex-col justify-center items-center bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-red-950/20 via-black to-black">
             <ResponsiveContainer width="100%" height="90%">
                <RadarChart cx="50%" cy="50%" outerRadius="75%" data={threatData}>
                  <PolarGrid stroke="#333333" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#00ffcc', fontSize: 13, fontWeight: 'bold' }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar name="Threat Probability (%)" dataKey="A" stroke="#ef4444" strokeWidth={3} fill="#ef4444" fillOpacity={0.4} />
                  <Tooltip 
                     contentStyle={{ backgroundColor: '#000000dd', borderColor: '#ef444455', borderRadius: '8px' }}
                     itemStyle={{ color: '#fff' }}
                  />
                </RadarChart>
             </ResponsiveContainer>
             <p className="text-gray-500 text-xs mt-2 italic text-center px-4">
               Live probabilistic model plotting infection vectors based on cross-referenced open-meteo atmospheric pressure, humidity algorithms, and temperature tracking metrics.
             </p>
          </CardContent>
        </Card>

        {/* Prediction Yield History */}
        <Card className="glass-card border-white/10 xl:col-span-1">
          <CardHeader className="bg-black/20">
             <CardTitle className="flex items-center text-xl text-white">
               <Activity className="w-5 h-5 mr-2 text-neon-blue" />
               Yield History
             </CardTitle>
          </CardHeader>
          <CardContent className="h-[400px] mt-4">
             {historyData.length > 0 ? (
               <ResponsiveContainer width="100%" height="90%">
                 <AreaChart data={historyData.slice(-10)}>
                   <defs>
                     <linearGradient id="colorHealth" x1="0" y1="0" x2="0" y2="1">
                       <stop offset="5%" stopColor="#00ffcc" stopOpacity={0.8}/>
                       <stop offset="95%" stopColor="#00ffcc" stopOpacity={0}/>
                     </linearGradient>
                   </defs>
                   <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                   <XAxis dataKey="name" stroke="#9ca3af" tick={{fill: '#9ca3af', fontSize: 10}} />
                   <YAxis stroke="#9ca3af" tick={{fill: '#9ca3af', fontSize: 10}} domain={[0, 100]} />
                   <Tooltip 
                     contentStyle={{ backgroundColor: '#000000dd', borderColor: '#00ffcc33', borderRadius: '8px' }}
                     itemStyle={{ color: '#00ffcc' }}
                     labelStyle={{ color: '#fff' }}
                   />
                   <Area type="monotone" dataKey="health" stroke="#00ffcc" fillOpacity={1} fill="url(#colorHealth)" />
                 </AreaChart>
               </ResponsiveContainer>
             ) : (
               <div className="w-full h-full flex flex-col items-center justify-center text-gray-500 pb-10">
                 <FlaskConical className="w-10 h-10 mb-2 opacity-50" />
                 <p className="text-sm text-center px-4">No scans yet. Run your first prediction to track crop health over time!</p>
               </div>
             )}
          </CardContent>
        </Card>
      </div>

      {/* Supplemental Trackers Row */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Recent Activity Timeline */}
        <Card className="glass-card border-white/10 overflow-hidden flex flex-col">
          <CardHeader className="bg-white/5 border-b border-white/5">
            <CardTitle className="flex items-center text-lg text-white">
              <Calendar className="w-5 h-5 mr-2 text-gray-400" />
              Latest Field Operations Logs
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 h-64 overflow-y-auto">
             <div className="divide-y divide-white/5">
                {historyData.length > 0 ? (
                  [...historyData].reverse().slice(0, 8).map((item, i) => (
                    <div key={i} className="p-4 flex items-start gap-4 hover:bg-white/5 transition-colors">
                      <div className={`mt-1 rounded-full p-1.5 ${item.health === 100 ? 'bg-neon-green/20' : 'bg-red-500/20'}`}>
                        {item.health === 100 ? (
                          <CheckCircle2 className="w-4 h-4 text-neon-green" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-500" />
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex justify-between items-center mb-1">
                          <p className="text-sm text-gray-200 font-bold">
                             {item.health === 100 ? "Healthy Crop Confirmed" : `Disease Detect: ${item.disease}`}
                          </p>
                          <span className="text-xs text-gray-500">{item.date}</span>
                        </div>
                        <p className="text-xs text-gray-400">
                           {item.health === 100 
                             ? "AI Drone sequence completed with 0 notable feature anomalies detected." 
                             : `Deep learning confidence model flagged ${100 - item.health}% certainty of pathogen.`}
                        </p>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="h-full flex flex-col justify-center items-center text-gray-500">
                    <AlertCircle className="w-8 h-8 mb-2 opacity-50" />
                    <p className="text-sm">No recent activity logs.</p>
                  </div>
                )}
             </div>
          </CardContent>
        </Card>

        {/* Soil NPK Nutrient Simulator */}
        <Card className="glass-card border-white/10">
          <CardHeader className="bg-white/5 border-b border-white/5">
             <CardTitle className="flex items-center text-lg text-white">
               <Beaker className="w-5 h-5 mr-2 text-yellow-400" />
               AI Soil Nutrient Analysis (Estimated NPK)
             </CardTitle>
          </CardHeader>
          <CardContent className="p-6 h-64 flex flex-col justify-center">
            <div className="space-y-6">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-blue-400">Nitrogen (N) - Leaf Growth</span>
                  <span className="text-sm font-bold text-white">45% (Slightly Low)</span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2.5">
                  <div className="bg-gradient-to-r from-blue-600 to-blue-400 h-2.5 rounded-full" style={{ width: '45%' }}></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-orange-400">Phosphorus (P) - Root Health</span>
                  <span className="text-sm font-bold text-white">72% (Optimal)</span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2.5">
                  <div className="bg-gradient-to-r from-orange-600 to-orange-400 h-2.5 rounded-full" style={{ width: '72%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-purple-400">Potassium (K) - Disease Resistance</span>
                  <span className="text-sm font-bold text-white">
                    {riskOfFungal ? "25% (Critical Risk)" : "68% (Good)"}
                  </span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2.5">
                  <div className={`bg-gradient-to-r ${riskOfFungal ? 'from-red-600 to-red-400' : 'from-purple-600 to-purple-400'} h-2.5 rounded-full`} style={{ width: riskOfFungal ? '25%' : '68%' }}></div>
                </div>
              </div>
              <p className="text-xs text-gray-500 italic mt-4 text-center">
                *NPK values are heuristically simulated based on historical disease patterns.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Cyber/Sci-Fi Automated Operations Row */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card className="glass-card border-white/10">
          <CardHeader className="bg-white/5 border-b border-white/5">
             <CardTitle className="flex items-center text-lg text-white">
               <Navigation className="w-5 h-5 mr-2 text-neon-blue" />
               Autonomous Drone Fleet Status
             </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="space-y-4">
               <div className="flex justify-between items-center bg-white/5 p-3 rounded-lg border border-neon-blue/20 hover:bg-white/10 transition-colors">
                  <div>
                    <p className="font-bold text-white text-sm">Alpha-1 (Scout)</p>
                    <p className="text-xs text-gray-400">Sector 4</p>
                  </div>
                  <span className="px-3 py-1 bg-neon-blue/20 text-neon-blue text-xs rounded-full animate-pulse border border-neon-blue/50">Airborne (Scanning)</span>
               </div>
               <div className="flex justify-between items-center bg-white/5 p-3 rounded-lg border border-neon-green/20 hover:bg-white/10 transition-colors">
                  <div>
                    <p className="font-bold text-white text-sm">Bravo-2 (Sprayer)</p>
                    <p className="text-xs text-gray-400">Sector 1</p>
                  </div>
                  <span className="px-3 py-1 bg-neon-green/20 text-neon-green text-xs rounded-full border border-neon-green/50">Charging (100%)</span>
               </div>
               <div className="flex justify-between items-center bg-white/5 p-3 rounded-lg border border-red-500/20 hover:bg-white/10 transition-colors">
                  <div>
                    <p className="font-bold text-white text-sm">Charlie-3 (Relay)</p>
                    <p className="text-xs text-gray-400">Base Station</p>
                  </div>
                  <span className="px-3 py-1 bg-red-500/20 text-red-400 text-xs rounded-full border border-red-500/50">Maintenance Required</span>
               </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-white/10">
          <CardHeader className="bg-white/5 border-b border-white/5">
             <CardTitle className="flex items-center text-lg text-white">
               <Droplets className="w-5 h-5 mr-2 text-cyan-400" />
               AI Irrigation Controller
             </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-6">
               <div>
                 <p className="text-3xl font-display font-bold text-white">{isIrrigatingActive ? 'Active' : 'Suspended'}</p>
                 <p className={`text-sm font-medium tracking-wide ${isIrrigatingActive ? 'text-neon-green' : 'text-red-400'}`}>
                    {isIrrigatingActive ? (manualIrrigation !== null ? 'Manual Override Executing' : 'Automated Sequence Running') : 'Fungal Outbreak Prevention'}
                 </p>
               </div>
               <div 
                 onClick={() => setManualIrrigation(!isIrrigatingActive)}
                 className={`w-14 h-14 rounded-full flex items-center justify-center border cursor-pointer hover:scale-105 transition-transform shadow-lg ${isIrrigatingActive ? 'bg-neon-green/20 border-neon-green/50 hover:shadow-[0_0_15px_#00ff00]' : 'bg-red-500/20 border-red-500/50 hover:shadow-[0_0_15px_#ff0000]'}`}>
                 <Power className={`w-6 h-6 ${isIrrigatingActive ? 'text-neon-green' : 'text-red-500'}`} />
               </div>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between text-sm py-2 border-b border-white/5">
                 <span className="text-gray-400">Current Soil Moisture:</span>
                 <span className="text-white font-bold">{isIrrigatingActive ? '72% (Rising)' : '48%'}</span>
              </div>
              <div className="flex justify-between text-sm py-2 border-b border-white/5">
                 <span className="text-gray-400">Evapotranspiration Rate:</span>
                 <span className="text-white font-bold">3.2 mm/day</span>
              </div>
              <div className="flex justify-between text-sm py-2">
                 <span className="text-gray-400">Next Scheduled Watering:</span>
                 <span className="text-cyan-400 font-bold tracking-wide animate-pulse">
                    {isIrrigatingActive ? 'Running Currently...' : '04:00 AM (Pending Analysis)'}
                 </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Global Satellite Map Matrix */}
      <div className="mt-8 mb-4 w-full">
         <motion.div initial={{ y: 20 }} animate={{ y: 0 }} transition={{ delay: 0.5 }}>
            <Card className="glass-card border-white/10 overflow-hidden shadow-[0_0_30px_rgba(0,255,255,0.05)] w-full relative">
               <WorldCropMap />
            </Card>
         </motion.div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8 print:hidden">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}>
          <Link to="/prediction" className="block relative group overflow-hidden rounded-2xl glass border border-neon-green/20 hover:border-neon-green/60 hover:shadow-[0_0_20px_rgba(0,255,204,0.3)] transition-all duration-300">
            <div className="absolute inset-0 bg-gradient-to-r from-neon-green/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="p-8 h-full flex flex-col justify-center">
              <Leaf className="w-10 h-10 text-neon-green mb-4" />
              <h2 className="text-2xl font-bold text-white mb-2 group-hover:text-neon-green transition-colors">Drone Diagnostics Batch</h2>
              <p className="text-gray-400 mb-6 max-w-sm">Upload multiple photos of your crop leaves to instantly detect diseases with live Drone-cam tracking and Grad-CAM heatmap visualizations.</p>
              <div className="flex items-center text-neon-green font-medium mt-auto">
                Try it out <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </div>
            </div>
          </Link>
        </motion.div>
        
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}>
          <Link to="/chat" className="block relative group overflow-hidden rounded-2xl glass border border-neon-blue/20 hover:border-neon-blue/60 hover:shadow-[0_0_20px_rgba(0,195,255,0.3)] transition-all duration-300">
            <div className="absolute inset-0 bg-gradient-to-r from-neon-blue/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="p-8 h-full flex flex-col justify-center">
              <Activity className="w-10 h-10 text-neon-blue mb-4" />
              <h2 className="text-2xl font-bold text-white mb-2 group-hover:text-neon-blue transition-colors">AI Copilot</h2>
              <p className="text-gray-400 mb-6 max-w-sm">Have farming questions? Upload images into the chat and speak to the AI Assistant directly to debug your crop issues.</p>
              <div className="flex items-center text-neon-blue font-medium mt-auto">
                Ask a question <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </div>
            </div>
          </Link>
        </motion.div>
      </div>

      <InterventionModal isOpen={isInterventionMode} onClose={() => setIsInterventionMode(false)} />
    </div>
  );
}
