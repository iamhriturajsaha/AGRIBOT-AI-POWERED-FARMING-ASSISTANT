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
  
  const [dayOffset, setDayOffset] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isInterventionMode, setIsInterventionMode] = useState(false);
  const [manualIrrigation, setManualIrrigation] = useState(null);

  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setDayOffset(prev => {
          if (prev >= 60) {
            setIsPlaying(false);
            return 0; // Loop back to 0 if we hit the end
          }
          return prev + 1;
        });
      }, 150);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);
  
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

  const simScale = 1 + (dayOffset * 0.05); 
  const simDecay = 1 - (dayOffset * 0.012);

  const baseHealth = historyData.length > 0 ? historyData[historyData.length - 1].health : 94;
  const simulatedHealth = Math.max(12, Math.round(baseHealth * simDecay));

  const stats = [
    { title: "Crop Health Index", value: `${simulatedHealth}%`, icon: Activity, color: simulatedHealth > 50 ? "text-agri-lightGreen" : "text-red-500", bg: "bg-agri-lightGreen/10", border: simulatedHealth < 50 ? "border-red-500 shadow-sm" : "border-slate-800" },
    { title: "NDVI Biomass Index", value: `${(0.82 * simDecay).toFixed(2)} ${dayOffset > 20 ? '(Critical)' : '(High)'}`, icon: Leaf, color: "text-blue-500", bg: "bg-blue-500/10", border: "border-slate-800" },
    { title: "Active Pest Threat", value: dayOffset > 14 ? "CRITICAL RISK" : "Low Risk", icon: AlertCircle, color: dayOffset > 14 ? "text-red-500" : "text-red-400", bg: "bg-red-50", border: dayOffset > 14 ? "border-red-500 shadow-sm" : "border-slate-800" },
  ];

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
            className="text-3xl font-bold text-white tracking-tight"
          >
            {greeting}, <span className="text-agri-green">{user?.username || 'Farmer'}</span> 👋
          </motion.h1>
          <p className="text-slate-400 mt-1">Connecting to live weather and analytics tracking modules...</p>
        </div>
      </header>

      {/* Time-Travel Predictive Dashboard Simulator & Intervention */}
      <motion.div initial={{ y: 20 }} animate={{ y: 0 }} transition={{ delay: 0.05 }}>
        <Card className="bg-slate-900/40 border-slate-800 shadow-sm overflow-hidden relative">
           <CardContent className="p-6 md:px-8 flex flex-col md:flex-row items-center gap-6">
              <div className="flex-1 w-full">
                <h2 className="text-xl font-bold flex items-center mb-4 text-white">
                   <Calendar className="w-5 h-5 mr-2 text-agri-green" /> Predictive Timeline Simulation
                </h2>
                 <div className="flex items-center gap-4">
                   <button 
                     onClick={() => setIsPlaying(!isPlaying)}
                     className={`flex-shrink-0 p-2.5 rounded-full ${isPlaying ? 'bg-red-100 text-red-600' : 'bg-agri-green/10 text-agri-green'} hover:opacity-80 transition-opacity`}
                     title={isPlaying ? "Pause Simulation" : "Play Simulation"}
                   >
                     {isPlaying ? <span className="block w-4 h-4 bg-current rounded-[2px]" /> : <span className="block w-0 h-0 border-t-[8px] border-t-transparent border-l-[12px] border-l-current border-b-[8px] border-b-transparent ml-1" />}
                   </button>
                   <span className="font-semibold text-slate-400 text-sm tracking-wide w-12 text-right">Day 0</span>
                   <input 
                      type="range" 
                      min="0" max="60" 
                      value={dayOffset}
                      onChange={(e) => { setDayOffset(parseInt(e.target.value)); setIsPlaying(false); }}
                      className="flex-1 accent-agri-green cursor-pointer h-2 bg-slate-200 rounded-lg appearance-none"
                   />
                   <span className="font-semibold text-red-500 text-sm tracking-wide w-16">Day +60</span>
                 </div>
                <div className="mt-4 text-sm text-slate-400 flex items-center font-medium">
                   Viewing algorithmic projection for: <span className="text-white font-bold text-lg mx-2 bg-slate-800/80 px-3 py-1 rounded">Day +{dayOffset}</span>
                   {dayOffset > 0 && <span className="ml-4 text-orange-500 animate-pulse hidden md:inline">⚠️ Extrapolating future states...</span>}
                </div>
              </div>
              
              {/* Emergency Intervention Button */}
              <div className="md:border-l md:border-slate-800 md:pl-8 flex flex-col items-center mt-4 md:mt-0">
                 <button 
                   onClick={() => setIsInterventionMode(true)}
                   className="inline-flex items-center justify-center px-6 py-3 font-semibold text-white bg-red-600 rounded-xl hover:bg-red-700 transition-all shadow-sm"
                 >
                    <span className="flex items-center z-10 text-sm">
                       <Power className="mr-2 w-5 h-5" /> Emergency Protocol
                    </span>
                 </button>
                 <span className="text-xs text-slate-400 mt-2 font-medium">System Override</span>
              </div>
           </CardContent>
        </Card>
      </motion.div>

      {/* Weather & Micro-climate Widget */}
      <motion.div initial={{ y: 20 }} animate={{ y: 0 }} transition={{ delay: 0.1 }}>
        <Card className="bg-slate-900/40 border-slate-800 shadow-sm overflow-hidden relative">
          <CardContent className="p-6 md:p-8">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
              <div className="flex items-start gap-6">
                <div className="flex-shrink-0 w-16 h-16 bg-blue-50 border border-blue-100 rounded-xl flex items-center justify-center mt-1">
                  <CloudRain className="w-8 h-8 text-blue-500" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white mb-1">Local Farm Climate <span className="text-sm font-normal text-slate-400">(Real-time)</span></h2>
                  <p className="text-slate-400 font-medium mb-3 text-sm">
                    {loadingWeather ? "Detecting local climate..." : "Connected to Open-Meteo Satellites"}
                  </p>
                  
                  {/* Disease Risk Warning Indicator */}
                  {riskOfFungal ? (
                    <div className="inline-flex items-center px-3 py-1.5 rounded-md bg-red-50 border border-red-200 text-red-600 text-sm font-medium">
                      <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse mr-2" />
                      High humidity ({humidity}%) → Severe Fungal Outbreak Risk
                    </div>
                  ) : (
                    <div className="inline-flex items-center px-3 py-1.5 rounded-md bg-agri-lightGreen/10 border border-agri-lightGreen/20 text-agri-green text-sm font-medium">
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
                  <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">Air Temp</span>
                </div>
                <div className="hidden md:flex flex-col items-center">
                  <Droplets className="w-6 h-6 text-blue-400 mb-2" />
                  <span className="text-2xl font-bold text-white">{humidity}%</span>
                  <span className="text-xs text-blue-500 font-medium uppercase tracking-wider">Humidity</span>
                </div>
                <div className="flex flex-col items-center">
                  <Wind className="w-6 h-6 text-slate-400 mb-2" />
                  <span className="text-2xl font-bold text-white">{wind}</span>
                  <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">Wind</span>
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
          <Card className="bg-slate-900/40 border-slate-800 shadow-sm h-full flex flex-col justify-center items-center text-center p-6">
             <Sprout className="w-10 h-10 text-agri-green mb-3" />
             <h3 className="text-slate-400 font-medium text-xs uppercase tracking-wider mb-1">AI Best Crop</h3>
             <p className="text-2xl font-bold text-white mb-1">{bestCrop}</p>
             <p className="text-xs text-slate-400">Based on live climate variables</p>
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
              <Card className={`bg-slate-900/40 border ${stat.border} shadow-sm h-full`}>
                <CardContent className="flex flex-col items-center justify-center p-6 text-center h-full">
                  <div className={`p-3 rounded-xl ${stat.bg} mb-3`}>
                    <Icon className={`w-7 h-7 ${stat.color}`} />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white mt-1">{stat.value}</p>
                    <p className="text-xs text-slate-400 font-medium mt-1 uppercase tracking-wider">{stat.title}</p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <Card className="bg-slate-900/40 border-slate-800 shadow-sm overflow-hidden relative xl:col-span-2">
          <CardHeader className="bg-slate-800/50 border-b border-slate-800">
             <CardTitle className="flex items-center text-lg text-white">
               <AlertCircle className="w-5 h-5 mr-2 text-red-500" />
               Environmental Threat Assessment
             </CardTitle>
          </CardHeader>
          <CardContent className="h-[400px] flex flex-col justify-center items-center pt-6">
             <ResponsiveContainer width="100%" height="90%">
                <RadarChart cx="50%" cy="50%" outerRadius="75%" data={threatData}>
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#475569', fontSize: 13, fontWeight: '500' }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar name="Threat Probability (%)" dataKey="A" stroke="#ef4444" strokeWidth={2} fill="#ef4444" fillOpacity={0.2} />
                  <Tooltip 
                     contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px' }}
                     itemStyle={{ color: '#0f172a' }}
                  />
                </RadarChart>
             </ResponsiveContainer>
             <p className="text-slate-400 text-xs mt-2 text-center px-4">
               Probabilistic model plotting infection vectors based on cross-referenced open-meteo metrics.
             </p>
          </CardContent>
        </Card>

        {/* Prediction Yield History */}
        <Card className="bg-slate-900/40 border-slate-800 shadow-sm xl:col-span-1 flex flex-col">
          <CardHeader className="bg-slate-800/50 border-b border-slate-800">
             <CardTitle className="flex items-center text-lg text-white">
               <Activity className="w-5 h-5 mr-2 text-agri-green" />
               Yield History
             </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 mt-6 h-[380px]">
             {historyData.length > 0 ? (
               <ResponsiveContainer width="100%" height="100%">
                 <AreaChart data={historyData.slice(-10)}>
                   <defs>
                     <linearGradient id="colorHealth" x1="0" y1="0" x2="0" y2="1">
                       <stop offset="5%" stopColor="#22c55e" stopOpacity={0.4}/>
                       <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                     </linearGradient>
                   </defs>
                   <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                   <XAxis dataKey="name" stroke="#cbd5e1" tick={{fill: '#64748b', fontSize: 10}} />
                   <YAxis stroke="#cbd5e1" tick={{fill: '#64748b', fontSize: 10}} domain={[0, 100]} />
                   <Tooltip 
                     contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px' }}
                     itemStyle={{ color: '#0f172a' }}
                     labelStyle={{ color: '#64748b' }}
                   />
                   <Area type="monotone" dataKey="health" stroke="#22c55e" strokeWidth={2} fillOpacity={1} fill="url(#colorHealth)" />
                 </AreaChart>
               </ResponsiveContainer>
             ) : (
               <div className="w-full h-full flex flex-col items-center justify-center text-slate-400">
                 <FlaskConical className="w-10 h-10 mb-2 opacity-50 text-slate-300" />
                 <p className="text-sm text-center px-4">No scans yet. Run your first prediction to track crop health over time.</p>
               </div>
             )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card className="bg-slate-900/40 border-slate-800 shadow-sm overflow-hidden flex flex-col">
          <CardHeader className="bg-slate-800/50 border-b border-slate-800">
            <CardTitle className="flex items-center text-lg text-white">
              <Calendar className="w-5 h-5 mr-2 text-slate-400" />
              Latest Field Operations Logs
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 h-64 overflow-y-auto">
             <div className="divide-y divide-slate-100">
                {historyData.length > 0 ? (
                  [...historyData].reverse().slice(0, 8).map((item, i) => (
                    <div key={i} className="p-4 flex items-start gap-4 hover:bg-slate-800/50 transition-colors">
                      <div className={`mt-1 rounded-full p-1.5 ${item.health === 100 ? 'bg-agri-lightGreen/10' : 'bg-red-50'}`}>
                        {item.health === 100 ? (
                          <CheckCircle2 className="w-4 h-4 text-agri-green" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-500" />
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex justify-between items-center mb-1">
                          <p className="text-sm text-white font-semibold">
                             {item.health === 100 ? "Healthy Crop Confirmed" : `Disease Detect: ${item.disease}`}
                          </p>
                          <span className="text-xs text-slate-400">{item.date}</span>
                        </div>
                        <p className="text-xs text-slate-400">
                           {item.health === 100 
                             ? "Diagnostics completed. No notable anomalies detected." 
                             : `Detection model flagged ${100 - item.health}% certainty of pathogen.`}
                        </p>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="h-full flex flex-col justify-center items-center text-slate-400">
                    <AlertCircle className="w-8 h-8 mb-2 opacity-30 text-slate-300" />
                    <p className="text-sm">No recent activity logs.</p>
                  </div>
                )}
             </div>
          </CardContent>
        </Card>

        {/* Soil NPK Nutrient Simulator */}
        <Card className="bg-slate-900/40 border-slate-800 shadow-sm">
          <CardHeader className="bg-slate-800/50 border-b border-slate-800">
             <CardTitle className="flex items-center text-lg text-white">
               <Beaker className="w-5 h-5 mr-2 text-blue-500" />
               Soil Nutrient Analysis (Estimated NPK)
             </CardTitle>
          </CardHeader>
          <CardContent className="p-6 h-64 flex flex-col justify-center">
            <div className="space-y-6">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-slate-400">Nitrogen (N) - Leaf Growth</span>
                  <span className="text-sm font-semibold text-white">45% (Slightly Low)</span>
                </div>
                <div className="w-full bg-slate-800/80 rounded-full h-2.5">
                  <div className="bg-blue-500 h-2.5 rounded-full" style={{ width: '45%' }}></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-slate-400">Phosphorus (P) - Root Health</span>
                  <span className="text-sm font-semibold text-white">72% (Optimal)</span>
                </div>
                <div className="w-full bg-slate-800/80 rounded-full h-2.5">
                  <div className="bg-amber-500 h-2.5 rounded-full" style={{ width: '72%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-slate-400">Potassium (K) - Disease Resistance</span>
                  <span className={`text-sm font-semibold ${riskOfFungal ? 'text-red-500' : 'text-white'}`}>
                    {riskOfFungal ? "25% (Critical Risk)" : "68% (Good)"}
                  </span>
                </div>
                <div className="w-full bg-slate-800/80 rounded-full h-2.5">
                  <div className={`${riskOfFungal ? 'bg-red-500' : 'bg-purple-500'} h-2.5 rounded-full`} style={{ width: riskOfFungal ? '25%' : '68%' }}></div>
                </div>
              </div>
              <p className="text-xs text-slate-400 mt-4 text-center">
                *NPK values are heuristically simulated based on historical disease patterns.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card className="bg-slate-900/40 border-slate-800 shadow-sm">
          <CardHeader className="bg-slate-800/50 border-b border-slate-800">
             <CardTitle className="flex items-center text-lg text-white">
               <Navigation className="w-5 h-5 mr-2 text-agri-green" />
               Drone Fleet Operations
             </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="space-y-4">
               <div className="flex justify-between items-center bg-slate-800/50 p-3 rounded-lg border border-slate-100">
                  <div>
                    <p className="font-semibold text-white text-sm">Alpha-1 (Scout)</p>
                    <p className="text-xs text-slate-400">Sector 4</p>
                  </div>
                  <span className="px-3 py-1 bg-blue-50 text-blue-600 text-xs rounded-full font-medium">Airborne (Scanning)</span>
               </div>
               <div className="flex justify-between items-center bg-slate-800/50 p-3 rounded-lg border border-slate-100">
                  <div>
                    <p className="font-semibold text-white text-sm">Bravo-2 (Sprayer)</p>
                    <p className="text-xs text-slate-400">Sector 1</p>
                  </div>
                  <span className="px-3 py-1 bg-agri-lightGreen/10 text-agri-green text-xs rounded-full font-medium">Charging (100%)</span>
               </div>
               <div className="flex justify-between items-center bg-slate-800/50 p-3 rounded-lg border border-slate-100">
                  <div>
                    <p className="font-semibold text-white text-sm">Charlie-3 (Relay)</p>
                    <p className="text-xs text-slate-400">Base Station</p>
                  </div>
                  <span className="px-3 py-1 bg-red-50 text-red-600 text-xs rounded-full font-medium">Maintenance</span>
               </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/40 border-slate-800 shadow-sm">
          <CardHeader className="bg-slate-800/50 border-b border-slate-800">
             <CardTitle className="flex items-center text-lg text-white">
               <Droplets className="w-5 h-5 mr-2 text-blue-500" />
               Irrigation Controller
             </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-6">
               <div>
                 <p className="text-3xl font-bold text-white">{isIrrigatingActive ? 'Active' : 'Suspended'}</p>
                 <p className={`text-sm font-medium mt-1 ${isIrrigatingActive ? 'text-agri-green' : 'text-red-500'}`}>
                    {isIrrigatingActive ? (manualIrrigation !== null ? 'Manual Override Executing' : 'Automated Sequence Running') : 'Prevention Halt'}
                 </p>
               </div>
               <div 
                 onClick={() => setManualIrrigation(!isIrrigatingActive)}
                 className={`w-14 h-14 rounded-full flex items-center justify-center border cursor-pointer hover:shadow-md transition-shadow ${isIrrigatingActive ? 'bg-agri-lightGreen/10 border-agri-lightGreen/30' : 'bg-red-50 border-red-200'}`}>
                 <Power className={`w-6 h-6 ${isIrrigatingActive ? 'text-agri-green' : 'text-red-500'}`} />
               </div>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between text-sm py-2 border-b border-slate-100">
                 <span className="text-slate-400">Current Soil Moisture:</span>
                 <span className="text-white font-semibold">{isIrrigatingActive ? '72% (Rising)' : '48%'}</span>
              </div>
              <div className="flex justify-between text-sm py-2 border-b border-slate-100">
                 <span className="text-slate-400">Evapotranspiration Rate:</span>
                 <span className="text-white font-semibold">3.2 mm/day</span>
              </div>
              <div className="flex justify-between text-sm py-2">
                 <span className="text-slate-400">Next Scheduled Watering:</span>
                 <span className="text-blue-500 font-semibold">
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
            <Card className="bg-slate-900/40 border-slate-800 shadow-sm overflow-hidden w-full relative">
               <WorldCropMap />
            </Card>
         </motion.div>
      </div>

      <InterventionModal isOpen={isInterventionMode} onClose={() => setIsInterventionMode(false)} />
    </div>
  );
}
