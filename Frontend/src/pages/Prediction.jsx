import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, CheckCircle2, AlertTriangle, RefreshCcw, Loader2, Leaf, Layers, Calendar, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '../components/common/Card';
import { Button } from '../components/common/Button';
import api from '../services/api';

export default function Prediction() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [historyData, setHistoryData] = useState([]);
  const fileInputRef = useRef(null);

  const fetchLogs = () => {
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
            };
          }).reverse();
          setHistoryData(formattedData);
        }
      })
      .catch(console.error);
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFiles = (selectedFiles) => {
    const validFiles = selectedFiles.filter(f => f.type.startsWith('image/')).slice(0, 10);
    if (validFiles.length === 0) {
      alert('Please upload valid image files (Max 10)');
      return;
    }
    setFiles(validFiles);
    
    // Create previews
    const newPreviews = validFiles.map(file => URL.createObjectURL(file));
    setPreviews(newPreviews);
    setResults(null);
  };

  const analyzeImages = async () => {
    if (files.length === 0) return;
    setIsAnalyzing(true);
    
    const formData = new FormData();
    files.forEach(file => {
      formData.append('images', file);
    });

    try {
      if (files.length === 1) {
        formData.delete('images');
        formData.append('image', files[0]);
        const res = await api.post('/disease/predict/', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setResults([res.data.data]);
      } else {
        const res = await api.post('/disease/predict/batch/', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setResults(res.data.data);
      }
      fetchLogs(); // Sync logs after prediction
    } catch (error) {
      console.error(error);
      const serverMessage = error.response?.data?.message || error.response?.data?.detail || error.message;
      alert(`Failed to analyze images: ${serverMessage}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetForm = () => {
    setFiles([]);
    previews.forEach(p => URL.revokeObjectURL(p));
    setPreviews([]);
    setResults(null);
  };

  return (
    <div className="max-w-6xl mx-auto pb-10">
      <div className="mb-8">
        <h1 className="text-3xl font-display font-bold flex items-center text-white">
           <Layers className="w-8 h-8 mr-3 text-neon-green" />
           AI Crop Diagnostic & Batch Analysis
        </h1>
        <p className="text-gray-400 mt-2">Upload up to 10 photos of crop leaves. Our AI will analyze them and generate Grad-CAM heatmaps showing exactly where the disease is located.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Upload Area */}
        <div className="lg:col-span-4 space-y-4">
          <div
            className={`relative flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-2xl transition-all duration-300 ${
              isDragging 
                ? 'border-neon-green bg-neon-green/5' 
                : files.length > 0
                  ? 'border-panelBorder bg-panel/50' 
                  : 'border-panelBorder bg-panel hover:border-neon-green/50 hover:bg-white/5'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => files.length === 0 && fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              multiple
              onChange={(e) => e.target.files && handleFiles(Array.from(e.target.files))}
            />

            {isAnalyzing ? (
              <div className="w-full text-center relative overflow-hidden py-4">
                <div className="flex flex-wrap gap-2 justify-center mb-4 max-h-40 overflow-hidden relative px-4">
                  {previews.map((preview, idx) => (
                    <div key={idx} className="relative w-20 h-20 rounded-lg border border-neon-green/50 overflow-hidden shadow-[0_0_15px_rgba(0,255,0,0.3)]">
                       <img src={preview} alt="Scanning" className="w-full h-full object-cover opacity-60" />
                       <motion.div 
                         className="absolute inset-0 border-t-2 border-neon-green bg-gradient-to-b from-neon-green/40 to-transparent shadow-[0_4px_10px_rgba(0,255,0,0.5)]"
                         animate={{ y: ['-100%', '100%'] }}
                         transition={{ repeat: Infinity, duration: 1.5, ease: 'linear' }}
                       />
                       <div className="absolute top-1 left-1 w-2 h-2 border-t-2 border-l-2 border-neon-green/80" />
                       <div className="absolute top-1 right-1 w-2 h-2 border-t-2 border-r-2 border-neon-green/80" />
                       <div className="absolute bottom-1 left-1 w-2 h-2 border-b-2 border-l-2 border-neon-green/80" />
                       <div className="absolute bottom-1 right-1 w-2 h-2 border-b-2 border-r-2 border-neon-green/80" />
                    </div>
                  ))}
                </div>
                <div className="flex items-center justify-center text-neon-green font-mono text-sm tracking-widest uppercase animate-pulse">
                  <span className="w-2 h-2 bg-neon-green rounded-full mr-2"></span>
                  Live Drone Feed Active
                </div>
                <p className="text-gray-400 text-xs mt-1">Extracting multispectral feature vectors...</p>
              </div>
            ) : files.length > 0 ? (
              <div className="w-full text-center">
                <div className="flex flex-wrap gap-2 justify-center mb-4 max-h-40 overflow-y-auto">
                  {previews.map((preview, idx) => (
                    <img key={idx} src={preview} alt="Preview" className="w-16 h-16 object-cover rounded-lg border border-white/10" />
                  ))}
                </div>
                <p className="text-neon-green font-bold mb-4">{files.length} Image(s) Select</p>
                <Button variant="ghost" onClick={(e) => { e.stopPropagation(); resetForm(); }} className="w-full">
                  <RefreshCcw className="w-4 h-4 mr-2" /> Start Over
                </Button>
              </div>
            ) : (
              <div className="text-center cursor-pointer pointer-events-none">
                <div className="w-20 h-20 bg-neon-green/10 rounded-full flex items-center justify-center mx-auto mb-4 border border-neon-green/20 hover:scale-110 transition-transform">
                  <UploadCloud className="w-10 h-10 text-neon-green" />
                </div>
                <h3 className="text-lg font-medium text-white mb-2">Drag & Drop Images</h3>
                <p className="text-sm text-gray-400">or click to browse</p>
                <p className="text-xs text-gray-500 mt-4">Up to 10 images (JPG, PNG)</p>
              </div>
            )}
          </div>
          
          <Button 
            className="w-full relative overflow-hidden group" 
            size="lg" 
            disabled={files.length === 0 || isAnalyzing || results} 
            onClick={analyzeImages}
          >
            <div className="absolute inset-0 bg-white/20 group-hover:bg-transparent transition-colors" />
            {isAnalyzing ? (
              <>
                <Loader2 className="animate-spin w-5 h-5 mr-2" /> 
                Deep Learning Analysis...
              </>
            ) : results ? (
              'Batch Analysis Complete'
            ) : (
              'Analyze Images'
            )}
          </Button>
        </div>

        {/* Results Area */}
        <div className="lg:col-span-8 relative min-h-[400px]">
          <AnimatePresence mode="wait">
            {!results && !isAnalyzing && (
              <motion.div 
                key="empty"
                initial={{ opacity: 0 }} 
                animate={{ opacity: 1 }} 
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex items-center justify-center border-2 border-dashed border-panelBorder rounded-2xl bg-panel/30"
              >
                <div className="text-center text-gray-500 px-6">
                  <Layers className="w-12 h-12 mx-auto mb-4 opacity-20" />
                  <p>Upload and analyze images to reveal AI Grad-CAM insights.</p>
                </div>
              </motion.div>
            )}

            {isAnalyzing && (
              <motion.div
                key="loading"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex flex-col items-center justify-center border border-panelBorder rounded-2xl bg-panel shadow-[0_0_50px_rgba(0,255,204,0.1)]"
              >
                <div className="w-24 h-24 relative mb-6">
                  <div className="absolute inset-0 rounded-full border-t-2 border-neon-green animate-spin" />
                  <div className="absolute inset-2 rounded-full border-r-2 border-neon-blue animate-[spin_1.5s_linear_reverse_infinite]" />
                  <div className="absolute inset-4 rounded-full border-l-2 border-neon-pink animate-[spin_2s_linear_infinite]" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Leaf className="w-8 h-8 text-neon-green animate-pulse" />
                  </div>
                </div>
                <h3 className="text-lg font-bold text-white tracking-widest animate-pulse">GENERATING GRAD-CAM</h3>
                <p className="text-sm text-neon-blue/70 mt-2">Computing gradients for visualization...</p>
              </motion.div>
            )}

            {results && (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {results.map((res, idx) => (
                  <Card key={idx} className={`border ${res.isHealthy !== false && res.disease?.toLowerCase() === 'healthy' ? 'border-neon-green/30' : 'border-red-500/30'} overflow-hidden shadow-2xl`}>
                    <CardContent className="p-0 flex flex-col md:flex-row">
                      {/* Image Visualizer */}
                      <div className="md:w-2/5 bg-black/40 flex flex-col p-4 md:border-r border-white/10">
                         {res.heatmap_url ? (
                           <div className="w-full space-y-4 flex-1 flex flex-col justify-center">
                              <div className="relative w-full h-48 rounded-xl overflow-hidden border border-white/10 shadow-lg">
                                <img src={previews[idx]} alt="Original" className="w-full h-full object-cover" />
                                <div className="absolute top-2 left-2 bg-black/80 px-3 py-1 rounded-md text-[10px] text-gray-300 font-bold uppercase tracking-widest backdrop-blur-md shadow-sm border border-white/10">Original Scan</div>
                              </div>
                              <div className="relative w-full h-48 rounded-xl overflow-hidden border border-red-500/40 shadow-[0_0_20px_rgba(255,0,0,0.15)]">
                                <img src={res.heatmap_url} alt="Grad-CAM Heatmap" className="w-full h-full object-cover" />
                                <div className="absolute top-2 left-2 bg-red-950/80 px-3 py-1 rounded-md text-[10px] text-red-400 font-bold uppercase tracking-widest backdrop-blur-md shadow-sm border border-red-500/30">AI Threat Heatmap</div>
                              </div>
                           </div>
                         ) : (
                           <div className="relative w-full h-full min-h-[250px] rounded-xl overflow-hidden border border-white/10">
                             <img src={previews[idx]} alt="Input preview" className="w-full h-full object-cover" />
                           </div>
                         )}
                      </div>

                      {/* Analysis Text */}
                      <div className="md:w-3/5 p-6 md:p-8 flex flex-col justify-center">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                              <h2 className="text-3xl font-bold text-white mb-1">{res.crop || 'Plant'}</h2>
                              <p className="text-gray-400 capitalize">{res.disease?.replace('_', ' ') || 'Unknown condition'}</p>
                            </div>
                            <div className={`p-4 rounded-xl ${res.disease?.toLowerCase() === 'healthy' ? 'bg-neon-green/10 text-neon-green' : 'bg-red-500/10 text-red-500'}`}>
                              {res.disease?.toLowerCase() === 'healthy' ? <CheckCircle2 className="w-8 h-8" /> : <AlertTriangle className="w-8 h-8 animate-pulse" />}
                            </div>
                        </div>

                        {res.confidence && (
                            <div className="mb-8">
                              <div className="flex justify-between text-sm mb-2">
                                <span className="text-gray-400 font-medium">Neural Network Confidence</span>
                                <span className={`font-bold ${res.confidence > 90 ? 'text-neon-green' : 'text-yellow-400'}`}>{res.confidence}%</span>
                              </div>
                              <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                                <motion.div 
                                  initial={{ width: 0 }} 
                                  animate={{ width: `${res.confidence}%` }} 
                                  transition={{ duration: 1.5, ease: "easeOut" }}
                                  className={`h-full ${res.disease?.toLowerCase() === 'healthy' ? 'bg-neon-green' : 'bg-gradient-to-r from-red-500 to-orange-500 shadow-[0_0_10px_rgba(255,0,0,0.5)]'}`}
                                />
                              </div>
                            </div>
                        )}

                        {res.treatment && typeof res.treatment === 'object' ? (
                          <div className="p-1 rounded-xl bg-gradient-to-r from-neon-blue/30 via-purple-500/30 to-neon-pink/30 relative overflow-hidden group">
                            <div className="absolute inset-0 bg-black/60 group-hover:bg-black/40 transition-colors" />
                            <div className="relative p-5">
                               <h4 className="flex items-center text-neon-blue font-bold tracking-wider mb-4 border-b border-white/10 pb-2">
                                 <Leaf className="w-5 h-5 mr-2 text-neon-pink" /> 🧠 GOOGLE AI INSIGHT
                               </h4>
                               <ul className="space-y-4 text-sm">
                                 <li className="flex flex-col">
                                   <span className="text-gray-400 font-bold uppercase tracking-wider text-[10px] mb-1">Causal Pathogen</span>
                                   <span className="text-white leading-relaxed font-medium">{res.treatment.cause}</span>
                                 </li>
                                 <li className="flex flex-col">
                                   <span className="text-gray-400 font-bold uppercase tracking-wider text-[10px] mb-1">Recommended Action</span>
                                   <span className="text-neon-green leading-relaxed font-medium">{res.treatment.action}</span>
                                 </li>
                                 <li className="flex items-center justify-between pt-2 border-t border-white/5">
                                   <span className="text-gray-400 font-bold uppercase tracking-wider text-[10px]">Assessed Risk Level</span>
                                   <span className={`px-3 py-1 rounded shadow-sm text-xs font-bold ${res.treatment.risk_level === 'High' || res.treatment.risk_level === 'Critical' ? 'bg-red-500 text-white' : 'bg-neon-green/20 text-neon-green'}`}>
                                     {res.treatment.risk_level}
                                   </span>
                                 </li>
                               </ul>
                            </div>
                          </div>
                        ) : res.treatment && (
                           <p className="text-gray-300 text-sm">{String(res.treatment)}</p>
                        )}
                        {res.error && (
                           <div className="p-4 bg-red-500/10 text-red-500 rounded border border-red-500/30">Error: {res.error}</div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

    </div>
  );
}
