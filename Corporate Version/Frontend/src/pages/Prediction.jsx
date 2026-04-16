import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, CheckCircle2, AlertTriangle, RefreshCcw, Loader2, Leaf, Layers, Calendar, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '../components/common/Card';
import { Button } from '../components/common/Button';
import api from '../services/api';
// import Footer from '../components/layout/Footer';

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
        <h1 className="text-3xl font-bold flex items-center text-white tracking-tight">
           <Layers className="w-8 h-8 mr-3 text-agri-green" />
           AI Crop Diagnostic & Batch Analysis
        </h1>
        <p className="text-slate-400 mt-2">Upload up to 10 photos of crop leaves. Our AI will analyze them and generate detailed diagnostic reports with Grad-CAM heatmaps.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Upload Area */}
        <div className="lg:col-span-4 space-y-4">
          <div
            className={`relative flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer ${
              isDragging 
                ? 'border-agri-green bg-agri-lightGreen/10' 
                : files.length > 0
                  ? 'border-slate-700 bg-slate-800/50' 
                  : 'border-slate-700 bg-slate-900/40 hover:border-agri-green hover:bg-slate-800/50'
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
              <div className="w-full text-center py-4">
                <div className="flex flex-wrap gap-2 justify-center mb-4 max-h-40 overflow-hidden px-4">
                  {previews.map((preview, idx) => (
                    <div key={idx} className="relative w-20 h-20 rounded-lg overflow-hidden border border-slate-800">
                       <img src={preview} alt="Scanning" className="w-full h-full object-cover opacity-50" />
                       <motion.div 
                         className="absolute inset-x-0 h-1 bg-agri-green shadow-[0_0_8px_#16a34a]"
                         animate={{ top: ['0%', '100%', '0%'] }}
                         transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
                       />
                    </div>
                  ))}
                </div>
                <div className="flex items-center justify-center text-agri-green font-semibold text-sm animate-pulse">
                  <span className="w-2 h-2 bg-agri-green rounded-full mr-2"></span>
                  Processing Images
                </div>
                <p className="text-slate-400 text-xs mt-1">Extracting feature vectors...</p>
              </div>
            ) : files.length > 0 ? (
              <div className="w-full text-center">
                <div className="flex flex-wrap gap-2 justify-center mb-4 max-h-40 overflow-y-auto">
                  {previews.map((preview, idx) => (
                    <img key={idx} src={preview} alt="Preview" className="w-16 h-16 object-cover rounded-lg border border-slate-800 shadow-sm" />
                  ))}
                </div>
                <p className="text-white font-semibold mb-4">{files.length} Image(s) Selected</p>
                <Button variant="outline" onClick={(e) => { e.stopPropagation(); resetForm(); }} className="w-full justify-center">
                  <RefreshCcw className="w-4 h-4 mr-2" /> Start Over
                </Button>
              </div>
            ) : (
              <div className="text-center pointer-events-none">
                <div className="w-16 h-16 bg-blue-50 bg-opacity-50 rounded-full flex items-center justify-center mx-auto mb-4 border border-blue-100">
                  <UploadCloud className="w-8 h-8 text-blue-500" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-1">Drag & Drop Images</h3>
                <p className="text-sm text-slate-400">or click to browse</p>
                <p className="text-xs text-slate-400 mt-4">Up to 10 images (JPG, PNG)</p>
              </div>
            )}
          </div>
          
          <Button 
            className="w-full" 
            size="lg" 
            disabled={files.length === 0 || isAnalyzing || results} 
            onClick={analyzeImages}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="animate-spin w-5 h-5 mr-2" /> 
                Analyzing...
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
                className="absolute inset-0 flex items-center justify-center border-2 border-dashed border-slate-800 rounded-2xl bg-slate-800/50"
              >
                <div className="text-center text-slate-400 px-6">
                  <Layers className="w-12 h-12 mx-auto mb-4 text-slate-300" />
                  <p>Upload and analyze images to reveal AI diagnostic insights.</p>
                </div>
              </motion.div>
            )}

            {isAnalyzing && (
              <motion.div
                key="loading"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex flex-col items-center justify-center border border-slate-800 rounded-2xl bg-slate-900/40 shadow-sm"
              >
                <div className="w-16 h-16 relative mb-6">
                  <Loader2 className="w-16 h-16 text-agri-green animate-spin" />
                </div>
                <h3 className="text-xl font-bold text-white">Generating Diagnostics</h3>
                <p className="text-sm text-slate-400 mt-2">Computing Grad-CAM heatmaps...</p>
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
                  <Card key={idx} className={`border ${res.isHealthy !== false && res.disease?.toLowerCase() === 'healthy' ? 'border-green-200' : 'border-red-200'} bg-slate-900/40 overflow-hidden shadow-sm`}>
                    <CardContent className="p-0 flex flex-col md:flex-row">
                      {/* Image Visualizer */}
                      <div className="md:w-2/5 bg-slate-800/50 flex flex-col p-4 md:border-r border-slate-800">
                         {res.heatmap_url ? (
                           <div className="w-full space-y-4 flex-1 flex flex-col justify-center">
                              <div className="relative w-full h-48 rounded-xl overflow-hidden border border-slate-800 shadow-sm">
                                <img src={previews[idx]} alt="Original" className="w-full h-full object-cover" />
                                <div className="absolute top-2 left-2 bg-slate-900/40/90 px-3 py-1 rounded text-xs font-semibold text-slate-300 shadow-sm">Original Scan</div>
                              </div>
                              <div className="relative w-full h-48 rounded-xl overflow-hidden border border-slate-800 shadow-sm">
                                <img src={res.heatmap_url} alt="Grad-CAM Heatmap" className="w-full h-full object-cover" />
                                <div className="absolute top-2 left-2 bg-slate-900/40/90 px-3 py-1 rounded text-xs font-semibold text-red-600 shadow-sm">AI Heatmap</div>
                              </div>
                           </div>
                         ) : (
                           <div className="relative w-full h-full min-h-[250px] rounded-xl overflow-hidden border border-slate-800 shadow-sm">
                             <img src={previews[idx]} alt="Input preview" className="w-full h-full object-cover" />
                           </div>
                         )}
                      </div>

                      {/* Analysis Text */}
                      <div className="md:w-3/5 p-6 md:p-8 flex flex-col justify-center">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                              <h2 className="text-2xl font-bold text-white mb-1">{res.crop || 'Plant'}</h2>
                              <p className="text-slate-400 font-medium capitalize">{res.disease?.replace('_', ' ') || 'Unknown condition'}</p>
                            </div>
                            <div className={`p-3 rounded-xl ${res.disease?.toLowerCase() === 'healthy' ? 'bg-green-50 text-green-600' : 'bg-red-50 text-red-600'}`}>
                              {res.disease?.toLowerCase() === 'healthy' ? <CheckCircle2 className="w-8 h-8" /> : <AlertTriangle className="w-8 h-8" />}
                            </div>
                        </div>

                        {res.confidence && (
                            <div className="mb-8">
                              <div className="flex justify-between text-sm mb-2">
                                <span className="text-slate-400 font-medium">Diagnostic Confidence</span>
                                <span className={`font-bold ${res.confidence > 90 ? 'text-green-600' : 'text-amber-500'}`}>{res.confidence}%</span>
                              </div>
                              <div className="w-full h-2 bg-slate-800/80 rounded-full overflow-hidden">
                                <motion.div 
                                  initial={{ width: 0 }} 
                                  animate={{ width: `${res.confidence}%` }} 
                                  transition={{ duration: 1.5, ease: "easeOut" }}
                                  className={`h-full ${res.disease?.toLowerCase() === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`}
                                />
                              </div>
                            </div>
                        )}

                        {res.treatment && typeof res.treatment === 'object' ? (
                          <div className="rounded-xl border border-slate-700 bg-slate-900/40 overflow-hidden shadow-sm">
                            <div className="p-5">
                               <h4 className="flex items-center text-agri-lightGreen font-bold tracking-wide mb-4 border-b border-slate-700 pb-2">
                                 <Layers className="w-5 h-5 mr-2 text-agri-green" /> AI Diagnostic Report
                               </h4>
                               <ul className="space-y-4 text-sm">
                                 <li className="flex flex-col">
                                   <span className="text-slate-400 font-semibold uppercase text-xs mb-1">Causal Pathogen</span>
                                   <span className="text-white leading-relaxed font-medium">{res.treatment.cause}</span>
                                 </li>
                                 <li className="flex flex-col">
                                   <span className="text-slate-400 font-semibold uppercase text-xs mb-1">Recommended Action</span>
                                   <span className="text-white leading-relaxed font-medium">{res.treatment.action}</span>
                                 </li>
                                 <li className="flex items-center justify-between pt-3 border-t border-slate-700">
                                   <span className="text-slate-400 font-semibold uppercase text-xs">Assessed Risk Level</span>
                                   <span className={`px-2 py-1 rounded text-xs font-bold ${res.treatment.risk_level === 'High' || res.treatment.risk_level === 'Critical' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                                     {res.treatment.risk_level}
                                   </span>
                                 </li>
                               </ul>
                            </div>
                          </div>
                        ) : res.treatment && (
                           <p className="text-slate-400 text-sm">{String(res.treatment)}</p>
                        )}
                        {res.error && (
                           <div className="p-4 bg-red-50 text-red-600 rounded-lg border border-red-200 mt-4">Error: {res.error}</div>
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
