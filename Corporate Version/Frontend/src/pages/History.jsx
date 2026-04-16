import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Clock, Loader2, Image as ImageIcon, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { Card, CardContent } from '../components/common/Card';
import api from '../services/api';

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await api.get('/disease/history/');
        setHistory(res.data.data || []);
      } catch (err) {
        console.error(err);
        setError('Failed to load diagnosis history.');
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  return (
    <div className="max-w-6xl mx-auto pb-10 pt-4">
      <div className="mb-10 flex items-center">
        <motion.div 
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="inline-flex w-16 h-16 rounded-2xl bg-blue-50 border border-blue-100 items-center justify-center mr-6"
        >
          <Clock className="w-8 h-8 text-blue-600" />
        </motion.div>
        <div>
          <h1 className="text-4xl font-bold text-white mb-2 tracking-tight">Farm Health Log</h1>
          <p className="text-slate-400">Track and review all previous crop disease predictions.</p>
        </div>
      </div>

      <motion.div
        initial={{ y: 20 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <Card className="bg-slate-900/40 border border-slate-800 shadow-sm overflow-hidden rounded-2xl">
          {loading ? (
            <div className="flex items-center justify-center h-64 text-slate-400">
              <Loader2 className="w-10 h-10 animate-spin opacity-50" />
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-64 text-red-500">
              <AlertTriangle className="w-12 h-12 mb-4 opacity-70" />
              <p>{error}</p>
            </div>
          ) : history.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400">
              <ImageIcon className="w-12 h-12 mb-4 opacity-50 text-slate-400" />
              <p>No predictions yet. Head to the prediction tool to get started.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-800/50">
                    <th className="px-6 py-4 font-semibold text-slate-400 text-sm tracking-wide">Date</th>
                    <th className="px-6 py-4 font-semibold text-slate-400 text-sm tracking-wide">Image</th>
                    <th className="px-6 py-4 font-semibold text-slate-400 text-sm tracking-wide">Crop & Diagnosis</th>
                    <th className="px-6 py-4 font-semibold text-slate-400 text-sm tracking-wide">Confidence</th>
                    <th className="px-6 py-4 font-semibold text-slate-400 text-sm tracking-wide">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {history.map((record, idx) => {
                    // Extract crop/disease from result pattern "Crop___Disease"
                    let crop = "Unknown";
                    let disease = record.result;
                    if (record.result && record.result.includes('___')) {
                      [crop, disease] = record.result.split('___');
                    }
                    const diseaseClean = disease ? disease.replace(/_/g, ' ') : 'N/A';
                    const isHealthy = diseaseClean.toLowerCase().includes('healthy');
                    
                    return (
                      <motion.tr 
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.05 + 0.3 }}
                        key={record.id} 
                        className="hover:bg-slate-800/50 transition-colors"
                      >
                        <td className="px-6 py-4 whitespace-nowrap text-slate-400 font-medium text-sm">
                          {new Date(record.created_at).toLocaleDateString()}
                        </td>
                        <td className="px-6 py-4">
                          <div className="w-12 h-12 rounded-lg bg-slate-800/80 overflow-hidden border border-slate-800 flex items-center justify-center">
                            {record.image ? (
                              <img src={record.image} alt="crop" className="w-full h-full object-cover" />
                            ) : (
                              <ImageIcon className="w-6 h-6 text-slate-400" />
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="font-semibold text-white">{crop}</div>
                          <div className="text-sm text-slate-400 capitalize">{diseaseClean}</div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center">
                            <span className="text-slate-300 font-bold mr-3 text-sm">{(record.confidence * 100).toFixed(1)}%</span>
                            <div className="w-20 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${isHealthy ? 'bg-agri-green' : 'bg-amber-500'}`} 
                                style={{ width: `${record.confidence * 100}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          {isHealthy ? (
                            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-agri-lightGreen/10 text-agri-green border border-agri-lightGreen/20">
                              <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Healthy
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-red-50 text-red-600 border border-red-200">
                              <AlertTriangle className="w-3.5 h-3.5 mr-1" /> Detected
                            </span>
                          )}
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </motion.div>
    </div>
  );
}
