import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Sparkles, Image as ImageIcon, X, ChevronDown, Globe2, ArrowRight } from 'lucide-react';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';

export default function Chatbot() {
  const { user } = useAuth();
  const [messages, setMessages] = useState([
    { id: 1, role: 'bot', text: `Hello ${user?.username || 'there'}! I'm your AgriBot AI assistant. How can I help you with your crops today?` }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const [selectedImage, setSelectedImage] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);

  const availableLanguages = [
    'English', 'Hindi', 'Bengali', 'Telugu', 'Marathi',
    'Tamil', 'Urdu', 'Gujarati', 'Kannada', 'Malayalam', 'Punjabi',
    'Spanish', 'French', 'Mandarin', 'Arabic', 'Russian',
    'Portuguese', 'German', 'Japanese', 'Korean'
  ];
  const [inputLang, setInputLang] = useState('English');
  const [outputLang, setOutputLang] = useState('English');
  const [isInputMenuOpen, setIsInputMenuOpen] = useState(false);
  const [isOutputMenuOpen, setIsOutputMenuOpen] = useState(false);

  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const quickReplies = [
    "Suggest the best fertilizer",
    "How do I prevent root rot?",
    "Analyze my latest prediction"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, previewImage]);

  useEffect(() => {
    // Clear the backend database history for this session when opening the tab
    api.delete('/chat/clear/').catch(() => { });
  }, []);

  const speak = (text) => {
    // TTS removed as per user request
  };

  const handleImageSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewImage(URL.createObjectURL(file));
      setTimeout(scrollToBottom, 100);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    if (previewImage) {
      URL.revokeObjectURL(previewImage);
    }
    setPreviewImage(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSend = async (textToSend = input) => {
    if (!textToSend.trim() && !selectedImage) return;

    const userMsg = {
      id: Date.now(),
      role: 'user',
      text: textToSend,
      image: previewImage // preserve local preview url in msg history
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    const formData = new FormData();
    if (textToSend) {
      let finalPrompt = textToSend;
      if (outputLang !== 'English' || inputLang !== 'English') {
        finalPrompt = `[CRITICAL SYSTEM COMMAND: Irrespective of any past messages or language history. The user is writing in ${inputLang}. You MUST automatically translate AND reply to this prompt strictly and exclusively in ${outputLang}. Do NOT append English explanations.]\n\nUser Question (${inputLang}): ${textToSend}`;
      }
      formData.append('message', finalPrompt);
    }
    if (selectedImage) formData.append('image', selectedImage);

    // clear attachments from input box
    const currentSelectedImage = selectedImage;
    setSelectedImage(null);
    setPreviewImage(null);

    try {
      const res = await api.post('/chat/message/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const botResponse = res.data.data.response;
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'bot', text: botResponse }]);
      speak(botResponse);
    } catch (error) {
      console.error(error);
      const errText = "Sorry, I am unable to connect to the server right now.";
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'bot', text: errText }]);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div className="h-[calc(100vh-6rem)] flex flex-col max-w-5xl mx-auto px-4 md:px-0">
      <div className="mb-6 flex flex-col md:flex-row md:justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center text-white tracking-tight">
            <Sparkles className="text-agri-green mr-3 w-8 h-8" />
            AI Copilot
          </h1>
          <p className="text-slate-400 mt-1">Chat and upload images. I can see your crops!</p>
        </div>

        {/* Multilingual Translation Module */}
        <div className="flex items-center space-x-3 w-full md:w-auto pb-2 md:pb-0">
          {/* Translate From */}
          <div className="relative min-w-[130px] z-[60]">
            <button
              onClick={() => { setIsInputMenuOpen(!isInputMenuOpen); setIsOutputMenuOpen(false); }}
              className={`w-full flex items-center justify-between px-3 py-2 border rounded-xl bg-slate-900/40 shadow-sm transition-all duration-300 ${isInputMenuOpen ? 'border-agri-green text-agri-green' : 'border-slate-700 text-slate-300 hover:border-slate-400'}`}
            >
              <div className="flex items-center gap-2">
                 <Globe2 className="w-4 h-4 text-slate-400" />
                 <span className="text-sm font-semibold tracking-wide">{inputLang}</span>
              </div>
              <ChevronDown className={`w-4 h-4 ml-2 transition-transform duration-300 ${isInputMenuOpen ? 'rotate-180' : ''}`} />
            </button>

            <AnimatePresence>
              {isInputMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  className="absolute left-0 mt-2 w-[180px] max-h-[300px] overflow-y-auto bg-slate-900/40 border border-slate-800 rounded-xl shadow-xl"
                >
                  <div className="sticky top-0 bg-slate-800/50 p-3 border-b border-slate-100 text-[11px] text-slate-400 font-bold tracking-wider uppercase flex items-center">
                    <Globe2 className="w-3 h-3 mr-2" /> Input Language
                  </div>
                  {availableLanguages.map(lang => (
                    <button
                      key={lang}
                      onClick={() => { setInputLang(lang); setIsInputMenuOpen(false); }}
                      className={`w-full text-left px-4 py-3 text-sm font-medium transition-colors ${inputLang === lang ? 'text-agri-green bg-agri-lightGreen/10' : 'text-slate-300 hover:text-agri-green hover:bg-slate-800/50'}`}
                    >
                      {lang}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-400 shrink-0" />

          {/* Translate To */}
          <div className="relative min-w-[130px] z-[60]">
            <button
              onClick={() => { setIsOutputMenuOpen(!isOutputMenuOpen); setIsInputMenuOpen(false); }}
              className={`w-full flex items-center justify-between px-3 py-2 border rounded-xl bg-slate-900/40 shadow-sm transition-all duration-300 ${isOutputMenuOpen ? 'border-agri-green text-agri-green' : 'border-slate-700 text-slate-300 hover:border-slate-400'}`}
            >
              <div className="flex items-center gap-2">
                 <Bot className="w-4 h-4 text-agri-green" />
                 <span className="text-sm font-semibold tracking-wide">{outputLang}</span>
              </div>
              <ChevronDown className={`w-4 h-4 ml-2 transition-transform duration-300 ${isOutputMenuOpen ? 'rotate-180' : ''}`} />
            </button>

            <AnimatePresence>
              {isOutputMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  className="absolute right-0 mt-2 w-[180px] max-h-[300px] overflow-y-auto bg-slate-900/40 border border-slate-800 rounded-xl shadow-xl"
                >
                  <div className="sticky top-0 bg-slate-800/50 p-3 border-b border-slate-100 text-[11px] text-slate-400 font-bold tracking-wider uppercase flex items-center">
                    <Bot className="w-3 h-3 mr-2 text-agri-green" /> Output Language
                  </div>
                  {availableLanguages.map(lang => (
                    <button
                      key={lang}
                      onClick={() => { setOutputLang(lang); setIsOutputMenuOpen(false); }}
                      className={`w-full text-left px-4 py-3 text-sm font-medium transition-colors ${outputLang === lang ? 'text-agri-green bg-agri-lightGreen/10' : 'text-slate-300 hover:text-agri-green hover:bg-slate-800/50'}`}
                    >
                      {lang}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      <Card className="flex-1 flex flex-col overflow-hidden border-slate-800 shadow-sm bg-slate-900/40">
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
          {messages.map((msg) => (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex max-w-[85%] md:max-w-[70%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center border shadow-sm ${msg.role === 'user' ? 'bg-agri-green/20 border-agri-green/40 ml-3 md:ml-4' : 'bg-blue-900/20 border-blue-800 mr-3 md:mr-4'
                  }`}>
                  {msg.role === 'user' ? <User className="w-5 h-5 text-agri-green" /> : <Bot className="w-5 h-5 text-blue-500" />}
                </div>

                <div className={`p-4 rounded-2xl shadow-sm ${msg.role === 'user'
                    ? 'bg-agri-green/20 text-white border border-agri-green/40 rounded-tr-none'
                    : 'bg-slate-900/40 text-white border border-slate-800 rounded-tl-none'
                  }`}>
                  {msg.image && (
                    <div className="mb-3">
                      <img src={msg.image} alt="User upload" className="max-w-[200px] rounded-lg border border-slate-800 shadow-sm" />
                    </div>
                  )}
                  {msg.role === 'bot' && msg.text.includes('AI Insight:') ? (
                    <div>
                      <div className="mb-3 whitespace-pre-wrap leading-relaxed">{msg.text.split('AI Insight:')[0]}</div>
                      <div className="bg-blue-50/50 border border-blue-100 p-3 rounded-lg mt-2">
                        <div className="font-semibold text-blue-700 flex items-center mb-1 text-sm"><Sparkles className="w-4 h-4 mr-2" /> AI Insight</div>
                        <div className="whitespace-pre-wrap text-sm text-slate-300">{msg.text.split('AI Insight:')[1].trim()}</div>
                      </div>
                    </div>
                  ) : (
                    <p className="leading-relaxed whitespace-pre-wrap">{msg.text.replace(/\*\*/g, '')}</p>
                  )}
                </div>
              </div>
            </motion.div>
          ))}

          {isLoading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
              <div className="flex flex-row max-w-[85%] md:max-w-[70%]">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-100 border border-blue-200 mr-3 md:mr-4 flex items-center justify-center shadow-sm">
                  <Bot className="w-5 h-5 text-blue-700 animate-pulse" />
                </div>
                <div className="p-4 rounded-2xl bg-slate-900/40 border border-slate-800 rounded-tl-none w-48 space-y-3 shadow-sm">
                  <div className="h-2 bg-slate-200 rounded animate-pulse" />
                  <div className="h-2 bg-slate-200 rounded w-5/6 animate-[pulse_1s_ease-in-out_infinite_0.2s]" />
                  <div className="h-2 bg-slate-200 rounded w-4/6 animate-[pulse_1s_ease-in-out_infinite_0.4s]" />
                </div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Replies */}
        {messages.length >= 1 && !isLoading && (
          <div className="px-4 md:px-6 pb-3 flex flex-wrap gap-2">
            {quickReplies.map((reply) => (
              <button
                key={reply}
                onClick={() => handleSend(reply)}
                className="text-xs bg-slate-800/50 hover:bg-agri-green/20 border border-slate-800 hover:border-agri-green/40 text-slate-400 hover:text-white py-1.5 px-3 rounded-full transition-colors font-medium flex items-center shadow-sm"
              >
                <Sparkles className="w-3 h-3 mr-1 text-agri-green" /> {reply}
              </button>
            ))}
          </div>
        )}

        {/* Attachment Preview Area */}
        <AnimatePresence>
          {previewImage && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="px-4 md:px-6 pb-2"
            >
              <div className="relative inline-block">
                <img src={previewImage} alt="Preview" className="h-20 rounded-lg border border-slate-700 shadow-sm" />
                <button
                  onClick={clearImage}
                  className="absolute -top-2 -right-2 bg-slate-900/40 border border-slate-700 text-slate-400 hover:text-white rounded-full p-1 shadow-md"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input Area */}
        <div className="p-4 bg-slate-800/50 border-t border-slate-800 relative">
          <form
            onSubmit={(e) => { e.preventDefault(); handleSend(); }}
            className="flex items-center space-x-2 bg-slate-900/40 p-2 rounded-xl border border-slate-700 focus-within:border-agri-green focus-within:ring-1 focus-within:ring-agri-green transition-all shadow-sm"
          >
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              onChange={handleImageSelect}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-3 text-slate-400 hover:text-agri-green hover:bg-green-50 rounded-lg transition-colors"
              title="Attach Image"
            >
              <ImageIcon className="w-5 h-5" />
            </button>

            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask your Copilot or attach an image..."
              className="flex-1 bg-transparent border-none focus:outline-none text-white px-2 placeholder-slate-400"
            />
            <Button type="submit" disabled={(!input.trim() && !selectedImage) || isLoading} variant="primary" className="px-4 min-w-[50px] !py-3">
              <Send className="w-5 h-5 mx-auto" />
            </Button>
          </form>
        </div>
      </Card>
    </div>
  );
}
