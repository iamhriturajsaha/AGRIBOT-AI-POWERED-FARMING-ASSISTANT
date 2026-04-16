import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Sparkles, Image as ImageIcon, X, Volume2, VolumeX, ChevronDown, Globe2, ArrowRight } from 'lucide-react';
import { toast } from 'react-hot-toast';
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
  const [isListening, setIsListening] = useState(false);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);

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
      if (inputLang !== 'English' || outputLang !== 'English') {
        finalPrompt = `[SYSTEM INSTRUCTION: The user is speaking in ${inputLang}. You must respond strictly and ONLY in ${outputLang}.]\n\nUser Message: ${textToSend}`;
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
    <div className="h-[calc(100vh-6rem)] flex flex-col max-w-5xl mx-auto">
      <div className="mb-6 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-display font-bold flex items-center text-white">
            <Sparkles className="text-neon-blue mr-3 w-8 h-8" />
            AI Copilot
          </h1>
          <p className="text-gray-400 mt-1">Chat and upload images. I can see your crops!</p>
        </div>

        {/* Multilingual Translation Module - Futuristic UI */}
        <div className="flex items-center space-x-3">
          {/* Translate From */}
          <div className="relative">
            <button
              onClick={() => { setIsInputMenuOpen(!isInputMenuOpen); setIsOutputMenuOpen(false); }}
              className={`flex items-center justify-between px-3 py-1.5 border rounded-lg glass-card backdrop-blur-md transition-all duration-300 min-w-[120px] ${isInputMenuOpen ? 'border-neon-pink shadow-[0_0_15px_rgba(244,43,142,0.4)] text-neon-pink' : 'border-white/10 text-gray-300 hover:border-neon-blue hover:text-neon-blue'}`}
            >
              <span className="text-[10px] font-bold uppercase tracking-widest">{inputLang}</span>
              <ChevronDown className={`w-3 h-3 ml-2 transition-transform duration-300 ${isInputMenuOpen ? 'rotate-180' : ''}`} />
            </button>

            <AnimatePresence>
              {isInputMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute z-50 top-full right-0 mt-2 w-48 max-h-[300px] overflow-y-auto overflow-x-hidden glass-card bg-black/95 border border-neon-pink/40 rounded-xl shadow-[0_0_30px_rgba(244,43,142,0.2)] scroll-smooth"
                >
                  <div className="sticky top-0 bg-black/90 p-2 border-b border-white/10 text-[9px] text-gray-500 font-bold tracking-widest uppercase flex items-center">
                    <Globe2 className="w-3 h-3 mr-2" /> Detect Language
                  </div>
                  {availableLanguages.map(lang => (
                    <button
                      key={lang}
                      onClick={() => { setInputLang(lang); setIsInputMenuOpen(false); }}
                      className={`w-full text-left px-4 py-2.5 text-xs font-bold uppercase tracking-wider transition-colors ${inputLang === lang ? 'text-neon-pink bg-neon-pink/10 border-l-2 border-neon-pink shadow-[inset_10px_0_10px_-10px_rgba(244,43,142,0.5)]' : 'text-gray-400 hover:text-white hover:bg-white/5'}`}
                    >
                      {lang}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <ArrowRight className="w-4 h-4 text-neon-blue/50" />

          {/* Translate To */}
          <div className="relative">
            <button
              onClick={() => { setIsOutputMenuOpen(!isOutputMenuOpen); setIsInputMenuOpen(false); }}
              className={`flex items-center justify-between px-3 py-1.5 border rounded-lg glass-card backdrop-blur-md transition-all duration-300 min-w-[120px] ${isOutputMenuOpen ? 'border-neon-blue shadow-[0_0_15px_rgba(0,195,255,0.4)] text-neon-blue' : 'border-neon-blue/40 text-neon-blue hover:shadow-[0_0_10px_rgba(0,195,255,0.2)]'}`}
            >
              <span className="text-[10px] font-bold uppercase tracking-widest">{outputLang}</span>
              <ChevronDown className={`w-3 h-3 ml-2 transition-transform duration-300 ${isOutputMenuOpen ? 'rotate-180' : ''}`} />
            </button>

            <AnimatePresence>
              {isOutputMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute z-50 top-full right-0 mt-2 w-48 max-h-[300px] overflow-y-auto overflow-x-hidden glass-card bg-black/95 border border-neon-blue/40 rounded-xl shadow-[0_0_30px_rgba(0,195,255,0.2)] scroll-smooth"
                >
                  <div className="sticky top-0 bg-black/90 p-2 border-b border-white/10 text-[9px] text-gray-500 font-bold tracking-widest uppercase flex items-center">
                    <Bot className="w-3 h-3 mr-2" /> Output Language
                  </div>
                  {availableLanguages.map(lang => (
                    <button
                      key={lang}
                      onClick={() => { setOutputLang(lang); setIsOutputMenuOpen(false); }}
                      className={`w-full text-left px-4 py-2.5 text-xs font-bold uppercase tracking-wider transition-colors ${outputLang === lang ? 'text-neon-blue bg-neon-blue/10 border-l-2 border-neon-blue shadow-[inset_10px_0_10px_-10px_rgba(0,195,255,0.5)]' : 'text-gray-400 hover:text-white hover:bg-white/5'}`}
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

      <Card className="flex-1 flex flex-col overflow-hidden border-neon-blue/20">
        <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth">
          {messages.map((msg) => (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${msg.role === 'user' ? 'bg-neon-green/20 ml-4' : 'bg-neon-blue/20 mr-4 shadow-[0_0_15px_rgba(0,153,255,0.3)]'
                  }`}>
                  {msg.role === 'user' ? <User className="w-5 h-5 text-neon-green" /> : <Bot className="w-5 h-5 text-neon-blue" />}
                </div>

                <div className={`p-4 rounded-2xl ${msg.role === 'user'
                    ? 'bg-neon-green/10 text-white border border-neon-green/20 rounded-tr-none'
                    : 'bg-white/5 text-gray-200 border border-panelBorder rounded-tl-none'
                  }`}>
                  {msg.image && (
                    <div className="mb-3">
                      <img src={msg.image} alt="User upload" className="max-w-[200px] rounded-lg border border-white/10" />
                    </div>
                  )}
                  {msg.role === 'bot' && msg.text.includes('AI Insight:') ? (
                    <div>
                      <div className="mb-3 whitespace-pre-wrap leading-relaxed">{msg.text.split('AI Insight:')[0]}</div>
                      <div className="bg-black/30 border border-neon-blue/20 p-3 rounded-lg">
                        <div className="font-bold text-neon-blue flex items-center mb-1"><Sparkles className="w-4 h-4 mr-2" /> AI Insight</div>
                        <div className="whitespace-pre-wrap text-sm">{msg.text.split('AI Insight:')[1].trim()}</div>
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
              <div className="flex flex-row max-w-[80%]">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-neon-blue/20 mr-4 flex items-center justify-center shadow-[0_0_15px_rgba(0,153,255,0.5)]">
                  <Bot className="w-5 h-5 text-neon-blue animate-pulse" />
                </div>
                <div className="p-4 rounded-2xl bg-white/5 border border-panelBorder rounded-tl-none w-48 space-y-3">
                  <div className="h-2 bg-neon-blue/50 rounded animate-pulse" />
                  <div className="h-2 bg-neon-blue/50 rounded w-5/6 animate-[pulse_1s_ease-in-out_infinite_0.2s]" />
                  <div className="h-2 bg-neon-blue/50 rounded w-4/6 animate-[pulse_1s_ease-in-out_infinite_0.4s]" />
                </div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Replies */}
        {messages.length >= 1 && !isLoading && (
          <div className="px-6 pb-2 flex flex-wrap gap-2">
            {quickReplies.map((reply) => (
              <button
                key={reply}
                onClick={() => handleSend(reply)}
                className="text-xs bg-white/5 hover:bg-neon-blue/10 border border-white/10 hover:border-neon-blue/30 text-gray-300 py-1.5 px-3 rounded-full transition-colors font-medium flex items-center"
              >
                <Sparkles className="w-3 h-3 mr-1 text-neon-blue" /> {reply}
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
              className="px-6 pb-2"
            >
              <div className="relative inline-block">
                <img src={previewImage} alt="Preview" className="h-20 rounded-lg border border-neon-blue/30" />
                <button
                  onClick={clearImage}
                  className="absolute -top-2 -right-2 bg-panel border border-panelBorder text-gray-400 hover:text-white rounded-full p-1"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input Area */}
        <div className="p-4 bg-black/40 border-t border-panelBorder relative">
          <form
            onSubmit={(e) => { e.preventDefault(); handleSend(); }}
            className="flex items-center space-x-2 bg-white/5 p-2 rounded-xl border border-white/10 focus-within:border-neon-blue/50 focus-within:ring-1 focus-within:ring-neon-blue/50 transition-all shadow-lg"
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
              className="p-3 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              title="Attach Image"
            >
              <ImageIcon className="w-5 h-5" />
            </button>



            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask your Copilot or attach an image..."
              className="flex-1 bg-transparent border-none focus:outline-none text-white px-2 placeholder-gray-500"
            />
            <Button type="submit" disabled={(!input.trim() && !selectedImage) || isLoading} variant="secondary" className="px-4 min-w-[50px] !py-3">
              <Send className="w-5 h-5 mx-auto" />
            </Button>
          </form>
        </div>
      </Card>
    </div>
  );
}
