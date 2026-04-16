import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HelpCircle, ChevronDown, Leaf } from 'lucide-react';
import { Card, CardContent } from '../components/common/Card';

const faqData = [
  {
    question: "How accurate is the AI Crop Diagnostic?",
    answer: "Our advanced neural network model is trained on thousands of high-resolution crop images and currently boasts an accuracy rate of 94%+ for detecting common diseases like Blight, Rust, and Spots. For the best accuracy, ensure your photos are well-lit and focused."
  },
  {
    question: "What crops are currently supported?",
    answer: "AgriBot supports major agricultural crops including Tomatoes, Corn, Potatoes, Wheat, and Rice. We are constantly updating our model to include more varied flora and specific plant diseases."
  },
  {
    question: "Can I use AgriBot offline?",
    answer: "Currently, AgriBot requires an active internet connection to communicate securely with our cloud-based AI inference engine. We are looking into lightweight edge models for future offline support."
  },
  {
    question: "Is my farm data and location secure?",
    answer: "Absolutely. We do not store geotags from your uploaded images, and any data passed to our AI is encrypted at rest. Your chat history is private to your authenticated account."
  },
  {
    question: "How do I upgrade my account or get API credentials?",
    answer: "Enterprise users looking to integrate our prediction API directly into their own IoT farming pipelines can reach out to enterprise@agribot.ai for an API key."
  },
  {
    question: "How do the Proactive Push Notifications work?",
    answer: "AgriBot interfaces securely with your browser's native Notification API. If our open-meteo satellite feed detects a severe anomaly (like fungal-level humidity or massive UV spikes), the system will ping a warning directly to your OS tray to allow early intervention."
  },
  {
    question: "How do I generate an offline PDF Report?",
    answer: "On the top-right corner of your Dashboard, click 'Export Intelligence Report'. This triggers an internal cross-origin snapshot rendering of your entire Threat Assessment matrix and Yield History, bundling it into a professional PDF that downloads locally."
  },
  {
    question: "Does the AI Copilot understand images I previously sent it?",
    answer: "Yes! Unlike standard chatbots, our Deep Learning backend physically injects the generated Grad-CAM prediction metadata into your rolling chat history. This means the AI will remember the exact visual symptoms of an uploaded leaf several messages later."
  }
];

function FAQItem({ faq, index, activeIndex, setActiveIndex }) {
  const isActive = activeIndex === index;

  return (
    <div className="border-b border-panelBorder last:border-b-0">
      <button
        onClick={() => setActiveIndex(isActive ? null : index)}
        className="w-full py-6 flex items-center justify-between text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-neon-green rounded-lg px-2 group"
      >
        <span className={`text-lg font-medium transition-colors duration-200 ${isActive ? 'text-neon-green' : 'text-gray-200 group-hover:text-white'}`}>
          {faq.question}
        </span>
        <motion.div
          animate={{ rotate: isActive ? 180 : 0 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
        >
          <ChevronDown className={`w-5 h-5 ${isActive ? 'text-neon-green' : 'text-gray-400 group-hover:text-white'}`} />
        </motion.div>
      </button>
      <AnimatePresence>
        {isActive && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="pb-6 px-2 text-gray-400 leading-relaxed">
              {faq.answer}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function FAQ() {
  const [activeIndex, setActiveIndex] = useState(0);

  return (
    <div className="max-w-4xl mx-auto pb-10 pt-4">
      <div className="mb-10 text-center">
        <motion.div 
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="inline-flex w-16 h-16 rounded-2xl bg-neon-green/10 border border-neon-green/20 items-center justify-center mb-6"
        >
          <HelpCircle className="w-8 h-8 text-neon-green" />
        </motion.div>
        <h1 className="text-4xl font-display font-bold text-white mb-4">Frequently Asked Questions</h1>
        <p className="text-gray-400 max-w-xl mx-auto">
          Need help navigating your new AI farming assistant? Browse our most common answers below.
        </p>
      </div>

      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="mb-12"
      >
        <Card className="glass-card border-neon-green/10">
          <CardContent className="p-8 md:p-10">
            <h2 className="text-2xl font-display font-black text-white italic tracking-tighter uppercase mb-6 flex items-center">
              <span className="w-8 h-8 rounded-lg bg-neon-green/20 flex items-center justify-center mr-4">
                <Leaf className="w-5 h-5 text-neon-green" />
              </span>
              About AgriBot // The Core
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
              <div className="space-y-4">
                <p className="text-gray-300 leading-relaxed font-medium">
                  AgriBot is a next-generation AI platform designed to empower modern farmers with real-time crop diagnostics and data-driven insights. 
                  Our system bridges the gap between traditional agriculture and cutting-edge cybernetics.
                </p>
                <div className="flex flex-wrap gap-3">
                  <span className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-xs font-bold text-neon-green uppercase tracking-widest italic">// Neural Vision</span>
                  <span className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-xs font-bold text-neon-blue uppercase tracking-widest italic">// Bio-Telemetry</span>
                  <span className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-xs font-bold text-neon-pink uppercase tracking-widest italic">// Geo-Sync</span>
                </div>
              </div>
              <div className="p-6 bg-black/40 rounded-2xl border border-white/5 space-y-4">
                <div className="flex items-start">
                  <div className="w-1.5 h-1.5 bg-neon-green rounded-full mt-2 mr-3 shadow-[0_0_8px_#0ff]" />
                  <p className="text-sm text-gray-400 font-medium italic"><strong className="text-white">Precision Diagnostics:</strong> Identify pathogens before they devastate yields.</p>
                </div>
                <div className="flex items-start">
                  <div className="w-1.5 h-1.5 bg-neon-blue rounded-full mt-2 mr-3 shadow-[0_0_8px_#f0f]" />
                  <p className="text-sm text-gray-400 font-medium italic"><strong className="text-white">Neural Assistant:</strong> A context-aware partner that remembers your field history.</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div
        initial={{ y: 20 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <Card className="glass-card">
          <CardContent className="p-2 md:p-6 opacity-90 hover:opacity-100 transition-opacity">
            <h2 className="text-xl font-bold text-white px-4 pt-4 pb-2 italic uppercase tracking-widest flex items-center">
              <HelpCircle className="w-5 h-5 mr-3 text-neon-green" /> Support Intelligence // FAQ
            </h2>
            {faqData.map((faq, index) => (
              <FAQItem 
                key={index} 
                faq={faq} 
                index={index} 
                activeIndex={activeIndex} 
                setActiveIndex={setActiveIndex} 
              />
            ))}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
