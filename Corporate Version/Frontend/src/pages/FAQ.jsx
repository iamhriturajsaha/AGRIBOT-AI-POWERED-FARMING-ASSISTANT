import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HelpCircle, ChevronDown, Info } from 'lucide-react';
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
    answer: "Enterprise users looking to integrate our prediction API directly into their own IoT farming pipelines can reach out to enterprise@agribot.team for an API key."
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
    answer: "Yes. Our Deep Learning backend physically injects the generated prediction metadata into your rolling chat history. This means the AI will remember the exact visual symptoms of an uploaded leaf several messages later."
  }
];

function FAQItem({ faq, index, activeIndex, setActiveIndex }) {
  const isActive = activeIndex === index;

  return (
    <div className="border-b border-slate-800 last:border-b-0">
      <button
        onClick={() => setActiveIndex(isActive ? null : index)}
        className="w-full py-6 flex items-center justify-between text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-agri-lightGreen rounded-lg px-2 group"
      >
        <span className={`text-lg font-medium transition-colors duration-200 ${isActive ? 'text-agri-green' : 'text-slate-300 group-hover:text-white'}`}>
          {faq.question}
        </span>
        <motion.div
          animate={{ rotate: isActive ? 180 : 0 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
        >
          <ChevronDown className={`w-5 h-5 ${isActive ? 'text-agri-green' : 'text-slate-400 group-hover:text-slate-400'}`} />
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
            <div className="pb-6 px-2 text-slate-400 leading-relaxed">
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
    <div className="max-w-4xl mx-auto pb-10 pt-8 px-4 md:px-0">
      
      {/* About Us Section */}
      <div className="mb-16">
        <div className="text-center mb-8">
          <motion.div 
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="inline-flex w-16 h-16 rounded-2xl bg-agri-lightGreen/10 border border-agri-lightGreen/20 items-center justify-center mb-4"
          >
            <Info className="w-8 h-8 text-agri-green" />
          </motion.div>
          <h1 className="text-4xl font-bold text-white mb-4">About AgriBot</h1>
        </div>
        <Card className="bg-slate-900/40 border text-center border-slate-800 shadow-sm">
           <CardContent className="p-8">
             <p className="text-lg text-slate-400 leading-relaxed mb-6">
                AgriBot was founded with a singular mission: to empower modern farmers and agricultural enterprises with cutting-edge, data-driven intelligence. In an era where precision agriculture is paramount, our goal is to bridge the gap between complex artificial intelligence and practical, on-the-ground farm management.
             </p>
             <p className="text-lg text-slate-400 leading-relaxed">
                By integrating state-of-the-art computer vision for disease diagnostics, real-time environmental telemetry, and proactive AI assistance, we are building the digital nervous system for the farms of tomorrow. We believe that sustainable, high-yield farming is achievable through smarter technology, and we are dedicated to providing the tools that make it possible.
             </p>
           </CardContent>
        </Card>
      </div>

      <div className="mb-10 text-center">
        <motion.div 
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="inline-flex w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/20 items-center justify-center mb-4"
        >
          <HelpCircle className="w-8 h-8 text-blue-600" />
        </motion.div>
        <h2 className="text-3xl font-bold text-white mb-4">Frequently Asked Questions</h2>
        <p className="text-slate-400 max-w-xl mx-auto">
          Need help navigating your enterprise farming assistant? Browse our most common answers below.
        </p>
      </div>

      <motion.div
        initial={{ y: 20 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <Card className="bg-slate-900/40 border border-slate-800 shadow-sm">
          <CardContent className="p-4 md:p-8">
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
