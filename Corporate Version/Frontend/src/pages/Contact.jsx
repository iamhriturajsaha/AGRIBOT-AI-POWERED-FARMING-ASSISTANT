import { useState } from 'react';
import { motion } from 'framer-motion';
import { Mail, Phone, MapPin, Send, Building } from 'lucide-react';
import { Card, CardContent } from '../components/common/Card';
import toast from 'react-hot-toast';

export default function Contact() {
  const [formData, setFormData] = useState({ firstName: '', lastName: '', email: '', message: '' });
  const [isSending, setIsSending] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.firstName || !formData.email || !formData.message) {
      toast.error('Please fill out all required fields.');
      return;
    }
    
    setIsSending(true);
    // Simulate network delay
    setTimeout(() => {
      setIsSending(false);
      toast.success('Message sent! Our support team will respond shortly.');
      setFormData({ firstName: '', lastName: '', email: '', message: '' });
    }, 800);
  };

  return (
    <div className="flex-1 w-full p-4 md:p-8 overflow-y-auto bg-transparent">
      <div className="max-w-6xl mx-auto space-y-8 pb-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-white">Contact the AgriBot Team</h1>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Have questions about our enterprise agricultural solutions? We're here to help you optimize your farm management.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-12">
          {/* Contact Info Cards */}
          <div className="lg:col-span-1 space-y-6">
            <Card className="bg-slate-900/40 border-slate-800 shadow-sm border">
              <CardContent className="p-6 flex items-start space-x-4">
                <div className="p-3 bg-agri-lightGreen/10 rounded-xl text-agri-green shrink-0">
                  <Building className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-semibold text-white text-lg">Headquarters</h3>
                  <p className="text-slate-400 mt-1">3180 18th St<br/>San Francisco, CA 94110</p>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/40 border-slate-800 shadow-sm border">
              <CardContent className="p-6 flex items-start space-x-4">
                <div className="p-3 bg-agri-lightGreen/10 rounded-xl text-agri-green shrink-0">
                  <Mail className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-semibold text-white text-lg">Email Us</h3>
                  <p className="text-slate-400 mt-1">helpdesk@agribot.ai<br/>sales@agribot.ai</p>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/40 border-slate-800 shadow-sm border">
              <CardContent className="p-6 flex items-start space-x-4">
                <div className="p-3 bg-agri-lightGreen/10 rounded-xl text-agri-green shrink-0">
                  <Phone className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-semibold text-white text-lg">Call Us</h3>
                  <p className="text-slate-400 mt-1">+1 (415) 555-0198<br/>Mon-Fri, 9am - 5pm PST</p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Contact Form */}
          <Card className="lg:col-span-2 bg-slate-900/40 border-slate-800 shadow-sm border">
            <CardContent className="p-8">
              <h2 className="text-2xl font-bold text-white mb-6">Send us a Message</h2>
              <form className="space-y-6" onSubmit={handleSubmit}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-300">First Name</label>
                    <input 
                      type="text" 
                      value={formData.firstName}
                      onChange={(e) => setFormData({...formData, firstName: e.target.value})}
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-agri-lightGreen focus:border-transparent transition-colors"
                      placeholder="John"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-300">Last Name</label>
                    <input 
                      type="text" 
                      value={formData.lastName}
                      onChange={(e) => setFormData({...formData, lastName: e.target.value})}
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-agri-lightGreen focus:border-transparent transition-colors"
                      placeholder="Doe"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-300">Email Address</label>
                  <input 
                    type="email" 
                    value={formData.email}
                    onChange={(e) => setFormData({...formData, email: e.target.value})}
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-agri-lightGreen focus:border-transparent transition-colors"
                    placeholder="john@example.com"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-300">Message</label>
                  <textarea 
                    rows={5}
                    value={formData.message}
                    onChange={(e) => setFormData({...formData, message: e.target.value})}
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-agri-lightGreen focus:border-transparent transition-colors resize-none"
                    placeholder="How can we help you?"
                    required
                  />
                </div>

                <motion.button
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                  disabled={isSending}
                  className={`w-full ${isSending ? 'bg-agri-green/70 cursor-not-allowed' : 'bg-agri-green hover:bg-agri-green/90'} text-white font-medium py-3 px-6 rounded-lg flex items-center justify-center space-x-2 transition-colors shadow-sm`}
                  type="submit"
                >
                  <Send className={`w-5 h-5 ${isSending ? 'animate-pulse' : ''}`} />
                  <span>{isSending ? 'Sending...' : 'Send Message'}</span>
                </motion.button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
