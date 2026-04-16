import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Mail, Bell, Shield, Camera, Save, Key, Globe, Layout, Smartphone } from 'lucide-react';
import { Card, CardContent } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';
import toast from 'react-hot-toast';

export default function Profile() {
  const { user, updateUser } = useAuth();
  const [activeTab, setActiveTab] = useState('profile');
  const [isSaving, setIsSaving] = useState(false);
  const fileInputRef = useRef(null);

  // Profile Form State
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [avatar, setAvatar] = useState(null);

  // Security Form State
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  // Notifications State
  const [pushToggled, setPushToggled] = useState(true);
  const [emailToggled, setEmailToggled] = useState(false);

  useEffect(() => {
    if (user) {
      setUsername(user.username || '');
      setEmail(user.email || '');
    }
    const savedAvatar = localStorage.getItem('agribot_avatar');
    if (savedAvatar) setAvatar(savedAvatar);
  }, [user]);

  const handleAvatarChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 2 * 1024 * 1024) {
        toast.error('Image exceeds 2MB limit.');
        return;
      }
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result;
        setAvatar(base64String);
        localStorage.setItem('agribot_avatar', base64String);
        toast.success('Avatar updated locally!');
      };
      reader.readAsDataURL(file);
    }
  };

  const handleProfileSave = async () => {
    setIsSaving(true);
    try {
      await api.patch('/users/profile/', { username, email });
      if (updateUser) {
        updateUser({ username, email });
      }
      toast.success('Profile settings updated successfully!');
    } catch (error) {
      if (error.response?.data?.errors) {
        const errs = error.response.data.errors;
        const msg = Object.values(errs).flat().join(', ');
        toast.error(`Error: ${msg}`);
      } else {
        toast.error('Failed to update profile.');
      }
      console.error(error.response?.data || error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleSecuritySave = async () => {
    if (!newPassword) {
      toast.error('Please enter a new password.');
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error('New passwords do not match.');
      return;
    }
    
    setIsSaving(true);
    try {
       await api.patch('/users/profile/', { password: newPassword });
       toast.success('Password updated successfully! Please re-login eventually.');
       setNewPassword('');
       setConfirmPassword('');
    } catch(err) {
       toast.error('Failed to update password.');
    } finally {
       setIsSaving(false);
    }
  };

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'security', label: 'Security', icon: Shield },
  ];

  return (
    <div className="max-w-4xl mx-auto pb-10 pt-4">
      <div className="mb-8">
        <h1 className="text-4xl font-display font-bold text-white mb-2">Account Settings</h1>
        <p className="text-gray-400">Manage your profile, preferences, and security settings.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] gap-8">
        {/* Sidebar/Navigation for Profile Settings */}
        <div className="space-y-2">
          {tabs.map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center px-4 py-3 rounded-xl font-medium transition-all duration-200 ${
                  isActive 
                    ? 'bg-neon-green/10 text-neon-green border border-neon-green/20 shadow-[0_0_15px_rgba(0,230,118,0.05)]' 
                    : 'text-gray-400 hover:text-white hover:bg-white/5 border border-transparent'
                }`}
              >
                <Icon className={`w-5 h-5 mr-3 transition-transform ${isActive ? 'scale-110' : ''}`} /> 
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Profile Form Content */}
        <div className="space-y-6">
          <AnimatePresence mode="wait">
            {activeTab === 'profile' && (
              <motion.div key="profile" initial={{ y: 10 }} animate={{ y: 0 }} exit={{ y: -10 }}>
                <Card className="glass-card">
                  <CardContent className="p-6 md:p-8">
                    <h2 className="text-2xl font-bold text-white mb-6">Public Profile</h2>
                    
                    <div className="flex items-center space-x-6 mb-8">
                      <div className="relative group">
                        <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-neon-green to-neon-blue flex items-center justify-center text-3xl font-bold text-black border-2 border-panelBorder overflow-hidden">
                          {avatar ? (
                            <img src={avatar} alt="Avatar" className="w-full h-full object-cover" />
                          ) : (
                            user?.username ? user.username[0].toUpperCase() : 'U'
                          )}
                        </div>
                        <button 
                          onClick={() => fileInputRef.current?.click()}
                          className="absolute inset-0 bg-black/60 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                        >
                          <Camera className="w-8 h-8 text-white" />
                        </button>
                        <input 
                          type="file" 
                          ref={fileInputRef} 
                          onChange={handleAvatarChange} 
                          accept="image/*" 
                          className="hidden" 
                        />
                      </div>
                      <div>
                        <h3 className="text-white font-medium text-lg">{username || 'user'}</h3>
                        <p className="text-sm text-neon-blue font-semibold">{email}</p>
                        <p className="text-[10px] text-gray-500 mt-1 uppercase tracking-wider">PNG or JPG no larger than 2MB.</p>
                      </div>
                    </div>

                    <div className="space-y-5">
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Username</label>
                        <div className="relative">
                          <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                          <input 
                            type="text" 
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="w-full pl-10 pr-4 py-3 bg-black/30 border border-white/10 rounded-xl text-white focus:outline-none focus:border-neon-green focus:ring-1 focus:ring-neon-green transition-all"
                          />
                        </div>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Email Address</label>
                        <div className="relative">
                          <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                          <input 
                            type="email" 
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="w-full pl-10 pr-4 py-3 bg-black/30 border border-white/10 rounded-xl text-white focus:outline-none focus:border-neon-green focus:ring-1 focus:ring-neon-green transition-all"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="mt-8 pt-6 border-t border-white/10 flex justify-end">
                      <Button onClick={handleProfileSave} disabled={isSaving}>
                        {isSaving ? 'Saving...' : <><Save className="w-4 h-4 mr-2" /> Save Changes</>}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {activeTab === 'notifications' && (
              <motion.div key="notifications" initial={{ y: 10 }} animate={{ y: 0 }} exit={{ y: -10 }}>
                <Card className="glass-card">
                  <CardContent className="p-6 md:p-8">
                    <h2 className="text-2xl font-bold text-white mb-6">Notification Preferences</h2>
                    
                    <div className="space-y-6">
                      <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/5">
                        <div className="flex items-center">
                          <div className="w-10 h-10 rounded-full bg-neon-blue/10 flex items-center justify-center mr-4">
                            <Smartphone className="w-5 h-5 text-neon-blue" />
                          </div>
                          <div>
                            <h4 className="text-white font-medium">Push Notifications</h4>
                            <p className="text-sm text-gray-400">Receive alerts about your crop health directly.</p>
                          </div>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input type="checkbox" className="sr-only peer" checked={pushToggled} onChange={() => setPushToggled(!pushToggled)} />
                          <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-neon-green"></div>
                        </label>
                      </div>

                      <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/5">
                        <div className="flex items-center">
                          <div className="w-10 h-10 rounded-full bg-yellow-400/10 flex items-center justify-center mr-4">
                            <Mail className="w-5 h-5 text-yellow-400" />
                          </div>
                          <div>
                            <h4 className="text-white font-medium">Email Digest</h4>
                            <p className="text-sm text-gray-400">Receive a weekly AI diagnostic summary.</p>
                          </div>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input type="checkbox" className="sr-only peer" checked={emailToggled} onChange={() => setEmailToggled(!emailToggled)} />
                          <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-neon-green"></div>
                        </label>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {activeTab === 'security' && (
              <motion.div key="security" initial={{ y: 10 }} animate={{ y: 0 }} exit={{ y: -10 }}>
                <Card className="glass-card">
                  <CardContent className="p-6 md:p-8">
                    <h2 className="text-2xl font-bold text-white mb-6">Security Settings</h2>
                    
                    <div className="space-y-5">
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Current Password</label>
                        <div className="relative">
                          <Key className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                          <input 
                            type="password" 
                            placeholder="••••••••"
                            className="w-full pl-10 pr-4 py-3 bg-black/30 border border-white/10 rounded-xl text-white focus:outline-none focus:border-neon-green focus:ring-1 focus:ring-neon-green transition-all"
                          />
                        </div>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">New Password</label>
                        <div className="relative">
                          <Key className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                          <input 
                            type="password" 
                            value={newPassword}
                            onChange={(e) => setNewPassword(e.target.value)}
                            placeholder="••••••••"
                            className="w-full pl-10 pr-4 py-3 bg-black/30 border border-white/10 rounded-xl text-white focus:outline-none focus:border-neon-green focus:ring-1 focus:ring-neon-green transition-all"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Confirm New Password</label>
                        <div className="relative">
                          <Key className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                          <input 
                            type="password" 
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            placeholder="••••••••"
                            className="w-full pl-10 pr-4 py-3 bg-black/30 border border-white/10 rounded-xl text-white focus:outline-none focus:border-neon-green focus:ring-1 focus:ring-neon-green transition-all"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="mt-8 pt-6 border-t border-white/10 flex justify-end">
                      <Button onClick={handleSecuritySave} disabled={isSaving}>
                        {isSaving ? 'Updating...' : <><Shield className="w-4 h-4 mr-2" /> Update Password</>}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
