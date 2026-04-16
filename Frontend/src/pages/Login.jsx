import { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import { Sprout } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { Button } from '../components/common/Button';
import { Input } from '../components/common/Input';
import { Card, CardContent } from '../components/common/Card';

import { BootSequence } from '../components/auth/BootSequence';

export default function Login() {
  const [formData, setFormData] = useState({ username: '', password: '' });
  const [isLoading, setIsLoading] = useState(false);
  const [isBooting, setIsBooting] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const from = location.state?.from?.pathname || "/home";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      await login(formData);
      setIsBooting(true);
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to login. Please check credentials.');
      setIsLoading(false);
    }
  };

  const executeNavigation = () => {
    toast.success('Welcome back to AgriBot!');
    navigate(from, { replace: true });
  }

  return (
    <>
      {isBooting && <BootSequence onComplete={executeNavigation} />}
      <div className="min-h-screen flex items-center justify-center bg-background px-4 relative overflow-hidden">
        {/* Background Decor */}
        <div className="absolute w-[600px] h-[600px] bg-neon-green/10 rounded-full blur-[100px] -top-48 -right-48 pointer-events-none animate-pulse" />
        <div className="absolute w-[600px] h-[600px] bg-neon-blue/10 rounded-full blur-[100px] -bottom-48 -left-48 pointer-events-none" />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className={`w-full max-w-md z-10 transition-opacity duration-500 ${isBooting ? 'opacity-0' : 'opacity-100'}`}
        >
        <Card className="border border-neon-green/20 shadow-[0_0_50px_rgba(0,230,118,0.1)]">
          <CardContent className="p-8">
            <div className="text-center mb-8">
              <motion.div 
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.2 }}
                className="w-16 h-16 bg-gradient-to-br from-neon-green/20 to-neon-blue/20 rounded-2xl mx-auto flex items-center justify-center mb-4 border border-neon-green/30"
              >
                <Sprout className="w-8 h-8 text-neon-green" />
              </motion.div>
              <h1 className="text-3xl font-display font-bold text-white mb-2">Welcome Back</h1>
              <p className="text-gray-400">Sign in to your AI farming assistant</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              <Input
                label="Username"
                type="text"
                placeholder="Enter your username"
                required
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              />
              <Input
                label="Password"
                type="password"
                placeholder="••••••••"
                required
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              />
              
              <Button type="submit" className="w-full mt-6" isLoading={isLoading} size="lg">
                Sign In
              </Button>
            </form>

            <div className="mt-6 text-center text-sm text-gray-400">
              Don't have an account?{' '}
              <Link to="/register" className="text-neon-green hover:underline">
                Register here
              </Link>
            </div>
          </CardContent>
        </Card>
        </motion.div>
      </div>
    </>
  );
}
