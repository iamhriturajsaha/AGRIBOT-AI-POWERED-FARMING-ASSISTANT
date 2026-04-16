import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import { Sprout } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { Button } from '../components/common/Button';
import { Input } from '../components/common/Input';
import { Card, CardContent } from '../components/common/Card';

export default function Register() {
  const [formData, setFormData] = useState({ username: '', email: '', password: '' });
  const [isLoading, setIsLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      await register(formData);
      toast.success('Account created successfully! Please log in.');
      navigate('/login');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to register. Username might be taken.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-4 relative overflow-hidden">
      {/* Background Decor */}
      <div className="absolute w-[600px] h-[600px] bg-neon-blue/10 rounded-full blur-[100px] -top-48 -left-48 pointer-events-none animate-pulse" />
      <div className="absolute w-[600px] h-[600px] bg-neon-pink/10 rounded-full blur-[100px] -bottom-48 -right-48 pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md z-10"
      >
        <Card className="border border-neon-blue/20 shadow-[0_0_50px_rgba(0,242,254,0.1)]">
          <CardContent className="p-8">
            <div className="text-center mb-8">
              <motion.div 
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.2 }}
                className="w-16 h-16 bg-gradient-to-br from-neon-blue/20 to-neon-pink/20 rounded-2xl mx-auto flex items-center justify-center mb-4 border border-neon-blue/30"
              >
                <Sprout className="w-8 h-8 text-neon-blue" />
              </motion.div>
              <h1 className="text-3xl font-display font-bold text-white mb-2">Create Account</h1>
              <p className="text-gray-400">Join the AI agriculture revolution</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <Input
                label="Username"
                type="text"
                required
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              />
              <Input
                label="Email Address"
                type="email"
                required
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              />
              <Input
                label="Password"
                type="password"
                required
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              />
              
              <Button variant="secondary" type="submit" className="w-full mt-6" isLoading={isLoading} size="lg">
                Create Account
              </Button>
            </form>

            <div className="mt-6 text-center text-sm text-gray-400">
              Already have an account?{' '}
              <Link to="/login" className="text-neon-blue hover:underline">
                Sign in Document
              </Link>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
