import { createContext, useState, useEffect, useContext } from 'react';
import api from '../services/api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('token');
      if (token) {
        try {
          // Assume there is a profile endpoint or we just decode JWT
          // For now, let's mock it if the backend profile endpoint doesn't exist yet
          const res = await api.get('/users/profile/').catch(() => ({ data: { data: { username: 'Farmer' } } }));
          setUser(res.data.data);
        } catch (error) {
          localStorage.removeItem('token');
        }
      }
      setLoading(false);
    };

    initAuth();

    const handleUnauthorized = () => {
      setUser(null);
    };
    window.addEventListener('auth:unauthorized', handleUnauthorized);
    return () => window.removeEventListener('auth:unauthorized', handleUnauthorized);
  }, []);

  const login = async (credentials) => {
    const res = await api.post('/users/login/', credentials);
    const { token, user: userData } = res.data;
    // Just in case backend returns access/refresh like simplejwt
    const accessToken = token || res.data.access; 
    localStorage.setItem('token', accessToken);
    
    // We fetch the profile immediately to get the fully hydrated user object
    try {
      const profileRes = await api.get('/users/profile/', {
         headers: { Authorization: `Bearer ${accessToken}` }
      });
      setUser(profileRes.data.data);
    } catch (e) {
      setUser(userData || { username: credentials.username || 'User' });
    }
    
    return res.data;
  };

  const updateUser = (newUserData) => {
    setUser(prev => ({ ...prev, ...newUserData }));
  };

  const register = async (data) => {
    const res = await api.post('/users/register/', data);
    return res.data;
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, updateUser }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
