import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Mail, Lock, User, AlertCircle, ArrowRight } from 'lucide-react';

const AuthModal = ({ onAuthSuccess, onClose }) => {
  const [isSignUp, setIsSignUp] = useState(false);
  
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  
  const [errors, setErrors] = useState({});

  const validate = () => {
    const newErrors = {};
    if (!formData.email) newErrors.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(formData.email)) newErrors.email = 'Invalid email format';
    
    if (!formData.password) newErrors.password = 'Password is required';
    
    if (isSignUp) {
      if (!formData.name) newErrors.name = 'Name is required';
      if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = 'Passwords do not match';
      }
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validate()) {
      // Simulate successful local authentication
      const stableId = btoa(formData.email).replace(/[^a-zA-Z0-9]/g, '');
      const user = {
        id: stableId,
        name: isSignUp ? formData.name : formData.email.split('@')[0],
        email: formData.email,
        initials: (isSignUp ? formData.name : formData.email).substring(0, 2).toUpperCase()
      };
      onAuthSuccess(user);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="absolute inset-0 bg-nyaya-text/30 backdrop-blur-[4px]"
        onClick={onClose}
      />

      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 15 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96, y: -15 }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="relative w-full max-w-[440px] bg-nyaya-surface rounded-3xl shadow-[0_20px_60px_-15px_rgba(0,0,0,0.2)] border border-nyaya-border/50 overflow-hidden"
      >
        <button 
          onClick={onClose}
          className="absolute top-5 right-5 p-2 text-nyaya-muted hover:text-nyaya-text hover:bg-nyaya-bg rounded-full transition-all z-10"
        >
          <X size={20} />
        </button>

        <div className="p-8 sm:p-10">
          <div className="text-center mb-8">
            <h2 className="text-[26px] font-black tracking-tight text-nyaya-text mb-2">
              {isSignUp ? 'Create an Account' : 'Welcome Back'}
            </h2>
            <p className="text-sm text-nyaya-muted font-medium">
              {isSignUp 
                ? 'Sign up to access your copilot engine' 
                : 'Sign in to continue your legal research'}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <AnimatePresence mode="popLayout">
              {isSignUp && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-nyaya-muted">
                      <User size={18} />
                    </div>
                    <input
                      type="text"
                      placeholder="Full Name"
                      value={formData.name}
                      onChange={(e) => setFormData({...formData, name: e.target.value})}
                      className={`w-full pl-11 pr-4 py-3.5 bg-nyaya-bg/50 border ${errors.name ? 'border-red-400' : 'border-nyaya-border/80'} rounded-2xl text-nyaya-text text-[15px] focus:outline-none focus:ring-4 focus:ring-nyaya-primary/10 focus:border-nyaya-primary transition-all`}
                    />
                  </div>
                  {errors.name && <p className="text-red-500 text-xs mt-1.5 ml-1 flex items-center gap-1 font-medium"><AlertCircle size={12}/>{errors.name}</p>}
                </motion.div>
              )}
            </AnimatePresence>

            <div>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-nyaya-muted">
                  <Mail size={18} />
                </div>
                <input
                  type="email"
                  placeholder="name@example.com"
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                  className={`w-full pl-11 pr-4 py-3.5 bg-nyaya-bg/50 border ${errors.email ? 'border-red-400' : 'border-nyaya-border/80'} rounded-2xl text-nyaya-text text-[15px] focus:outline-none focus:ring-4 focus:ring-nyaya-primary/10 focus:border-nyaya-primary transition-all`}
                />
              </div>
              {errors.email && <p className="text-red-500 text-xs mt-1.5 ml-1 flex items-center gap-1 font-medium"><AlertCircle size={12}/>{errors.email}</p>}
            </div>

            <div>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-nyaya-muted">
                  <Lock size={18} />
                </div>
                <input
                  type="password"
                  placeholder="Password"
                  value={formData.password}
                  onChange={(e) => setFormData({...formData, password: e.target.value})}
                  className={`w-full pl-11 pr-4 py-3.5 bg-nyaya-bg/50 border ${errors.password ? 'border-red-400' : 'border-nyaya-border/80'} rounded-2xl text-nyaya-text text-[15px] focus:outline-none focus:ring-4 focus:ring-nyaya-primary/10 focus:border-nyaya-primary transition-all`}
                />
              </div>
              {errors.password && <p className="text-red-500 text-xs mt-1.5 ml-1 flex items-center gap-1 font-medium"><AlertCircle size={12}/>{errors.password}</p>}
            </div>

            <AnimatePresence mode="popLayout">
              {isSignUp && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-nyaya-muted">
                      <Lock size={18} />
                    </div>
                    <input
                      type="password"
                      placeholder="Confirm Password"
                      value={formData.confirmPassword}
                      onChange={(e) => setFormData({...formData, confirmPassword: e.target.value})}
                      className={`w-full pl-11 pr-4 py-3.5 bg-nyaya-bg/50 border ${errors.confirmPassword ? 'border-red-400' : 'border-nyaya-border/80'} rounded-2xl text-nyaya-text text-[15px] focus:outline-none focus:ring-4 focus:ring-nyaya-primary/10 focus:border-nyaya-primary transition-all`}
                    />
                  </div>
                  {errors.confirmPassword && <p className="text-red-500 text-xs mt-1.5 ml-1 flex items-center gap-1 font-medium"><AlertCircle size={12}/>{errors.confirmPassword}</p>}
                </motion.div>
              )}
            </AnimatePresence>

            {!isSignUp && (
              <div className="flex justify-end pt-1">
                <button type="button" className="text-[13px] font-bold text-nyaya-primary hover:text-nyaya-primary/80 transition-colors">
                  Forgot password?
                </button>
              </div>
            )}

            <button 
              type="submit"
              className="w-full bg-nyaya-primary hover:bg-nyaya-primary/90 text-white font-bold py-3.5 rounded-2xl shadow-[0_8px_16px_rgba(79,70,229,0.25)] hover:shadow-[0_8px_24px_rgba(79,70,229,0.35)] hover:-translate-y-0.5 transition-all flex items-center justify-center gap-2 group mt-4"
            >
              {isSignUp ? 'Create Account' : 'Sign In'}
              <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
            </button>
          </form>

          <div className="mt-8 flex items-center gap-4">
            <div className="flex-1 h-px bg-nyaya-border/60"></div>
            <span className="text-[11px] font-black text-nyaya-muted/70 uppercase tracking-widest px-2">Or continue with</span>
            <div className="flex-1 h-px bg-nyaya-border/60"></div>
          </div>

          <button type="button" className="w-full mt-6 bg-white border border-nyaya-border hover:bg-nyaya-bg text-nyaya-text font-bold py-3.5 rounded-2xl transition-all flex items-center justify-center gap-3">
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            Google
          </button>

          <div className="mt-8 text-center text-[14px] text-nyaya-muted font-medium">
            {isSignUp ? 'Already have an account? ' : "Don't have an account? "}
            <button 
              onClick={() => {
                setIsSignUp(!isSignUp);
                setErrors({});
              }} 
              className="font-black text-nyaya-primary hover:text-nyaya-primary/80 transition-colors ml-1"
            >
              {isSignUp ? 'Sign in' : 'Sign up'}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default AuthModal;
