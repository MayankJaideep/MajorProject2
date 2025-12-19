import React, { useState } from 'react';
import { LayoutDashboard, MessageSquare, BookOpen, Upload, Clock } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ChatInterface from './components/ChatInterface';
import Dashboard from './components/Dashboard';
import PDFUploader from './components/PDFUploader';
import Chronology from './components/Chronology';

function App() {
  const [activeTab, setActiveTab] = useState('chat');

  const tabs = [
    { id: 'chat', label: 'Research Assistant', icon: MessageSquare },
    { id: 'dashboard', label: 'Prediction Dashboard', icon: LayoutDashboard },
    { id: 'chronology', label: 'Case Timeline', icon: Clock },
    { id: 'upload', label: 'Knowledge Base', icon: Upload },
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Sidebar / Navigation */}
      <div className="fixed top-0 left-0 h-full w-64 bg-white border-r border-slate-200 shadow-lg z-10 flex flex-col">
        <div className="p-6 border-b border-slate-100">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            Legal AI
          </h1>
          <p className="text-xs text-slate-500 mt-1">Research Engine v2.0</p>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 
                  ${activeTab === tab.id
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-200'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`}
              >
                <Icon size={20} />
                <span className="font-medium">{tab.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-100">
          <div className="bg-slate-50 rounded-lg p-3 text-xs text-slate-500">
            <p><strong>System Status:</strong> <span className="text-green-600">● Active</span></p>
            <p className="mt-1">Vector DB: Ready</p>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="pl-64 min-h-screen relative">
        <header className="sticky top-0 bg-white/80 backdrop-blur-md border-b border-slate-200 h-16 flex items-center px-8 z-20">
          <h2 className="text-lg font-semibold text-slate-800">
            {tabs.find(t => t.id === activeTab)?.label}
          </h2>
        </header>

        <main className="p-8 max-w-7xl mx-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              {activeTab === 'chat' && <ChatInterface />}
              {activeTab === 'dashboard' && <Dashboard />}
              {activeTab === 'chronology' && <Chronology />}
              {activeTab === 'upload' && <PDFUploader />}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
