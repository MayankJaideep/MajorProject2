import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import { useChatHistory } from '../context/ChatHistoryContext';
import { MessageSquare, Trash2, Plus, ChevronLeft, ChevronRight, Clock } from 'lucide-react';

export default function ChatHistoryPanel() {
  const { isAuthenticated } = useAuth();
  const { chatSessions, currentSessionId, loadSession, deleteSession } = useChatHistory();
  const [isExpanded, setIsExpanded] = useState(true);

  if (!isAuthenticated) return null;

  const handleNewChat = () => {
    window.dispatchEvent(new CustomEvent('lumina:new-chat'));
  };

  const handleLoadSession = (session) => {
    const messages = loadSession(session.id);
    if (messages) {
      window.dispatchEvent(new CustomEvent('lumina:load-session', { detail: { messages, sessionId: session.id } }));
    }
  };

  const handleDelete = (e, session) => {
    e.stopPropagation();
    deleteSession(session.id);
    if (currentSessionId === session.id) {
      handleNewChat();
    }
  };

  const getRelativeTime = (isoString) => {
    const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
    const daysDifference = Math.round((new Date(isoString) - new Date()) / (1000 * 60 * 60 * 24));
    if (daysDifference === 0) return 'Today';
    if (daysDifference === -1) return 'Yesterday';
    return rtf.format(daysDifference, 'day');
  };

  return (
    <div className="relative h-full flex shrink-0 z-20">
      <motion.div
        initial={false}
        animate={{ width: isExpanded ? 280 : 0, opacity: isExpanded ? 1 : 0 }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="h-full bg-nyaya-bg border-r border-nyaya-border overflow-hidden flex flex-col shadow-[4px_0_24px_rgba(0,0,0,0.02)]"
      >
        <div className="p-4 border-b border-nyaya-border/50 shrink-0">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2 bg-white border border-nyaya-border hover:border-nyaya-primary/50 text-nyaya-text hover:text-nyaya-primary font-bold py-2.5 rounded-xl transition-all shadow-sm group"
          >
            <Plus size={18} className="text-nyaya-muted group-hover:text-nyaya-primary transition-colors" />
            <span className="text-sm">New Chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-1">
          <div className="px-2 py-1.5 mb-2 flex items-center gap-2">
            <Clock size={14} className="text-nyaya-muted" />
            <span className="text-xs font-bold text-nyaya-muted uppercase tracking-widest">Recent Chats</span>
          </div>

          <AnimatePresence>
            {chatSessions.length === 0 ? (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="px-2 py-6 text-center text-nyaya-muted text-sm italic">
                No recent chats found.
              </motion.div>
            ) : (
              chatSessions.map((session) => (
                <motion.div
                  key={session.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, height: 0, overflow: 'hidden' }}
                  className="mb-1"
                >
                  <button
                    onClick={() => handleLoadSession(session)}
                    className={`w-full group flex items-start text-left gap-3 p-3 rounded-xl transition-all border ${
                      currentSessionId === session.id
                        ? 'bg-white border-nyaya-border shadow-sm'
                        : 'bg-transparent border-transparent hover:bg-white/50 hover:border-nyaya-border/50'
                    }`}
                  >
                    <MessageSquare size={16} className={`mt-0.5 shrink-0 ${currentSessionId === session.id ? 'text-nyaya-primary' : 'text-nyaya-muted'}`} />
                    <div className="flex-1 min-w-0">
                      <p className={`text-sm select-none truncate ${currentSessionId === session.id ? 'font-bold text-nyaya-text' : 'font-medium text-nyaya-text/80'}`}>
                        {session.title}
                      </p>
                      <p className="text-[11px] text-nyaya-muted mt-1 select-none">
                        {getRelativeTime(session.createdAt)}
                      </p>
                    </div>
                    <button
                      onClick={(e) => handleDelete(e, session)}
                      className="opacity-0 group-hover:opacity-100 p-1.5 text-nyaya-muted hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                      title="Delete Session"
                    >
                      <Trash2 size={14} />
                    </button>
                  </button>
                </motion.div>
              ))
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* Toggle Button */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="absolute -right-3.5 top-1/2 -translate-y-1/2 w-7 h-14 bg-white border border-nyaya-border rounded-full flex items-center justify-center text-nyaya-muted hover:text-nyaya-primary shadow-sm z-30 transition-colors"
      >
        {isExpanded ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
      </button>
    </div>
  );
}
