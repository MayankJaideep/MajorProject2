import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

const ChatHistoryContext = createContext(null);

export const ChatHistoryProvider = ({ children }) => {
  const { user, isAuthenticated } = useAuth();
  const [chatSessions, setChatSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [sessionsLoaded, setSessionsLoaded] = useState(false);

  const getChatKey = (userId) => `lumina_chats_${userId}`;

  useEffect(() => {
    if (isAuthenticated && user?.id) {
      try {
        const raw = localStorage.getItem(getChatKey(user.id));
        console.log('[LOAD] auth changed, loading for userId:', user?.id, 'raw from storage:', raw?.slice(0, 100));
        const saved = raw ? JSON.parse(raw) : [];
        setChatSessions(Array.isArray(saved) ? saved.slice(0, 10) : []);
        console.log('[LOAD] found:', Array.isArray(saved) ? saved.length : 0, 'sessions');
      } catch(e) {
        console.error('[LOAD] parse error:', e);
        setChatSessions([]);
      }
      setSessionsLoaded(true);
    } else if (!isAuthenticated) {
      setChatSessions([]);
      setSessionsLoaded(false);
    }
  }, [isAuthenticated, user?.id]);

  const saveSession = (messages, exactSessionId = null) => {
    const activeId = exactSessionId || currentSessionId;
    console.log('[SAVE] called with', messages.length, 'messages, activeId:', activeId, 'userId:', user?.id);
    if (!isAuthenticated || !user?.id) return;
    if (messages.length < 2) return;

    // If no current session ID, create one now
    let sessionId = activeId;
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
      setCurrentSessionId(sessionId);
    }

    const realMessages = messages.filter(m => !m.isSystem);
    if (realMessages.length < 2) return;

    const firstUserMsg = realMessages.find(m => m.role === 'user');
    const title = firstUserMsg ? firstUserMsg.content.slice(0, 40) : 'Untitled Session';

    setChatSessions(prev => {
      const existing = prev.find(s => s.id === sessionId);
      const updated = [
        {
          id: sessionId,
          title,
          createdAt: existing?.createdAt ?? new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          messages: realMessages
        },
        ...prev.filter(s => s.id !== sessionId)
      ].slice(0, 10);

      // Write synchronously inside the setter
      localStorage.setItem(getChatKey(user.id), JSON.stringify(updated));
      console.log('[SAVE] wrote to key:', getChatKey(user.id), 'sessions count:', updated.length);
      return updated;
    });
  };

  const loadSession = (sessionId) => {
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
      setCurrentSessionId(sessionId);
      return session.messages;
    }
    return null;
  };

  const startNewSession = () => {
    const newId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    setCurrentSessionId(newId);
    return newId;
  };

  const deleteSession = (sessionId) => {
    setChatSessions(prev => {
      const updated = prev.filter(s => s.id !== sessionId);
      if (user?.id) {
        localStorage.setItem(getChatKey(user.id), JSON.stringify(updated));
      }
      return updated;
    });
    if (currentSessionId === sessionId) {
      startNewSession(); // The UI component will handle dispatching new-chat event
    }
  };

  return (
    <ChatHistoryContext.Provider value={{
      chatSessions,
      currentSessionId,
      setCurrentSessionId,
      saveSession,
      loadSession,
      startNewSession,
      deleteSession
    }}>
      {children}
    </ChatHistoryContext.Provider>
  );
};

export const useChatHistory = () => {
  const context = useContext(ChatHistoryContext);
  if (!context) {
    throw new Error('useChatHistory must be used within a ChatHistoryProvider');
  }
  return context;
};
