import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Book, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import { motion } from 'framer-motion';

const API_URL = 'http://localhost:8000';

export default function ChatInterface() {
    const [messages, setMessages] = useState([
        { role: 'assistant', content: 'Hello! I am your Legal AI Assistant. How can I help you with your research today?' }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            // Format history for API
            const history = messages.map(m => ({ role: m.role, content: m.content }));

            const response = await axios.post(`${API_URL}/chat`, {
                message: userMessage.content,
                history: history
            });

            const aiMessage = {
                role: 'assistant',
                content: response.data.response,
                sources: response.data.sources
            };

            setMessages(prev => [...prev, aiMessage]);
        } catch (error) {
            setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error extracting information.' }]);
            console.error('Chat error:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-[calc(100vh-8rem)] bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages.map((msg, idx) => (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        key={idx}
                        className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                    >
                        {/* Avatar */}
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 
              ${msg.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-green-600 text-white'}`}>
                            {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                        </div>

                        {/* Bubble */}
                        <div className={`max-w-[70%] space-y-2`}>
                            <div className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm
                ${msg.role === 'user'
                                    ? 'bg-gradient-to-br from-indigo-500 to-blue-600 text-white rounded-tr-none'
                                    : 'bg-slate-100 text-slate-800 rounded-tl-none'
                                }`}>
                                {msg.role === 'assistant' ? (
                                    <div className="prose prose-sm max-w-none prose-p:my-1 prose-headings:my-2 prose-ul:my-1">
                                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                                    </div>
                                ) : (
                                    <p>{msg.content}</p>
                                )}
                            </div>

                            {/* Sources */}
                            {msg.sources && msg.sources.length > 0 && (
                                <div className="flex flex-wrap gap-2 mt-2">
                                    {msg.sources.map((src, i) => (
                                        <span key={i} className="inline-flex items-center gap-1 px-2 py-1 bg-slate-50 border border-slate-200 rounded text-xs text-slate-500">
                                            <Book size={12} />
                                            {src.type}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </div>
                    </motion.div>
                ))}
                {loading && (
                    <div className="flex gap-4">
                        <div className="w-10 h-10 rounded-full bg-green-600 text-white flex items-center justify-center">
                            <Bot size={20} />
                        </div>
                        <div className="bg-slate-50 p-4 rounded-2xl rounded-tl-none">
                            <Loader2 className="animate-spin text-slate-400" size={20} />
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Handling */}
            <div className="p-4 border-t border-slate-100 bg-slate-50">
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Ask a legal question..."
                        className="flex-1 px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white shadow-sm"
                    />
                    <button
                        onClick={handleSend}
                        disabled={loading || !input.trim()}
                        className="bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-md"
                    >
                        <Send size={20} />
                    </button>
                </div>
            </div>
        </div>
    );
}
