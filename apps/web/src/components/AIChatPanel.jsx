import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, Bot, User, Loader2, Sparkles, RefreshCw, Trash2, AlertCircle,
} from 'lucide-react';

const ML_BASE = '/ml';
const AUTH_STORAGE_KEY = 'gms.auth';

function getTokenFromStorage() {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed?.token || null;
  } catch {
    return null;
  }
}

/* ─── Single chat bubble ─────────────────────────────────────── */
function ChatBubble({ role, content, isStreaming }) {
  const isUser = role === 'user';
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.2 }}
      className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg">
          <Bot size={16} className="text-white" />
        </div>
      )}
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap
          ${isUser
            ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-br-md shadow-lg'
            : 'bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 rounded-bl-md shadow'
          }`}
      >
        {content}
        {isStreaming && (
          <span className="inline-block w-1.5 h-4 ml-0.5 bg-indigo-400 animate-pulse rounded-sm" />
        )}
      </div>
      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center shadow-lg">
          <User size={16} className="text-white" />
        </div>
      )}
    </motion.div>
  );
}

/* ─── Suggestion chip ────────────────────────────────────────── */
function SuggestionChip({ text, onClick }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 text-xs font-medium rounded-full border border-indigo-200
                 bg-white/80 text-indigo-600 hover:bg-indigo-50 hover:border-indigo-300
                 transition-all duration-200 shadow-sm hover:shadow"
    >
      <Sparkles size={12} className="inline mr-1" />
      {text}
    </button>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN CHAT PANEL
   ═══════════════════════════════════════════════════════════════ */
export default function AIChatPanel({ sessionId = 'default', className = '' }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load suggestions on mount
  useEffect(() => {
    const token = getTokenFromStorage();
    fetch(`${ML_BASE}/chat/suggestions`, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    })
      .then((r) => r.json())
      .then((d) => setSuggestions(d.suggestions || []))
      .catch(() => {});
  }, []);

  /* ─── Send message (streaming via SSE) ────────────────────── */
  const sendMessage = useCallback(async (text) => {
    const msg = (text || input).trim();
    if (!msg || isStreaming) return;
    setInput('');
    setError(null);

    // Add user message
    setMessages((prev) => [...prev, { role: 'user', content: msg }]);

    // Add placeholder for AI response
    setMessages((prev) => [...prev, { role: 'assistant', content: '', streaming: true }]);
    setIsStreaming(true);

    try {
      const token = getTokenFromStorage();
      const response = await fetch(`${ML_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ message: msg, session_id: sessionId, stream: true }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accum = '';
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Parse SSE lines
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const payload = line.slice(6).trim();
            if (payload === '[DONE]') continue;
            try {
              const parsed = JSON.parse(payload);
              if (parsed.token) {
                accum += parsed.token;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = { role: 'assistant', content: accum, streaming: true };
                  return updated;
                });
              }
              if (parsed.error) {
                throw new Error(parsed.error);
              }
            } catch {
              // non-JSON SSE line (e.g. keep-alive)
            }
          }
        }
      }

      // Finalize the AI message
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: 'assistant', content: accum || 'No response received.', streaming: false };
        return updated;
      });
    } catch (e) {
      // If streaming fails, try non-streaming fallback
      try {
        const token2 = getTokenFromStorage();
        const resp = await fetch(`${ML_BASE}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token2 ? { Authorization: `Bearer ${token2}` } : {}),
          },
          body: JSON.stringify({ message: msg, session_id: sessionId, stream: false }),
        });
        if (resp.ok) {
          const data = await resp.json();
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { role: 'assistant', content: data.response, streaming: false };
            return updated;
          });
        } else {
          throw e;
        }
      } catch {
        setError(e.message || 'Failed to connect to AI service');
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'assistant',
            content: `Sorry, I couldn't process that request. ${e.message || 'Please check that the ML service and Ollama are running.'}`,
            streaming: false,
          };
          return updated;
        });
      }
    } finally {
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  }, [input, isStreaming, sessionId]);

  /* ─── Clear chat ──────────────────────────────────────────── */
  const clearChat = useCallback(async () => {
    setMessages([]);
    setError(null);
    try {
      const token = getTokenFromStorage();
      await fetch(`${ML_BASE}/chat/${sessionId}`, {
        method: 'DELETE',
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
    } catch {
      // silent
    }
  }, [sessionId]);

  /* ─── Key handler ─────────────────────────────────────────── */
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className={`flex flex-col bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl overflow-hidden ${className}`}>
      {/* ─── Header ──────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-5 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">
            <Bot size={18} />
          </div>
          <div>
            <h3 className="font-bold text-sm">FitFlex AI</h3>
            <p className="text-[10px] opacity-80">
              {isStreaming ? 'Thinking...' : 'RAG-powered fitness assistant'}
            </p>
          </div>
        </div>
        <div className="flex gap-1">
          <button
            onClick={clearChat}
            className="p-1.5 rounded-lg hover:bg-white/20 transition"
            title="Clear chat"
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {/* ─── Messages ────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-[300px] max-h-[500px]">
        {messages.length === 0 && (
          <div className="text-center py-8">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-indigo-100 to-purple-100 flex items-center justify-center">
              <Sparkles size={28} className="text-indigo-500" />
            </div>
            <h4 className="font-semibold text-slate-700 mb-1">Ask FitFlex AI anything</h4>
            <p className="text-xs text-slate-400 mb-4">
              I'm trained on your gym's data — workouts, members, performance, and more.
            </p>
            {suggestions.length > 0 && (
              <div className="flex flex-wrap justify-center gap-2">
                {suggestions.map((s, i) => (
                  <SuggestionChip key={i} text={s} onClick={() => sendMessage(s)} />
                ))}
              </div>
            )}
          </div>
        )}

        <AnimatePresence>
          {messages.map((msg, i) => (
            <ChatBubble
              key={i}
              role={msg.role}
              content={msg.content}
              isStreaming={msg.streaming}
            />
          ))}
        </AnimatePresence>

        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-50 text-red-600 text-xs"
          >
            <AlertCircle size={14} />
            {error}
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* ─── Input ───────────────────────────────────────────── */}
      <div className="border-t border-slate-200 dark:border-slate-700 p-3">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isStreaming ? 'Wait for response...' : 'Ask about workouts, nutrition, progress...'}
            disabled={isStreaming}
            rows={1}
            className="flex-1 resize-none rounded-xl border border-slate-200 dark:border-slate-600
                       bg-slate-50 dark:bg-slate-800 px-4 py-2.5 text-sm
                       focus:ring-2 focus:ring-indigo-400 focus:border-transparent
                       disabled:opacity-50 transition placeholder:text-slate-400"
            style={{ minHeight: '42px', maxHeight: '120px' }}
            onInput={(e) => {
              e.target.style.height = 'auto';
              e.target.style.height = e.target.scrollHeight + 'px';
            }}
          />
          <button
            onClick={() => sendMessage()}
            disabled={isStreaming || !input.trim()}
            className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600
                       text-white flex items-center justify-center
                       hover:from-indigo-600 hover:to-purple-700 disabled:opacity-40
                       transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            {isStreaming ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
          </button>
        </div>
      </div>
    </div>
  );
}
