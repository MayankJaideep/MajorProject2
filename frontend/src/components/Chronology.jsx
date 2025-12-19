import React, { useState } from 'react';
import { Clock, Calendar, FileText, Loader2, AlertCircle, CheckCircle2, ChevronRight } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function Chronology() {
    const [caseText, setCaseText] = useState('');
    const [loading, setLoading] = useState(false);
    const [timeline, setTimeline] = useState([]);
    const [message, setMessage] = useState('');
    const [error, setError] = useState(null);
    const [selectedEvent, setSelectedEvent] = useState(null);

    const handleGenerateTimeline = async () => {
        if (!caseText || caseText.trim().length < 50) {
            setError('Please enter at least 50 characters of case text');
            return;
        }

        setLoading(true);
        setError(null);
        setTimeline([]);
        setMessage('');

        try {
            const response = await axios.post(`${API_URL}/visualize/timeline`, {
                text: caseText
            });

            setTimeline(response.data.events || []);
            setMessage(response.data.message || '');

            if (response.data.events && response.data.events.length > 0) {
                setSelectedEvent(response.data.events[0]);
            }
        } catch (err) {
            console.error('Timeline generation error:', err);
            setError(err.response?.data?.detail || 'Failed to generate timeline. Please check your API connection.');
        } finally {
            setLoading(false);
        }
    };

    const handleClear = () => {
        setCaseText('');
        setTimeline([]);
        setMessage('');
        setError(null);
        setSelectedEvent(null);
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Panel: Input */}
            <div className="lg:col-span-1 space-y-6">
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                    <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
                        <FileText className="text-blue-600" size={20} />
                        Case Document
                    </h3>

                    <textarea
                        value={caseText}
                        onChange={(e) => setCaseText(e.target.value)}
                        rows={16}
                        className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:outline-none resize-none bg-slate-50 focus:bg-white transition-colors text-sm"
                        placeholder="Paste your FIR, Judgment, or any legal document containing dates and events...

Example:
'On 15th January 2023, the complainant filed FIR No. 234/2023 at City Police Station alleging theft. The incident occurred on 14th January 2023 at 3:00 PM...'"
                    />

                    {error && (
                        <div className="mt-3 p-3 bg-red-50 text-red-700 rounded-lg text-sm flex items-center gap-2 border border-red-100">
                            <AlertCircle size={16} />
                            {error}
                        </div>
                    )}

                    {message && !error && timeline.length > 0 && (
                        <div className="mt-3 p-3 bg-green-50 text-green-700 rounded-lg text-sm flex items-center gap-2 border border-green-100">
                            <CheckCircle2 size={16} />
                            {message}
                        </div>
                    )}

                    <div className="mt-4 flex gap-3">
                        <button
                            onClick={handleGenerateTimeline}
                            disabled={loading || !caseText}
                            className="flex-1 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-medium hover:shadow-lg hover:scale-[1.02] transition-all disabled:opacity-50 disabled:hover:scale-100 flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="animate-spin" size={18} />
                                    Extracting Events...
                                </>
                            ) : (
                                <>
                                    <Calendar size={18} />
                                    Generate Timeline
                                </>
                            )}
                        </button>

                        {timeline.length > 0 && (
                            <button
                                onClick={handleClear}
                                className="px-6 py-3 bg-slate-100 text-slate-700 rounded-xl font-medium hover:bg-slate-200 transition-all"
                            >
                                Clear
                            </button>
                        )}
                    </div>

                    <div className="mt-4 text-xs text-slate-500">
                        <p className="font-medium text-slate-600 mb-1">💡 Tips for best results:</p>
                        <ul className="list-disc list-inside space-y-1 ml-2">
                            <li>Include specific dates (DD/MM/YYYY format works best)</li>
                            <li>Paste complete paragraphs with context</li>
                            <li>Works with FIRs, Judgments, Case summaries</li>
                        </ul>
                    </div>
                </div>
            </div>

            {/* Right Panel: Timeline Visualization */}
            <div className="lg:col-span-2">
                {!loading && timeline.length === 0 ? (
                    <div className="h-full bg-slate-50 rounded-2xl border-2 border-dashed border-slate-200 flex flex-col items-center justify-center text-slate-400 p-12">
                        <Clock size={64} className="mb-4 opacity-50" />
                        <p className="font-medium text-lg">Ready to Visualize</p>
                        <p className="text-sm mt-1 text-center max-w-md">
                            Paste your case text and click "Generate Timeline" to extract and visualize chronological events
                        </p>
                    </div>
                ) : loading ? (
                    <div className="h-full bg-white rounded-2xl shadow-sm border border-slate-200 flex items-center justify-center p-12">
                        <div className="text-center">
                            <Loader2 className="animate-spin text-blue-600 mx-auto mb-4" size={48} />
                            <p className="text-slate-600 font-medium">Analyzing case chronology...</p>
                            <p className="text-sm text-slate-400 mt-2">Using AI to extract dates and events</p>
                        </div>
                    </div>
                ) : (
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 h-full flex flex-col">
                        <h3 className="text-lg font-semibold text-slate-800 mb-6 flex items-center gap-2">
                            <Calendar className="text-indigo-600" size={20} />
                            Case Chronology ({timeline.length} Events)
                        </h3>

                        <div className="flex-1 overflow-auto">
                            {/* Custom Vertical Timeline */}
                            <div className="relative">
                                {/* Vertical Line*/}
                                <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-slate-200"></div>

                                {timeline.map((event, index) => (
                                    <div key={index} className="relative mb-8 pl-20">
                                        {/* Date Circle */}
                                        <div className="absolute left-0 flex items-center">
                                            <div className={`w-16 h-16 rounded-full flex items-center justify-center ${selectedEvent === event
                                                    ? 'bg-blue-600 text-white ring-4 ring-blue-100'
                                                    : 'bg-white border-2 border-slate-300 text-slate-600'
                                                } shadow-md z-10 cursor-pointer transition-all hover:scale-110`}
                                                onClick={() => setSelectedEvent(event)}
                                            >
                                                <div className="text-center">
                                                    <div className="text-xs font-bold">
                                                        {event.date !== 'Unknown' && event.date.split('-')[2]}
                                                    </div>
                                                    <div className="text-[10px]">
                                                        {event.date !== 'Unknown'
                                                            ? new Date(event.date).toLocaleDateString('en-US', { month: 'short' })
                                                            : '?'
                                                        }
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Event Card */}
                                        <div
                                            className={`bg-slate-50 border rounded-xl p-4 cursor-pointer transition-all ${selectedEvent === event
                                                    ? 'border-blue-500 shadow-lg scale-[1.02]'
                                                    : 'border-slate-200 hover:border-blue-300 hover:shadow-md'
                                                }`}
                                            onClick={() => setSelectedEvent(event)}
                                        >
                                            <div className="flex items-start justify-between mb-2">
                                                <h4 className="font-semibold text-slate-800 flex-1">{event.title}</h4>
                                                {selectedEvent === event && (
                                                    <ChevronRight className="text-blue-600 flex-shrink-0" size={20} />
                                                )}
                                            </div>
                                            <div className="text-xs text-slate-500 mb-2">
                                                {event.date !== 'Unknown'
                                                    ? new Date(event.date).toLocaleDateString('en-US', {
                                                        weekday: 'long',
                                                        year: 'numeric',
                                                        month: 'long',
                                                        day: 'numeric'
                                                    })
                                                    : 'Date Unknown'
                                                }
                                            </div>
                                            <p className="text-sm text-slate-600 line-clamp-2">{event.description}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Selected Event Detail (Bottom) */}
                        {selectedEvent && (
                            <div className="mt-6 pt-6 border-t border-slate-200">
                                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-100 rounded-xl p-4">
                                    <div className="flex items-start justify-between mb-2">
                                        <h4 className="font-bold text-slate-900">{selectedEvent.title}</h4>
                                        <span className="text-xs bg-blue-600 text-white px-2 py-1 rounded-full">
                                            {selectedEvent.date !== 'Unknown'
                                                ? selectedEvent.date
                                                : 'Date Unknown'
                                            }
                                        </span>
                                    </div>
                                    <p className="text-sm text-slate-700 leading-relaxed">{selectedEvent.description}</p>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
