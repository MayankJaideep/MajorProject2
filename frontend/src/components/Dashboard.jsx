import React, { useState } from 'react';
import { Activity, Gavel, Scale, AlertTriangle, CheckCircle2, Zap, Clock, Loader2, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function Dashboard() {
    const [formData, setFormData] = useState({
        description: '',
        court: '',
        judge: '',
        case_type: ''
    });
    const [useAdvancedModel, setUseAdvancedModel] = useState(true);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handlePredict = async () => {
        if (!formData.description) return;
        setLoading(true);
        setPrediction(null);
        setError(null);

        try {
            const response = await axios.post(`${API_URL}/predict`, {
                ...formData,
                model_version: useAdvancedModel ? 'advanced' : 'legacy'
            });
            if (response.data.result.error) {
                setError(response.data.result.error);
                setPrediction(null);
            } else {
                setPrediction(response.data.result);
            }
        } catch (error) {
            console.error('Prediction error:', error);
            setError('Failed to generate prediction. Please ensure the API is running and try again.');
        } finally {
            setLoading(false);
        }
    };

    const chartData = (prediction && prediction.probabilities) ? Object.entries(prediction.probabilities).map(([name, value]) => ({
        name: name.replace('_', ' ').toUpperCase(),
        value: parseFloat((value * 100).toFixed(1))
    })) : [];

    const COLORS = ['#4f46e5', '#10b981', '#f59e0b', '#ef4444'];

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Input Section */}
            <div className="space-y-6">
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                            <Scale className="text-blue-600" size={20} />
                            Case Details
                        </h3>

                        {/* Model Toggle */}
                        <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-lg">
                            <button
                                onClick={() => setUseAdvancedModel(false)}
                                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${!useAdvancedModel ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                            >
                                Legacy (Regex)
                            </button>
                            <button
                                onClick={() => setUseAdvancedModel(true)}
                                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${useAdvancedModel ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                            >
                                Advanced (AI)
                            </button>
                        </div>
                    </div>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Case Description <span className="text-red-500">*</span></label>
                            <textarea
                                rows={6}
                                value={formData.description}
                                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                                className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:outline-none resize-none bg-slate-50 focus:bg-white transition-colors"
                                placeholder="Enter detailed summary of the case facts..."
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-1">Court</label>
                                <input
                                    type="text"
                                    value={formData.court}
                                    onChange={(e) => setFormData({ ...formData, court: e.target.value })}
                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    placeholder="e.g. Supreme Court"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-1">Judge</label>
                                <input
                                    type="text"
                                    value={formData.judge}
                                    onChange={(e) => setFormData({ ...formData, judge: e.target.value })}
                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    placeholder="Name of Judge"
                                />
                            </div>
                        </div>

                        {error && (
                            <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm flex items-center gap-2 border border-red-100">
                                <AlertCircle size={16} />
                                {error}
                            </div>
                        )}

                        <button
                            onClick={handlePredict}
                            disabled={loading || !formData.description}
                            className="w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-medium hover:shadow-lg hover:scale-[1.02] transition-all disabled:opacity-50 disabled:hover:scale-100 flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="animate-spin" size={20} />
                                    Analyzing Features...
                                </>
                            ) : (
                                <>
                                    <Zap size={18} />
                                    Predict Outcome
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Results Section */}
            <div className="space-y-6">
                {prediction ? (
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 h-full flex flex-col">
                        <h3 className="text-lg font-semibold text-slate-800 mb-6 flex items-center gap-2">
                            <Activity className="text-indigo-600" size={20} />
                            Analysis Result
                        </h3>

                        {/* Main Result Card */}
                        <div className={`p-6 rounded-xl border mb-6 bg-gradient-to-br ${useAdvancedModel ? 'from-indigo-50 to-blue-50 border-indigo-100' : 'from-slate-50 to-gray-50 border-slate-200'}`}>
                            <div className="flex items-start justify-between">
                                <div>
                                    <p className="text-sm text-slate-500 font-medium uppercase tracking-wide">Predicted Outcome</p>
                                    <h2 className="text-3xl font-bold text-slate-900 mt-1 capitalize">
                                        {prediction.predicted_outcome.replace('_', ' ')}
                                    </h2>
                                    <p className="text-lg text-slate-600 mt-2">
                                        Confidence: <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
                                    </p>
                                </div>
                                <div className={`px-4 py-2 rounded-full text-sm font-bold shadow-sm
                  ${prediction.confidence_level === 'High' ? 'bg-green-100 text-green-700' :
                                        prediction.confidence_level === 'Medium' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                                    {prediction.confidence_level} Confidence
                                </div>
                            </div>
                            <div className="mt-4 flex items-center gap-2 text-sm text-slate-600">
                                {useAdvancedModel ? (
                                    <>
                                        <Zap size={16} className="text-indigo-500" />
                                        <span>Based on <strong>Stacking Ensemble</strong> (XGB+LGBM+RF) with InLegalBERT (88% Accuracy)</span>
                                    </>
                                ) : (
                                    <>
                                        <Clock size={16} className="text-slate-400" />
                                        <span>Based on <strong>Legacy Model</strong> (Metadata & rules)</span>
                                    </>
                                )}
                            </div>
                        </div>

                        {/* Legal Metrics Grid */}
                        {prediction.legal_metrics && (
                            <div className="grid grid-cols-2 gap-3 mb-6">
                                <div className="bg-gradient-to-br from-green-50 to-emerald-50 border border-green-100 rounded-lg p-4">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs font-semibold text-green-700 uppercase tracking-wide">Petitioner Win</span>
                                        <CheckCircle2 size={16} className="text-green-600" />
                                    </div>
                                    <p className="text-2xl font-bold text-green-900">{prediction.legal_metrics.petitioner_win_probability}%</p>
                                    <p className="text-xs text-green-600 mt-1">Appeal/Petition Success</p>
                                </div>

                                <div className="bg-gradient-to-br from-red-50 to-rose-50 border border-red-100 rounded-lg p-4">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs font-semibold text-red-700 uppercase tracking-wide">Dismissal Risk</span>
                                        <AlertTriangle size={16} className="text-red-600" />
                                    </div>
                                    <p className="text-2xl font-bold text-red-900">{prediction.legal_metrics.case_dismissal_risk}%</p>
                                    <p className="text-xs text-red-600 mt-1">Case Rejected/Denied</p>
                                </div>

                                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-100 rounded-lg p-4">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs font-semibold text-blue-700 uppercase tracking-wide">Respondent Win</span>
                                        <Gavel size={16} className="text-blue-600" />
                                    </div>
                                    <p className="text-2xl font-bold text-blue-900">{prediction.legal_metrics.respondent_win_probability}%</p>
                                    <p className="text-xs text-blue-600 mt-1">Defense Success</p>
                                </div>

                                <div className="bg-gradient-to-br from-purple-50 to-violet-50 border border-purple-100 rounded-lg p-4">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs font-semibold text-purple-700 uppercase tracking-wide">Appeal Success</span>
                                        <Scale size={16} className="text-purple-600" />
                                    </div>
                                    <p className="text-2xl font-bold text-purple-900">{prediction.legal_metrics.appeal_success_rate}%</p>
                                    <p className="text-xs text-purple-600 mt-1">If Allowed/Granted</p>
                                </div>
                            </div>
                        )}

                        {/* Probability Table */}
                        <div className="mb-6 bg-slate-50 rounded-lg p-4">
                            <h4 className="text-sm font-semibold text-slate-700 mb-3">Outcome Probabilities</h4>
                            <div className="space-y-2">
                                {chartData.sort((a, b) => b.value - a.value).map((item, idx) => (
                                    <div key={idx} className="flex items-center justify-between">
                                        <span className="text-sm font-medium text-slate-700 capitalize">{item.name}</span>
                                        <div className="flex items-center gap-3">
                                            <div className="w-32 bg-slate-200 rounded-full h-2">
                                                <div
                                                    className="h-2 rounded-full transition-all duration-500"
                                                    style={{ width: `${item.value}%`, backgroundColor: COLORS[idx % COLORS.length] }}
                                                />
                                            </div>
                                            <span className="text-sm font-bold text-slate-900 w-12 text-right">{item.value}%</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Chart */}
                        <div className="flex-1 min-h-[240px]">
                            <h4 className="text-sm font-medium text-slate-700 mb-4">Visual Distribution</h4>
                            <ResponsiveContainer width="100%" height={240}>
                                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                    <XAxis type="number" domain={[0, 100]} unit="%" />
                                    <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 11 }} />
                                    <Tooltip
                                        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                        cursor={{ fill: 'transparent' }}
                                        formatter={(value) => `${value}%`}
                                    />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={32} animationDuration={1500}>
                                        {chartData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                ) : (
                    <div className="h-full bg-slate-50 rounded-2xl border-2 border-dashed border-slate-200 flex flex-col items-center justify-center text-slate-400 p-8">
                        <Gavel size={48} className="mb-4 opacity-50" />
                        <p className="font-medium">Ready to Analyze</p>
                        <p className="text-sm mt-1">Enter details and select a model to generate prediction</p>
                    </div>
                )}
            </div>
        </div>
    );
}
