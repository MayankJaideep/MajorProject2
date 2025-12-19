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
            setPrediction(response.data.result);
        } catch (error) {
            console.error('Prediction error:', error);
            setError('Failed to generate prediction. Please ensure the API is running and try again.');
        } finally {
            setLoading(false);
        }
    };

    const chartData = prediction ? Object.entries(prediction.probabilities).map(([name, value]) => ({
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
                        <div className={`p-6 rounded-xl border mb-8 bg-gradient-to-br ${useAdvancedModel ? 'from-indigo-50 to-blue-50 border-indigo-100' : 'from-slate-50 to-gray-50 border-slate-200'}`}>
                            <div className="flex items-start justify-between">
                                <div>
                                    <p className="text-sm text-slate-500 font-medium uppercase tracking-wide">Predicted Outcome</p>
                                    <h2 className="text-3xl font-bold text-slate-900 mt-1 capitalize">
                                        {prediction.predicted_outcome.replace('_', ' ')}
                                    </h2>
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
                                        <span>Based on <strong>Stacking Ensemble</strong> (XGB+LGBM+RF) with BERT Embeddings</span>
                                    </>
                                ) : (
                                    <>
                                        <Clock size={16} className="text-slate-400" />
                                        <span>Based on <strong>Legacy Model</strong> (Metadata & rules)</span>
                                    </>
                                )}
                            </div>
                        </div>

                        {/* Chart */}
                        <div className="flex-1 min-h-[300px]">
                            <h4 className="text-sm font-medium text-slate-700 mb-4">Probability Distribution</h4>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                    <XAxis type="number" domain={[0, 100]} hide />
                                    <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 11 }} />
                                    <Tooltip
                                        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                        cursor={{ fill: 'transparent' }}
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
