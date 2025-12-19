import React, { useState, useRef } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function PDFUploader() {
    const [files, setFiles] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [status, setStatus] = useState(null); // success | error
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFiles(Array.from(e.target.files));
            setStatus(null);
        }
    };

    const handleUpload = async () => {
        if (files.length === 0) return;

        setUploading(true);
        setStatus(null);

        const formData = new FormData();
        files.forEach((file) => {
            formData.append('files', file);
        });

        try {
            const response = await axios.post(`${API_URL}/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            console.log(response.data);
            setStatus('success');
            setFiles([]);
        } catch (error) {
            console.error('Upload failed:', error);
            setStatus('error');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="max-w-2xl mx-auto space-y-6">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 text-center">
                <div
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-slate-300 rounded-xl p-10 cursor-pointer hover:border-blue-500 hover:bg-blue-50/50 transition-colors group"
                >
                    <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                        <Upload size={32} />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-800">Upload Legal Documents</h3>
                    <p className="text-slate-500 mt-2 text-sm">Drag and drop PDF files here, or click to browse</p>
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        multiple
                        accept=".pdf"
                        className="hidden"
                    />
                </div>

                {/* File List */}
                {files.length > 0 && (
                    <div className="mt-6 space-y-2">
                        {files.map((file, idx) => (
                            <div key={idx} className="flex items-center gap-3 bg-slate-50 p-3 rounded-lg text-sm text-slate-700">
                                <FileText size={16} className="text-blue-500" />
                                <span className="flex-1 text-left truncate">{file.name}</span>
                                <span className="text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                            </div>
                        ))}
                    </div>
                )}

                {/* Action Button */}
                <div className="mt-8">
                    <button
                        onClick={handleUpload}
                        disabled={uploading || files.length === 0}
                        className={`w-full py-3 px-4 rounded-xl font-medium flex items-center justify-center gap-2 transition-all
              ${files.length === 0
                                ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                                : 'bg-blue-600 text-white hover:bg-blue-700 shadow-md hover:shadow-lg'
                            }`}
                    >
                        {uploading ? (
                            <>
                                <Loader className="animate-spin" size={20} />
                                Processing...
                            </>
                        ) : status === 'success' ? (
                            <>
                                <CheckCircle size={20} />
                                Upload Complete
                            </>
                        ) : status === 'error' ? (
                            <>
                                <AlertCircle size={20} />
                                Upload Failed
                            </>
                        ) : (
                            'Upload to Knowledge Base'
                        )}
                    </button>
                </div>
            </div>

            <div className="text-center text-sm text-slate-500">
                <p>Supports multiple PDF files. Documents are indexed locally for privacy.</p>
            </div>
        </div>
    );
}
