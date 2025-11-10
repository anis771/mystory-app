import React, { useState } from 'react';
import { Upload, BookOpen, Image, AlertCircle, Play } from 'lucide-react';

const JsonStoryViewer = () => {
    const [storyData, setStoryData] = useState(null);
    const [error, setError] = useState('');

    // Same styles as your original component
    const styles = {
        container: {
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #f3e8ff 0%, #dbeafe 100%)',
            padding: '24px'
        },
        maxWidth: {
            maxWidth: '896px',
            margin: '0 auto'
        },
        header: {
            textAlign: 'center',
            marginBottom: '32px'
        },
        title: {
            fontSize: '36px',
            fontWeight: 'bold',
            color: '#1f2937',
            marginBottom: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '12px'
        },
        subtitle: {
            color: '#6b7280'
        },
        card: {
            backgroundColor: 'white',
            borderRadius: '12px',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
            padding: '24px',
            marginBottom: '24px'
        },
        flexBetween: {
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '16px'
        },
        button: {
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 16px',
            borderRadius: '8px',
            border: 'none',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '500',
            transition: 'all 0.2s'
        },
        buttonPurple: {
            backgroundColor: '#7c3aed',
            color: 'white',
            padding: '12px 24px',
            marginTop: '20px'
        },
        uploadArea: {
            border: '2px dashed #d1d5db',
            borderRadius: '12px',
            padding: '40px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.2s',
            backgroundColor: '#f9fafb'
        },
        storyTitle: {
            fontSize: '28px',
            fontWeight: 'bold',
            color: '#1f2937',
            marginBottom: '24px',
            textAlign: 'center'
        },
        paragraphContainer: {
            borderBottom: '1px solid #f3f4f6',
            paddingBottom: '24px',
            marginBottom: '32px'
        },
        imageContainer: {
            marginBottom: '16px',
            backgroundColor: '#f3f4f6',
            borderRadius: '8px',
            height: '512px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            overflow: 'hidden'
        },
        image: {
            width: '100%',
            height: '100%',
            objectFit: 'cover'
        },
        paragraphText: {
            color: '#374151',
            lineHeight: '1.7',
            fontSize: '18px',
            marginBottom: '8px'
        },
        details: {
            fontSize: '14px',
            color: '#6b7280'
        },
        summary: {
            cursor: 'pointer'
        },
        italic: {
            fontStyle: 'italic',
            marginTop: '4px'
        },
        errorMessage: {
            backgroundColor: '#fee2e2',
            border: '1px solid #fecaca',
            borderRadius: '8px',
            padding: '12px',
            color: '#dc2626',
            marginTop: '16px'
        }
    };

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const jsonData = JSON.parse(e.target.result);
                // Transform the JSON data to match your component's expected format
                const transformedData = transformJsonData(jsonData);
                setStoryData(transformedData);
                setError('');
            } catch (err) {
                setError('Invalid JSON file. Please check the format.');
                setStoryData(null);
            }
        };
        reader.readAsText(file);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const jsonData = JSON.parse(event.target.result);
                    const transformedData = transformJsonData(jsonData);
                    setStoryData(transformedData);
                    setError('');
                } catch (err) {
                    setError('Invalid JSON file. Please check the format.');
                    setStoryData(null);
                }
            };
            reader.readAsText(file);
        }
    };

    const transformJsonData = (jsonData) => {
        // Transform your JSON structure to match the component's expected format
        return {
            title: jsonData.title,
            context: jsonData.context,
            experiment_id: jsonData.id,
            paragraphs: jsonData.paragraphs.map(p => ({
                id: p.id,
                text: p.text,
                // For compare mode (3 methods)
                images: p.results ? Object.entries(p.results).map(([method, result]) => ({
                    method: method,
                    image_prompt: result.prompt,
                    image_url: result.image_url,
                    image_error: !result.image_url
                })) : null,
                // For single mode fallback
                image_prompt: p.results?.simple?.prompt || p.results?.[Object.keys(p.results)[0]]?.prompt,
                image_url: p.results?.simple?.image_url || p.results?.[Object.keys(p.results)[0]]?.image_url,
                image_error: !p.results?.simple?.image_url && !p.results?.[Object.keys(p.results)[0]]?.image_url
            }))
        };
    };

    return (
        <div style={styles.container}>
            <div style={styles.maxWidth}>
                {/* Header */}
                <div style={styles.header}>
                    <h1 style={styles.title}>
                        <BookOpen size={40} color="#7c3aed" />
                        JSON Story Viewer
                    </h1>
                    <p style={styles.subtitle}>View saved story experiments with comparison images</p>
                </div>

                {/* Upload Area */}
                {!storyData && (
                    <div style={styles.card}>
                        <div 
                            style={styles.uploadArea}
                            onDrop={handleDrop}
                            onDragOver={(e) => e.preventDefault()}
                            onDragEnter={(e) => e.preventDefault()}
                            onClick={() => document.getElementById('fileInput').click()}
                        >
                            <Upload size={48} color="#7c3aed" style={{ marginBottom: '16px' }} />
                            <h3 style={{ marginBottom: '8px', color: '#374151' }}>Load Story JSON</h3>
                            <p style={{ color: '#6b7280', marginBottom: '16px' }}>
                                Drop your story experiment JSON file here or click to browse
                            </p>
                            <button style={{ ...styles.button, ...styles.buttonPurple }}>
                                <Upload size={16} />
                                Choose File
                            </button>
                        </div>
                        <input
                            id="fileInput"
                            type="file"
                            accept=".json"
                            style={{ display: 'none' }}
                            onChange={handleFileUpload}
                        />
                        {error && (
                            <div style={styles.errorMessage}>
                                <AlertCircle size={16} style={{ marginRight: '8px', display: 'inline' }} />
                                {error}
                            </div>
                        )}
                    </div>
                )}

                {/* Generated Story - Exact same layout as your original */}
                {storyData && (
                    <div style={styles.card}>
                        <div style={styles.flexBetween}>
                            <h2 style={styles.storyTitle}>{storyData.title}</h2>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                <button
                                    onClick={() => setStoryData(null)}
                                    style={{
                                        ...styles.button,
                                        ...styles.buttonPurple,
                                        padding: '8px 16px'
                                    }}
                                >
                                    <Upload size={16} />
                                    Load New File
                                </button>
                            </div>
                        </div>

                        {storyData.paragraphs.map((paragraph, index) => (
                            <div
                                key={paragraph.id}
                                style={{
                                    ...styles.paragraphContainer,
                                    ...(index === storyData.paragraphs.length - 1 ? { borderBottom: 'none' } : {})
                                }}
                            >
                                {/* Image container - exactly like your original */}
                                <div style={styles.imageContainer}>
                                    {paragraph.images ? (
                                        // Compare mode - 3-wide grid for variants
                                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', width: '100%', height: '100%' }}>
                                            {paragraph.images.map((img, i) => (
                                                <div key={i} style={{ position: 'relative', backgroundColor: '#eef2ff', borderRadius: '8px', overflow: 'hidden' }}>
                                                    {img.image_url ? (
                                                        <img src={img.image_url} alt={`${img.method} variant`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                                    ) : img.image_error ? (
                                                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '6px', color: '#ef4444', padding: '8px', textAlign: 'center' }}>
                                                            <AlertCircle size={18} />
                                                            <span style={{ fontSize: '12px' }}>Failed ({img.method})</span>
                                                        </div>
                                                    ) : (
                                                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#9ca3af' }}>
                                                            <Image size={24} />
                                                        </div>
                                                    )}
                                                    <div style={{ position: 'absolute', left: 8, bottom: 8, background: 'rgba(0,0,0,0.5)', color: '#fff', padding: '2px 6px', borderRadius: '6px', fontSize: '12px' }}>
                                                        {img.method}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        // Single mode
                                        paragraph.image_url ? (
                                            <img
                                                src={paragraph.image_url}
                                                alt={`Illustration for paragraph ${paragraph.id}`}
                                                style={styles.image}
                                                onError={(e) => { e.target.onerror = null; e.target.style.display = 'none'; }}
                                            />
                                        ) : paragraph.image_error ? (
                                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', color: '#ef4444' }}>
                                                <AlertCircle size={48} />
                                                <span style={{ fontSize: '14px' }}>Failed to generate image</span>
                                            </div>
                                        ) : (
                                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', color: '#9ca3af' }}>
                                                <Image size={48} />
                                                <span style={{ fontSize: '14px' }}>Image will appear here</span>
                                            </div>
                                        )
                                    )}
                                </div>

                                <p style={styles.paragraphText}>
                                    {paragraph.text}
                                </p>

                                {/* Image prompt(s) - exactly like your original */}
                                <details style={styles.details}>
                                    <summary style={styles.summary}>Image prompt{paragraph.images ? 's' : ''}</summary>
                                    {paragraph.images ? (
                                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginTop: '8px' }}>
                                            {paragraph.images.map((img, i) => (
                                                <div key={i} style={{ fontSize: '13px' }}>
                                                    <div style={{ fontWeight: '600', marginBottom: '4px' }}>{img.method}</div>
                                                    <div style={styles.italic}>"{img.image_prompt}"</div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <p style={styles.italic}>"{paragraph.image_prompt}"</p>
                                    )}
                                </details>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default JsonStoryViewer;