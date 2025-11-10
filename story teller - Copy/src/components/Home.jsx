import '../App.css'
import { useState, useEffect } from 'react';
import { Play, BookOpen, Image, Server, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

const StoryGenerator = () => {
    const [serverStatus, setServerStatus] = useState('disconnected');
    const [modelWarmed, setModelWarmed] = useState(false);
    const [userPrompt, setUserPrompt] = useState('');
    const [numParagraphs, setNumParagraphs] = useState(5);
    const [story, setStory] = useState(null);
    const [loading, setLoading] = useState(false);
    const [currentStep, setCurrentStep] = useState('');
    const [generatingImages, setGeneratingImages] = useState({});
    const [imageGenerationProgress, setImageGenerationProgress] = useState(0);
    const [hasGeneratedImages, setHasGeneratedImages] = useState(false);
    // ADD state for extractor mode
    const [extractorMode, setExtractorMode] = useState('single'); // 'single' or 'compare'
    const [extractor, setExtractor] = useState("simple"); // used only when single

    const API_BASE = 'http://localhost:5000/api';

    // Styles object
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
        flexCenter: {
            display: 'flex',
            alignItems: 'center',
            gap: '12px'
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
        buttonGreen: {
            backgroundColor: '#10b981',
            color: 'white'
        },
        buttonGreenHover: {
            backgroundColor: '#059669'
        },
        buttonPurple: {
            backgroundColor: '#7c3aed',
            color: 'white',
            padding: '12px 24px',
            marginTop: '20px'
        },
        buttonPurpleHover: {
            backgroundColor: '#6d28d9'
        },
        buttonDisabled: {
            backgroundColor: '#d1d5db',
            cursor: 'not-allowed'
        },
        statusBanner: {
            backgroundColor: '#dbeafe',
            border: '1px solid #93c5fd',
            borderRadius: '8px',
            padding: '12px',
            color: '#1e40af'
        },
        formGroup: {
            marginBottom: '16px'
        },
        label: {
            display: 'block',
            fontSize: '14px',
            fontWeight: '500',
            color: '#374151',
            marginBottom: '8px'
        },
        textarea: {
            width: '100%',
            padding: '12px 16px',
            border: '1px solid #d1d5db',
            borderRadius: '8px',
            fontSize: '16px',
            resize: 'none',
            outline: 'none',
            transition: 'border-color 0.2s',
            boxSizing: 'border-box'
        },
        textareaFocus: {
            borderColor: '#7c3aed',
            boxShadow: '0 0 0 3px rgba(124, 58, 237, 0.1)'
        },
        select: {
            padding: '8px 12px',
            border: '1px solid #d1d5db',
            borderRadius: '8px',
            fontSize: '14px',
            outline: 'none'
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
        }
    };

    // Check server status on component mount
    useEffect(() => {
        checkServerStatus();
    }, []);

    useEffect(() => {
        if (story && !hasGeneratedImages) {
            generateAllImages(story.paragraphs);
            setHasGeneratedImages(true);
        }
    }, [story]);

    const checkServerStatus = async () => {
        try {
            const response = await fetch(`${API_BASE}/status`);
            const data = await response.json();
            setServerStatus(data.status);
            setModelWarmed(data.model_warmed);
        } catch (error) {
            setServerStatus('error');
            console.error('Server status check failed:', error);
        }
    };

    const warmupServer = async () => {
        setLoading(true);
        setCurrentStep('Warming up AI model...');

        try {
            const response = await fetch(`${API_BASE}/warmup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();

            if (data.status === 'success') {
                setModelWarmed(true);
                setCurrentStep('✅ Model ready!');
            } else {
                setCurrentStep('❌ Warmup failed');
                console.error('Warmup failed:', data.message);
            }
        } catch (error) {
            setCurrentStep('❌ Connection failed');
            console.error('Warmup error:', error);
        } finally {
            setLoading(false);
            setTimeout(() => setCurrentStep(''), 3000);
        }
    };

    const generateStory = async () => {
        if (!userPrompt.trim()) {
            alert('Please enter a story prompt!');
            return;
        }

        setHasGeneratedImages(false);
        setLoading(true);
        setCurrentStep('🤖 AI is crafting your story...');

        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
            /* 
            if you want to just generate story and dont want to save result then change the 
            end point to  'generate-story'
            */
            const response = await fetch(`${API_BASE}/compare-story`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: userPrompt,
                    paragraphs: numParagraphs,
                    mode: extractorMode,
                    extractor: extractor
                }),
                signal: controller.signal
            });
            clearTimeout(timeout);
            const data = await response.json();

            if (data.status === 'success') {
                // Create a new story object with initialized image fields
                const newStory = {
                    ...data.story,
                    paragraphs: data.story.paragraphs.map(p => ({
                        ...p,
                        image_url: null,
                        image_error: false
                    })),
                    experiment_id: data.experiment_id
                };

                setStory(newStory);
                setCurrentStep('✅ Story generated! Now creating images...');
            } else {
                setCurrentStep(`❌ Story generation failed: ${data.message}`);
            }
        } catch (error) {
            setCurrentStep('❌ Failed to generate story');
            console.error('Story generation error:', error);
        } finally {
            setLoading(false);
        }
    };


    const generateAllImages = async (paragraphs) => {
        if (!story) return;

        setImageGenerationProgress(0);
        const totalWork = extractorMode === 'compare'
            ? paragraphs.length * 3
            : paragraphs.length;
        let done = 0;

        // fresh copy
        const updatedStory = {
            ...story,
            paragraphs: story.paragraphs.map(p => ({ ...p }))
        };

        for (const p of paragraphs) {
            if (extractorMode === 'compare') {
                // ensure images array exists
                const idx = updatedStory.paragraphs.findIndex(x => x.id === p.id);
                if (idx < 0) continue;

                updatedStory.paragraphs[idx].images = (updatedStory.paragraphs[idx].images || []).map(img => ({ ...img }));

                for (let i = 0; i < updatedStory.paragraphs[idx].images.length; i++) {
                    const variant = updatedStory.paragraphs[idx].images[i];
                    try {
                        const result = await generateImageVariant(p.id, variant.image_prompt, variant.method);
                        console.log("result is ", result);

                        updatedStory.paragraphs[idx].images[i] = {
                            ...variant,
                            image_url: result?.image_url || null,
                            image_error: !result?.image_url
                        };

                        console.log("updated story object will be ", updatedStory.paragraphs[idx].images[i]);

                    } catch (e) {
                        updatedStory.paragraphs[idx].images[i] = {
                            ...variant,
                            image_error: true
                        };
                    } finally {
                        done += 1;
                        setImageGenerationProgress(Math.round((done / totalWork) * 100));
                    }
                }
            } else {
                // single mode
                try {
                    const result = await generateImageForParagraph(p);
                    const idx = updatedStory.paragraphs.findIndex(x => x.id === p.id);
                    if (idx >= 0) {
                        updatedStory.paragraphs[idx] = {
                            ...updatedStory.paragraphs[idx],
                            image_url: result?.image_url || null,
                            image_error: !result?.image_url
                        };
                    }
                } catch (e) {
                    const idx = updatedStory.paragraphs.findIndex(x => x.id === p.id);
                    if (idx >= 0) {
                        updatedStory.paragraphs[idx] = {
                            ...updatedStory.paragraphs[idx],
                            image_error: true
                        };
                    }
                } finally {
                    done += 1;
                    setImageGenerationProgress(Math.round((done / totalWork) * 100));
                }
            }
        }

        setStory(updatedStory);
        setCurrentStep('🎨 All images generated!');
        setTimeout(() => {
            setCurrentStep('');
            setImageGenerationProgress(0);
        }, 3000);
    };

    const generateImageVariant = async (paragraphId, imagePrompt, extractorMethod) => {
        setGeneratingImages(prev => ({ ...prev, [paragraphId]: true }));
        try {
            const timeout = 360000;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);

            const res = await fetch(`${API_BASE}/generate-image`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: imagePrompt,
                    paragraph_id: paragraphId,
                    story_context: story.context,
                    experiment_id: story.experiment_id,
                    method: extractorMethod
                }),
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            const data = await res.json();
            if (data.status === 'success') return { image_url: data.image_url };
            return { image_error: true };
        } catch (e) {
            return { image_error: true };
        } finally {
            setGeneratingImages(prev => ({ ...prev, [paragraphId]: false }));
        }
    };

    const generateImageForParagraph = async (paragraph) => {
        const paragraphId = paragraph.id;
        setGeneratingImages(prev => ({ ...prev, [paragraphId]: true }));

        try {

            const timeout = 900000
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);

            const response = await fetch(`${API_BASE}/generate-image`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: paragraph.image_prompt,
                    paragraph_id: paragraphId,
                    story_context: story.context
                }),
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            const data = await response.json();

            if (data.status === 'success') {
                return { image_url: data.image_url };  // Return the URL
            } else {
                console.error(`Image generation failed: ${data.message}`);
                return { image_error: true };
            }
        } catch (error) {
            console.error(`Image generation failed for paragraph ${paragraphId}:`, error);
            return { image_error: true };
        } finally {
            setGeneratingImages(prev => ({ ...prev, [paragraphId]: false }));
        }
    };

    const getStatusIcon = () => {
        switch (serverStatus) {
            case 'running':
                return <CheckCircle size={20} color="#10b981" />;
            case 'error':
                return <AlertCircle size={20} color="#ef4444" />;
            default:
                return <Server size={20} color="#6b7280" />;
        }
    };

    return (
        <div style={styles.container}>
            <div style={styles.maxWidth}>
                {/* Header */}
                <div style={styles.header}>
                    <h1 style={styles.title}>
                        <BookOpen size={40} color="#7c3aed" />
                        AI Story Generator
                    </h1>
                    <p style={styles.subtitle}>Create illustrated stories with AI magic</p>
                </div>

                {/* Server Status & Controls */}
                <div style={styles.card}>
                    <div style={styles.flexBetween}>
                        <div style={styles.flexCenter}>
                            {getStatusIcon()}
                            <span style={{ fontWeight: '500' }}>
                                Server: {serverStatus} {modelWarmed && '• Model Ready'}
                            </span>
                        </div>

                        <button
                            onClick={warmupServer}
                            disabled={loading || modelWarmed}
                            style={{
                                ...styles.button,
                                ...styles.buttonGreen,
                                ...(loading || modelWarmed ? styles.buttonDisabled : {})
                            }}
                            onMouseOver={(e) => {
                                if (!loading && !modelWarmed) {
                                    e.target.style.backgroundColor = '#059669';
                                }
                            }}
                            onMouseOut={(e) => {
                                if (!loading && !modelWarmed) {
                                    e.target.style.backgroundColor = '#10b981';
                                }
                            }}
                        >
                            {loading && currentStep.includes('Warming') ? (
                                <Loader2 size={16} className="animate-spin" />
                            ) : (
                                <Play size={16} />
                            )}
                            {modelWarmed ? 'Model Ready' : 'Start Server'}
                        </button>
                    </div>
                    <div style={styles.flexBetween}>
                        <button style={{
                            ...styles.button,
                            ...styles.buttonPurple
                        }}
                            onClick={() => window.open("/experiments", "_blank")}>View Experiments</button>
                    </div>

                    {currentStep && (
                        <div style={styles.statusBanner}>
                            {currentStep}
                        </div>
                    )}
                </div>

                {imageGenerationProgress > 0 && imageGenerationProgress < 100 && (
                    <div style={{
                        marginTop: '12px',
                        backgroundColor: '#e0e7ff',
                        borderRadius: '8px',
                        padding: '8px',
                        textAlign: 'center'
                    }}>
                        <div style={{
                            width: `${imageGenerationProgress}%`,
                            height: '8px',
                            backgroundColor: '#4f46e5',
                            borderRadius: '4px',
                            transition: 'width 0.3s ease'
                        }} />
                        <p style={{ marginTop: '4px', fontSize: '14px', color: '#4f46e5' }}>
                            Generating images: {imageGenerationProgress}%
                        </p>
                    </div>
                )}


                {/* Story Input */}
                <div style={styles.card}>
                    <h2 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '16px' }}>
                        Create Your Story
                    </h2>

                    <div>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>
                                Story Prompt
                            </label>
                            <textarea
                                value={userPrompt}
                                onChange={(e) => setUserPrompt(e.target.value)}
                                placeholder="E.g., A magical forest where trees can talk and animals wear tiny hats..."
                                style={styles.textarea}
                                rows={3}
                                onFocus={(e) => {
                                    e.target.style.borderColor = '#7c3aed';
                                    e.target.style.boxShadow = '0 0 0 3px rgba(124, 58, 237, 0.1)';
                                }}
                                onBlur={(e) => {
                                    e.target.style.borderColor = '#d1d5db';
                                    e.target.style.boxShadow = 'none';
                                }}
                            />
                        </div>

                        <div style={styles.flexCenter}>
                            <div>
                                <label style={styles.label}>
                                    Paragraphs
                                </label>
                                <select
                                    value={numParagraphs}
                                    onChange={(e) => setNumParagraphs(parseInt(e.target.value))}
                                    style={styles.select}
                                >
                                    <option value={3}>3</option>
                                    <option value={5}>5</option>
                                    <option value={7}>7</option>
                                </select>
                            </div>
                            {/* Keyword extraction mode */}
                            <div>
                                <label style={styles.label}>Keyword Extraction Mode</label>
                                <select
                                    value={extractorMode}
                                    onChange={(e) => setExtractorMode(e.target.value)}
                                    style={styles.select}
                                >
                                    <option value="single">Single Method</option>
                                    <option value="compare">Compare All Methods</option>
                                </select>
                            </div>

                            {/* Show method selection only if in single mode */}
                            {extractorMode === 'single' && (
                                <div>
                                    <label style={styles.label}>Extraction Method</label>
                                    <select
                                        value={extractor}
                                        onChange={(e) => setExtractor(e.target.value)}
                                        style={styles.select}
                                    >
                                        <option value="yake">YAKE</option>
                                        <option value="keybert">KeyBERT</option>
                                        <option value="simple">Simple</option>
                                    </select>
                                </div>
                            )}

                            <button
                                onClick={generateStory}
                                disabled={loading || !modelWarmed || !userPrompt.trim()}
                                style={{
                                    ...styles.button,
                                    ...styles.buttonPurple,
                                    ...(loading || !modelWarmed || !userPrompt.trim() ? styles.buttonDisabled : {})
                                }}
                                onMouseOver={(e) => {
                                    if (!loading && modelWarmed && userPrompt.trim()) {
                                        e.target.style.backgroundColor = '#6d28d9';
                                    }
                                }}
                                onMouseOut={(e) => {
                                    if (!loading && modelWarmed && userPrompt.trim()) {
                                        e.target.style.backgroundColor = '#7c3aed';
                                    }
                                }}
                            >
                                {loading && currentStep.includes('crafting') ? (
                                    <Loader2 size={16} className="animate-spin" />
                                ) : (
                                    <BookOpen size={16} />
                                )}
                                Generate Story
                            </button>
                        </div>
                    </div>
                </div>

                {/* Generated Story */}
                {story && (
                    <div style={styles.card}>
                        <div style={styles.flexBetween}>
                            <h2 style={styles.storyTitle}>{story.title}</h2>
                            <div style={styles.flexCenter}>
                                <button
                                    onClick={() => generateAllImages(story.paragraphs)}
                                    style={{
                                        ...styles.button,
                                        ...styles.buttonPurple,
                                        padding: '8px 16px'
                                    }}
                                >
                                    <Image size={16} />
                                    Regenerate All Images
                                </button>
                            </div>
                        </div>

                        {story.paragraphs.map((paragraph, index) => (

                            <div
                                key={paragraph.id}
                                style={{
                                    ...styles.paragraphContainer,
                                    ...(index === story.paragraphs.length - 1 ? { borderBottom: 'none' } : {})
                                }}
                            >

                                {/* Paragraph text */}
                                <div style={styles.imageContainer}>
                                    {extractorMode === 'compare' ? (
                                        // 3-wide grid for variants
                                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', width: '100%', height: '100%' }}>
                                            {(paragraph.images || []).map((img, i) => (
                                                <div key={i} style={{ position: 'relative', backgroundColor: '#eef2ff', borderRadius: '8px', overflow: 'hidden' }}>
                                                    {generatingImages[paragraph.id] && !img.image_url ? (
                                                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '8px', color: '#6b7280' }}>
                                                            <Loader2 size={24} className="animate-spin" />
                                                            <span style={{ fontSize: '12px' }}>Generating {img.method}…</span>
                                                        </div>
                                                    ) : img.image_url ? (
                                                        <img src={img.image_url} alt={`${img.method} variant`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                                    ) : img.image_error ? (
                                                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '6px', color: '#ef4444', padding: '8px', textAlign: 'center' }}>
                                                            <AlertCircle size={18} />
                                                            <span style={{ fontSize: '12px' }}>Failed ({img.method})</span>
                                                            <button
                                                                onClick={async () => {
                                                                    const updated = { ...story };
                                                                    const pidx = updated.paragraphs.findIndex(x => x.id === paragraph.id);
                                                                    if (pidx < 0) return;
                                                                    const res = await generateImageVariant(paragraph.id, img.image_prompt);
                                                                    updated.paragraphs[pidx].images[i] = {
                                                                        ...img,
                                                                        image_url: res?.image_url || null,
                                                                        image_error: !res?.image_url
                                                                    };
                                                                    setStory(updated);
                                                                }}
                                                                style={{ ...styles.button, padding: '4px 8px', fontSize: '12px' }}
                                                            >
                                                                Retry
                                                            </button>
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
                                        // single mode (your original block, slightly guarded)
                                        generatingImages[paragraph.id] ? (
                                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', color: '#6b7280' }}>
                                                <Loader2 size={32} className="animate-spin" />
                                                <span style={{ fontSize: '14px' }}>Generating image...</span>
                                            </div>
                                        ) : paragraph.image_url ? (
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
                                                <button
                                                    onClick={() => generateImageForParagraph(paragraph)}
                                                    style={{ ...styles.button, padding: '4px 8px', fontSize: '12px' }}
                                                >
                                                    Retry
                                                </button>
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

                                {/* Image prompt (for debugging) */}
                                <details style={styles.details}>
                                    <summary style={styles.summary}>Image prompt{extractorMode === 'compare' ? 's' : ''}</summary>
                                    {extractorMode === 'compare' ? (
                                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginTop: '8px' }}>
                                            {(paragraph.images || []).map((img, i) => (
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

export default StoryGenerator;