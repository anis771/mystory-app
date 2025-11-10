// import { useEffect, useState } from "react";

// const Experiments = () => {
//   const [experiments, setExperiments] = useState([]);
//   const [selected, setSelected] = useState(null);
//   const [details, setDetails] = useState(null);
//   const API_BASE = 'http://localhost:5000/api';
//   useEffect(() => {
//     fetch(`${API_BASE}/experiments`)
//       .then(res => res.json())
//       .then(data => setExperiments(data.experiments || []));
//   }, []);

//   const loadDetails = async (id) => {
//     const res = await fetch(`${API_BASE}/experiments/${id}`);
//     const data = await res.json();
//     setSelected(id);
//     setDetails(data);
//   };

//   return (
//     <div style={{ padding: "20px" }}>
//       <h2>Experiments</h2>
//       <ul>
//         {experiments.map(exp => (
//           <li key={exp.id}>
//             <button onClick={() => loadDetails(exp.id)}>
//               {exp.id} — {exp.title}
//             </button>
//           </li>
//         ))}
//       </ul>

//       {details && (
//         <div>
//           <h3>{details.story.title}</h3>
//           {details.story.paragraphs.map(p => (
//             <div key={p.id} style={{ marginBottom: "20px" }}>
//               <p>{p.text}</p>
//               <div style={{ display: "flex", gap: "16px" }}>
//                 {["yake", "keybert", "simple"].map(method => (
//                   <div key={method} style={{ flex: 1 }}>
//                     <img src={p.results[method].image_url} style={{ width: "100%", borderRadius: "6px" }} />
//                     <p><b>{method.toUpperCase()}</b></p>
//                     <p>Prompt: {p.results[method].prompt}</p>
//                     <p>CLIP: {p.results[method].clip_score.toFixed(2)}</p>
//                   </div>
//                 ))}
//               </div>
//             </div>
//           ))}
//           <a href={`${API_BASE}/experiments/${selected}/csv`} download>
//             Download CSV
//           </a>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Experiments;
import { useEffect, useState } from "react";

const Experiments = () => {
  const [experiments, setExperiments] = useState([]);
  const [selected, setSelected] = useState(null);
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const API_BASE = 'http://localhost:5000/api';

  useEffect(() => {
    const fetchExperiments = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE}/experiments`);
        const data = await res.json();
        setExperiments(data.experiments || []);
      } catch (err) {
        setError("Failed to load experiments");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchExperiments();
  }, []);

  const loadDetails = async (id) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/experiments/${id}`);
      const data = await res.json();
      setSelected(id);
      setDetails(data);
    } catch (err) {
      setError(`Failed to load details for experiment ${id}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Get all unique methods from the data
  const getAvailableMethods = () => {
    if (!details) return [];
    const methods = new Set();
    details.story.paragraphs.forEach(p => {
      if (p.results) {
        Object.keys(p.results).forEach(method => methods.add(method));
      }
    });
    return Array.from(methods);
  };

  if (loading) return <div style={{ padding: "20px" }}>Loading...</div>;
  if (error) return <div style={{ padding: "20px", color: "red" }}>{error}</div>;

  return (
    <div style={{ padding: "20px" }}>
      <h2>Experiments</h2>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {experiments.map(exp => (
          <li key={exp.id} style={{ marginBottom: "8px" }}>
            <button 
              onClick={() => loadDetails(exp.id)}
              style={{
                background: selected === exp.id ? "#eee" : "white",
                border: "1px solid #ddd",
                padding: "8px 12px",
                borderRadius: "4px",
                cursor: "pointer",
                width: "100%",
                textAlign: "left"
              }}
            >
              {exp.id} — {exp.title}
            </button>
          </li>
        ))}
      </ul>

      {details && (
        <div style={{ marginTop: "20px" }}>
          <h3>{details.story.title}</h3>
          {details.story.paragraphs.map(p => (
            <div key={p.id} style={{ marginBottom: "40px", padding: "16px", border: "1px solid #eee", borderRadius: "8px" }}>
              <p style={{ fontSize: "1.1em", marginBottom: "16px" }}>{p.text}</p>
              <div style={{ display: "flex", gap: "16px", overflowX: "auto", paddingBottom: "16px" }}>
                {getAvailableMethods().map(method => (
                  <div key={method} style={{ flex: "0 0 300px" }}>
                    {p.results?.[method]?.image_url ? (
                      <img 
                        src={p.results[method].image_url} 
                        style={{ 
                          width: "100%", 
                          borderRadius: "6px",
                          aspectRatio: "1/1",
                          objectFit: "cover",
                          background: "#f5f5f5"
                        }} 
                        alt={`Generated for ${method}`}
                      />
                    ) : (
                      <div style={{
                        width: "100%",
                        aspectRatio: "1/1",
                        background: "#f5f5f5",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        borderRadius: "6px"
                      }}>
                        No image
                      </div>
                    )}
                    <p><b>{method.toUpperCase()}</b></p>
                    <p>Prompt: {p.results?.[method]?.prompt || "N/A"}</p>
                    <p>CLIP: {p.results?.[method]?.clip_score?.toFixed(2) ?? "N/A"}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
          <a 
            href={`${API_BASE}/experiments/${selected}/csv`} 
            download
            style={{
              display: "inline-block",
              padding: "8px 16px",
              background: "#007bff",
              color: "white",
              borderRadius: "4px",
              textDecoration: "none",
              marginTop: "16px"
            }}
          >
            Download CSV
          </a>
        </div>
      )}
    </div>
  );
};

export default Experiments;