import { useEffect, useState } from 'react';

const API_URL = "http://192.168.0.159:8000/detections";

function App() {
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    fetch(API_URL)
      .then(res => res.json())
      .then(data => setDetections(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="p-4">
      <h1>Wildlife Detections</h1>
      <div className="grid grid-cols-2 gap-4 mt-8">
        {detections.map((d, idx) => (
          <div key={idx} className="border p-2 shadow rounded">
            <img src={d.gcs_url} alt={d.label} className="w-full h-60 object-cover" />
            <div>
              <strong>{d.label}</strong> ({d.confidence})
            </div>
            <small>{new Date(d.timestamp).toLocaleString()}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
