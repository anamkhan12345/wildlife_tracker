import { useEffect, useState } from 'react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';

const API_URL = "http://192.168.0.159:8000/detections"; // replace with actual IP

function App() {
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    axios.get(API_URL)
      .then(res => setDetections(res.data))
      .catch(err => console.error(err));
  }, []);

  const chartData = {
    labels: [...new Set(detections.map(d => d.label))],
    datasets: [{
      label: "Detections",
      data: detections.reduce((acc, d) => {
        acc[d.label] = (acc[d.label] || 0) + 1;
        return acc;
      }, {}),
      backgroundColor: 'rgba(100, 150, 255, 0.6)'
    }]
  };

  return (
    <div className="p-4">
      <h1>Wildlife Detections</h1>

      <Bar data={chartData} />

      <div className="grid grid-cols-2 gap-4 mt-8">
        {detections.map((d, idx) => (
          <div key={idx} className="border p-2 shadow rounded">
            <img src={d.gcs_url} alt={d.label} className="w-full h-60 object-cover" />
            <div>
              <strong>{d.label}</strong> ({d.totalDetections})
            </div>
            <small>{d.timestamp}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;

