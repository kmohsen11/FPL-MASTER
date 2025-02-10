import { useState, useEffect } from 'react';
import './Home.css';

const Home = () => {
  const [mainTeam, setMainTeam] = useState([]);
  const [bench, setBench] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchBestSquad = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/api/best-squad');
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();

        console.log("🚀 API Response:", data); // Debugging API response

        if (!data.main || !Array.isArray(data.main) || data.main.length === 0) {
          throw new Error("⚠️ API returned empty or invalid data.");
        }

        setMainTeam(data.main);
        setBench(data.bench || []);
      } catch (error) {
        console.error('Error fetching best squad:', error);
        setError(error.message);
      }
    };

    fetchBestSquad();
  }, []);

  return (
    <div className="home-container">
      <h2>Best Squad for Next Round</h2>
      {error && <p className="error-message">{error}</p>}

      <div className="pitch">
        {/* Goalkeeper */}
        <div className="position-row goalkeepers">
          {mainTeam.filter(player => player.position.toLowerCase() === 'gk').map((player, index) => (
            <div key={index} className="player-card goalkeeper">
              <h3>🧤 {player.name}</h3>
              <p>Points: {Math.round(player.predicted_points)}</p>
            </div>
          ))}
        </div>

        {/* Defenders */}
        <div className="position-row defenders">
          {mainTeam.filter(player => player.position.toLowerCase() === 'def').map((player, index) => (
            <div key={index} className="player-card defender">
              <h3>🛡️ {player.name}</h3>
              <p>Points: {Math.round(player.predicted_points)}</p>
            </div>
          ))}
        </div>

        {/* Midfielders */}
        <div className="position-row midfielders">
          {mainTeam.filter(player => player.position.toLowerCase() === 'mid').map((player, index) => (
            <div key={index} className="player-card midfielder">
              <h3>⚡ {player.name}</h3>
              <p>Points: {Math.round(player.predicted_points)}</p>
            </div>
          ))}
        </div>

        {/* Forwards */}
        <div className="position-row forwards">
          {mainTeam.filter(player => player.position.toLowerCase() === 'fwd').map((player, index) => (
            <div key={index} className="player-card forward">
              <h3>🔥 {player.name}</h3>
              <p>Points: {Math.round(player.predicted_points)}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Bench */}
      <h3>Bench</h3>
      <div className="bench">
        {bench.map((player, index) => (
          <div key={index} className="player-card bench-player">
            <h3>🔄 {player.name}</h3>
            <p>Position: {player.position.toUpperCase()}</p>
            <p>Points: {Math.round(player.predicted_points)}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Home;
