import React, { useState, useEffect } from "react";
import "./Home.css";

const Home = () => {
  const [mainTeam, setMainTeam] = useState([]);
  const [bench, setBench] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchBestSquad = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/api/best-squad");
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();

        // Separate main team and bench players
        setMainTeam(data.main);
        setBench(data.bench);
      } catch (error) {
        console.error("Error fetching best squad:", error);
        setError("Failed to fetch the best squad. Please try again.");
      }
    };

    fetchBestSquad();
  }, []);

  return (
    <div className="home-container">
      <h2>Best Squad for Next Round</h2>
      {error && <p className="error-message">{error}</p>}

      {/* Team pitch layout */}
      <div className="pitch">
        <div className="position-row goalkeepers">
          {mainTeam
            .filter((player) => player.position === "GK")
            .map((player, index) => (
              <div key={index} className="player-card">
                <h3>{player.name}</h3>
                <p>Points: {Math.round(player.predicted_points)}</p>
              </div>
            ))}
        </div>

        <div className="position-row defenders">
          {mainTeam
            .filter((player) => player.position === "DEF")
            .map((player, index) => (
              <div key={index} className="player-card">
                <h3>{player.name}</h3>
                <p>Points: {Math.round(player.predicted_points)}</p>
              </div>
            ))}
        </div>

        <div className="position-row midfielders">
          {mainTeam
            .filter((player) => player.position === "MID")
            .map((player, index) => (
              <div key={index} className="player-card">
                <h3>{player.name}</h3>
                <p>Points: {Math.round(player.predicted_points)}</p>
              </div>
            ))}
        </div>

        <div className="position-row forwards">
          {mainTeam
            .filter((player) => player.position === "FWD")
            .map((player, index) => (
              <div key={index} className="player-card">
                <h3>{player.name}</h3>
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
            <h3>{player.name}</h3>
            <p>Position: {player.position}</p>
            <p>Points: {Math.round(player.predicted_points)}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Home;
