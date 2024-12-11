import React, { useState } from "react";
import "./Players.css";

function Players() {
  const [players, setPlayers] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    setError("");
    setPlayers([]); // Clear previous results
    try {
      const response = await fetch(
        `http://127.0.0.1:5000/api/search?query=${searchTerm}`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      const data = await response.json();
      setPlayers(data.matches);
    } catch (error) {
      console.error("Error searching for players:", error);
      setError("Failed to fetch players. Please try again.");
    }
    setLoading(false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (searchTerm.trim()) {
      handleSearch();
    }
  };

  return (
    <div className="players-container">
      <h2>Search for Players</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter player name"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
        <button type="submit" className="search-button">
          Search
        </button>
      </form>

      {loading && <p className="loading-text">Searching...</p>}
      {error && <p className="error-text">{error}</p>}

      {players.length > 0 && (
        <div className="results-container">
          <h3>Search Results</h3>
          <ul>
            {players.map((player, index) => (
              <li key={index} className="player-result">
                {player.name} (Points: {Math.round(player.points)})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default Players;
